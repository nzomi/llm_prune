import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import random
import argparse
from icecream import ic
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=FutureWarning)

from utils import *
from loader import *
from method9B import *
from wrapper9B import *

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_random_images(img_path, img_num, seed=None):
    all_images = os.listdir(img_path)
    if seed is not None:
        random.seed(seed)  
    random.shuffle(all_images)
    return all_images[:img_num]

def get_pixel_data(img_path, nsamples):
    pixel_data = []
    image_list = get_random_images(img_path, nsamples, seed=42)
    for img in image_list:
        if img.endswith(('.png', '.jpg', '.jpeg')):
            # print(f"Processing {img}")
            pixel_values = load_image(os.path.join(img_path, img)).to(torch.bfloat16).cuda()
            pixel_data.append(pixel_values)
    return pixel_data

def prepare_calib_inputs(model, tokenizer, pixel_data, generation_config, prompt):
    llm_layers = model.language_model.model.layers
    cache = {'i':0, 'position_ids':None, 'attention_mask':None, 'position_embeddings':None}
    inputs = [None]*len(pixel_data)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_type = module.attention_type
        def forward(self, inp, **kwargs):
            cache['hidden_states'] = inp
            cache['position_ids'] = kwargs['position_ids']
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_embeddings'] = kwargs['position_embeddings']
            inputs[cache['i']] = copy.deepcopy(cache)
            cache['i'] += 1
            raise ValueError("Hooked and stopped forward")

    llm_layers[0] = Catcher(llm_layers[0]) # wrap for hook first llm_layer input

    for pixel_values in pixel_data:
        try:
            model.chat(tokenizer, pixel_values, prompt, generation_config)
        except ValueError:
            continue  

    llm_layers[0] = llm_layers[0].module  # turn to original layer

    return inputs

def prune(args, model, tokenizer, generation_config, img_path, prompt):
    pixel_data = get_pixel_data(img_path, args.nsamples)
    if args.prune_type == 'sequential':
        return prune_sequential(args, model, tokenizer, pixel_data, generation_config, prompt)
    elif args.prune_type == 'parrallel':
        return _prune_parrallel(args, model, tokenizer, pixel_data, generation_config, prompt)
    else:
        raise ValueError
            
def prune_sequential(args, model, tokenizer, pixel_data, generation_config, prompt):
    with torch.no_grad():
        inputs = prepare_calib_inputs(model, tokenizer, pixel_data, generation_config, prompt)
    model.config.llm_config.use_cache = False
    llm_layer = model.language_model.model.layers
    _prune_sequential(args, llm_layer, inputs)

def _prune_parrallel(args, model, tokenizer, pixel_data, generation_config, prompt):
    keep_indices = []
    prune_indices = []
    llm_layers = model.language_model.model.layers
    wrapped_layers = defaultdict(dict)
    for i, layers in enumerate(llm_layers):
        sub_layers = find_sub_layers(layers)
        for sub_layer in sub_layers:
            wrapped_layers[i][sub_layer] = WrappedLayer(sub_layers[sub_layer], i, sub_layer, kde_samples=args.kde_nsamples)

    def hook_io(i, sub_layer):
        def tmp(_, input, output):
            # if i == 0 and sub_layer == 'mlp.down_proj':
                # print(f"Hook called for layer {i}/{sub_layer}. Input shape: {input[0].shape}")
            if input[0].shape[1] == 1:
                wrapped_layers[i][sub_layer].add_batch(input[0].data, output.data)
        return tmp

    handles = []

    for i, layers in enumerate(llm_layers):
        sub_layers = find_sub_layers(layers)
        for sub_layer in sub_layers:
            handles.append(sub_layers[sub_layer].register_forward_hook(hook_io(i, sub_layer)))

    with torch.no_grad():
        for pixel_values in tqdm(pixel_data):
            try:
                model.chat(tokenizer, pixel_values, prompt, generation_config)
            # except Exception as e:
            #         print(f"An error occurred: {e}")
            #         import traceback
            #         traceback.print_exc()
            #         continue
            except:
                continue

    for h in handles:
        h.remove()

    entropy = [None] * len(llm_layers)
    if args.method in ['entropy', 'magent', 'esparse', 'test']:
        for i, layers in tqdm(enumerate(llm_layers)):
            wrapped_layers[i]['feed_forward.w2'].prepare_for_kde()
            entropy[i] = wrapped_layers[i]['feed_forward.w2'].calculate_entropy_kde_total()
            # entropy[i] = wrapped_layers[i]['mlp.down_proj'].calculate_entropy_kde()

    if args.structure_prune:
        for i, layers in tqdm(enumerate(llm_layers)):
            sub_layers = find_sub_layers(layers)
            imp = 0
            wanda = 0
            for sub_layer in sub_layers:
                weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                x_norm_l2 = torch.sqrt(wrapped_layers[i][sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)
                wanda += get_importance('group_wanda', sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                
                if args.method == 'entropy':
                    imp = get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                    continue
                else:
                    imp += get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10) # (1, in)  

            # wanda = norm_value(wanda)

            if args.method == 'magent':
                imp = (1-args.alpha/10) * wanda + args.alpha/10 * norm_value(entropy[i])            

            if args.method == 'test':
                imp = wanda * norm_value(entropy[i])    

            W_mask = (torch.zeros_like(imp) == 1)

            sort_res = torch.sort(imp, dim=0, stable=True)    # sort along row
            indice = sort_res.indices[:int(imp.shape[0]*(args.prune_ratio/10))]

            if args.plot_wanda_ent and args.method == 'magent':
                sort_wanda = torch.sort(wanda, dim=0, stable=True)
                sort_ent = torch.sort(norm_value(entropy[i]), dim=0, stable=True)
                indice_wanda = sort_wanda.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
                indice_ent = sort_ent.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
                plot_wanda_ent(wanda, norm_value(entropy[i]), layer_idx=i, wanda_idx=indice_wanda, ent_idx=indice_ent, mag_ent_idx=indice, pr=args.prune_ratio, a=args.alpha)

            prune_indices.append(indice)
            for sub_layer in sub_layers:
                # print(f'Prunning {i}.{sub_layer}') 
                W_mask.scatter_(0, indice, True)
                if 'w3' in sub_layer or 'w1' in sub_layer:
                    sub_layers[sub_layer].weight.data[W_mask,:]=0
                elif 'w2' in sub_layer:
                    sub_layers[sub_layer].weight.data[:,W_mask]=0
                else:
                    raise ValueError

            keep_indices.append(sort_res.indices[int(imp.shape[0]*(args.prune_ratio/10)):])

            del imp, weights, x_norm_l2, sort_res, indice, W_mask
    else:
        for i, layers in tqdm(enumerate(llm_layers)):
            sub_layers = find_sub_layers(layers)
            imp = 0
            wanda = 0
            for sub_layer in sub_layers:
                weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                x_norm_l2 = torch.sqrt(wrapped_layers[i][sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)
                wanda = weights * x_norm_l2  

                W_mask = (torch.zeros_like(wanda) == 1)
                sort_res = torch.sort(wanda, dim=-1, stable=True)
                indices = sort_res[1][:,:int(wanda.shape[1]*(args.prune_ratio/10))]
                W_mask.scatter_(1, indices, True)

                sub_layers[sub_layer].weight.data[W_mask]=0
    
    torch.cuda.empty_cache()

    return keep_indices, prune_indices

def _prune_sequential(args, layers, inputs):
    keep_indices = []
    prune_indices = []

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        sub_layers = find_sub_layers(layer)

        wrapped_layers = {}
        for sub_layer in sub_layers:
            wrapped_layers[sub_layer] = WrappedLayer(sub_layers[sub_layer], total_sample=args.nsamples)

        def hook_io(sub_layer):
            def tmp(_, input, output):
                wrapped_layers[sub_layer].add_batch(input[0].data, output.data)
            return tmp

        handles = []

        for sub_layer in wrapped_layers:
            handles.append(sub_layers[sub_layer].register_forward_hook(hook_io(sub_layer)))

        with torch.no_grad():
            for s in range(len(inputs)):
                if args.hook_type == 'prefill':
                    inputs[s].pop('i', None)
                    _ = layer(**inputs[s])[0]

        for h in handles:
            h.remove()

        if args.structure_prune:
            pass
        else:
            if args.method == 'wanda':
                for sub_layer in sub_layers:
                    weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                    x_norm_l2 = torch.sqrt(wrapped_layers[sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)
                    wanda = weights * x_norm_l2  

                    W_mask = (torch.zeros_like(wanda) == 1)
                    sort_res = torch.sort(wanda, dim=-1, stable=True)
                    indices = sort_res[1][:,:int(wanda.shape[1]*(args.prune_ratio/10))]
                    W_mask.scatter_(1, indices, True)

                    sub_layers[sub_layer].weight.data[W_mask]=0
            elif args.method == 'group_wanda':
                wanda = 0
                for sub_layer in sub_layers:
                    weight = torch.abs(sub_layers[sub_layer].weight.data)
                    act = torch.sqrt(wrapped_layers[sub_layer].x_norm_l2.reshape((1,-1)))
                    if 'up' in sub_layer or 'gate' in sub_layer:
                        wanda += act.t()*weight.t()
                    elif 'down' in sub_layer: 
                        wanda += weight*act
                    
                W_metric = wanda.sum(dim=0)
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

                sort_res = torch.sort(W_metric, dim=0, stable=True)
                # unstructured pruning
                indices = sort_res.indices[:int(W_metric.shape[0]*(args.prune_ratio/10))]
                for sub_layer in sub_layers:
                    W_mask.scatter_(0, indices, True)
                    if 'up' in sub_layer or 'gate' in sub_layer:
                        sub_layers[sub_layer].weight.data[W_mask,:]=0
                    elif 'down' in sub_layer:
                        sub_layers[sub_layer].weight.data[:,W_mask]=0

            with torch.no_grad():
                for j in range(len(inputs)):
                    if args.hook_type == 'prefill':
                        inputs[j]['hidden_states'] = layer(**inputs[j])[0]
    
    torch.cuda.empty_cache()

    return keep_indices, prune_indices

def prune_linear_channel(linear_layer, selected_idx, prune_channel):
    if prune_channel == 1:
        in_features = selected_idx.numel()
        out_features = linear_layer.out_features
    else:
        out_features = selected_idx.numel()
        in_features = linear_layer.in_features
    bias = linear_layer.bias is not None

    new_linear = nn.Linear(in_features, out_features, bias=bias)
    
    new_linear.weight.data.copy_(
        torch.index_select(linear_layer.weight.data, prune_channel, selected_idx)
    )

    if bias:
        new_linear.bias.data.copy_(linear_layer.bias.data)

    return new_linear

@torch.no_grad()
def apply_channel_prune(model, idx):
    model = copy.deepcopy(model)  
    all_down_linears = []
    all_up_linears = []
    all_gate_linears = []
    for i, decoder_layer in enumerate(model.language_model.model.layers):
        for name, module in decoder_layer.named_modules():
            if isinstance(module, nn.Linear) and 'w2' in name:
                all_down_linears.append((f"feed_forward.{i}.{name}", module))
            elif isinstance(module, nn.Linear) and  'w1' in name:
                all_up_linears.append((f"feed_forward.{i}.{name}", module))
            elif isinstance(module, nn.Linear) and  'w3' in name:
                all_gate_linears.append((f"feed_forward.{i}.{name}", module))
            else:
                continue

    with torch.no_grad():
        for i_linear in range(len(all_down_linears)):
            # sort_idx = torch.argsort(importance, descending=True)[:k]
            # topk = torch.topk(importance, k)
            cur_down_linear = all_down_linears[i_linear][1]
            cur_gate_linear = all_gate_linears[i_linear][1]
            cur_up_linear = all_up_linears[i_linear][1]
            new_down_linear = prune_linear_channel(cur_down_linear, idx[i_linear].to(model.device), 1)
            new_gate_linear = prune_linear_channel(cur_gate_linear, idx[i_linear].to(model.device), 0)
            new_up_linear = prune_linear_channel(cur_up_linear, idx[i_linear].to(model.device), 0)
            model.language_model.model.layers[i_linear].feed_forward.w2 = new_down_linear
            model.language_model.model.layers[i_linear].feed_forward.w3 = new_gate_linear
            model.language_model.model.layers[i_linear].feed_forward.w1 = new_up_linear

    return model

def copy_all_files(src_dir, dst_dir):
    """
    Copy all files from src_dir to dst_dir.

    Args:
        src_dir (str): Path to the source directory.
        dst_dir (str): Path to the destination directory.
    """
    os.makedirs(dst_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            # print(f"Copied {src_file} → {dst_file}")

def main():
    prompt_type = 'base'
    model, tokenizer = load_model_tokenizer(f'/data/zige.wang/deploy/Tagbar_2B_ver_20250619FVOB')
    prompt = load_prompt('/data/template.yaml', prompt_type)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    img_path = '/data/Dataset/filtered/tagbar'

    parser = argparse.ArgumentParser()
    parser.add_argument('--hook_type', type=str, default='prefill')
    parser.add_argument('--prune_type', type=str, default='sequential')
    parser.add_argument('--method', type=str, default='wanda', choices=['wanda', 'weight', 'esparse', 'entropy', 'magent', 'group_wanda', 'norm_group_wanda'])
    parser.add_argument('--prune_ratio', type=int, default=1)
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    
    args.structure_prune = True
    args.plot_wanda_ent = True

    keep_indices, prune_indices = prune(args, model, tokenizer, generation_config, img_path, prompt)
    prune_model = apply_channel_prune(model, keep_indices)

    prune_model_size = get_model_size(prune_model, count_nonzero_only=True)
    print(f"Prune model has size={prune_model_size/MiB:.2f} MiB")
    prune_model_params = get_num_parameters(prune_model, count_nonzero_only=True)
    print(f"Prune model has {prune_model_params/1e9:.2f}B parameters")

    copy_all_files('/data/base_model/base2B', dst_dir=args.save_path)
    prune_model.save_pretrained(args.save_path)
    print(f"Pruned model saved to {args.save_path}")

def debug():
    prompt_type = 'base'
    # model, tokenizer = load_model_tokenizer(f'/data/zige.wang/deploy/Tagbar_2B_ver_20250619FVOB')
    model, tokenizer = load_model_tokenizer(f'/data/zige.wang/deploy/InternVL3_9B_ver_20250610FVOB')
    prompt = load_prompt('/data/template.yaml', prompt_type)
    generation_config = dict(max_new_tokens=256, do_sample=False)
    img_path = '/data/Dataset/filtered/tagbar'

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.hook_type = 'prefill'
    args.hook_type = 'generate'
    args.prune_type = 'sequential'
    args.prune_type = 'parrallel'
    args.method = 'entropy'
    args.structure_prune = True
    args.prune_ratio = 5
    args.nsamples = 32
    args.alpha = 9
    args.plot_wanda_ent = False
    args.kde_nsamples = 40

    assert args.method in ['weight', 'wanda', 'entropy', 'esparse', 'magent', 'group_wanda', 'test']

    print(f'>>> Running: prune_ratio={args.prune_ratio}0%, method={args.method}, nsamples={args.nsamples}, entropy_ratio={args.alpha}0%')

    if not args.structure_prune:
        save_path = f'/data/prune/exp/{args.method}_{args.hook_type}_{args.prune_type}_r{args.prune_ratio}_uns'
        prune(args, model, tokenizer, generation_config, img_path, prompt)
        
    else:
        # save_path = f'/data/prune/sample/{args.method}_{args.hook_type}_{args.prune_type}_r{args.prune_ratio}_a{args.alpha}_s{args.nsamples}'
        save_path = f'/data/prune/9B/{args.method}_{args.hook_type}_{args.prune_type}_r{args.prune_ratio}_a{args.alpha}_s{args.nsamples}_ka'
        keep_indices, prune_indices = prune(args, model, tokenizer, generation_config, img_path, prompt)
        prune_model = apply_channel_prune(model, keep_indices)
    # torch.save(torch.stack(prune_indices), f'./pt/{args.method}_{args.nsamples}_{args.prune_ratio}.pt')

   

    prune_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"Prune model has size={prune_model_size/MiB:.2f} MiB")
    prune_model_params = get_num_parameters(model, count_nonzero_only=True)
    print(f"Prune model has {prune_model_params/1e9:.2f}B parameters")

    copy_all_files('/data/base_model/base9B', dst_dir=save_path)
    if not args.structure_prune:
        model.save_pretrained(save_path)
    else:
        prune_model.save_pretrained(save_path)
        ic(prune_model.language_model.model.layers[0].feed_forward)
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    debug()
    # main()



