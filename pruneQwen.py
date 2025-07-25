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
from method import *
from wrapper import *

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

def prepare_calib_inputs(args, model, tokenizer, pixel_data, generation_config, prompt):
    llm_layers = model.language_model.model.layers
    
    captured_data = {}

    class GenerationCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_type = module.attention_type
            self.sample_idx = -1

        def forward(self, inp, **kwargs):
            if self.sample_idx not in captured_data:
                captured_data[self.sample_idx] = {'prefill': None, 'generate': []}
                cache = copy.deepcopy(kwargs)
                cache['hidden_states'] = inp
                captured_data[self.sample_idx]['prefill'] = cache
                if args.hook_type == 'prefill':
                    raise ValueError("Hooked and stopped forward for prefill")
            else:
                cache = copy.deepcopy(kwargs)
                cache['hidden_states'] = inp
                captured_data[self.sample_idx]['generate'].append(cache)

            return self.module(inp, **kwargs)

    original_layer = llm_layers[0]
    catcher_module = GenerationCatcher(original_layer)
    catcher_module.mlp = original_layer.mlp
    llm_layers[0] = catcher_module

    act = {}
    def hook_act(_, input, output):
        if catcher_module.sample_idx not in act:
            act[catcher_module.sample_idx] = defaultdict(list)
        act[catcher_module.sample_idx]['input'].append(input)
        act[catcher_module.sample_idx]['output'].append(output)
    catcher_module.mlp.down_proj.register_forward_hook(hook_act)

    for i, pixel_values in enumerate(pixel_data):
        catcher_module.sample_idx = i 
        try:
            model.chat(tokenizer, pixel_values, prompt, generation_config)
        except:
            pass

    llm_layers[0] = original_layer

    return captured_data

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
        inputs = prepare_calib_inputs(args, model, tokenizer, pixel_data, generation_config, prompt)

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
            wrapped_layers[i][sub_layer] = WrappedLayer(sub_layers[sub_layer], i, sub_layer)

    def hook_io(i, sub_layer):
        def tmp(_, input, output):
            if i == 0 and sub_layer == 'mlp.down_proj':
                print(f"Hook called for layer {i}/{sub_layer}. Input shape: {input[0].shape}")
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
            wrapped_layers[i]['mlp.down_proj'].prepare_for_kde()
            entropy[i] = wrapped_layers[i]['mlp.down_proj'].calculate_entropy_kde_total()
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
                if 'up' in sub_layer or 'gate' in sub_layer:
                    sub_layers[sub_layer].weight.data[W_mask,:]=0
                elif 'down' in sub_layer:
                    sub_layers[sub_layer].weight.data[:,W_mask]=0
                else:
                    raise ValueError

            keep_indices.append(sort_res.indices[int(imp.shape[0]*(args.prune_ratio/10)):])

            del imp, weights, x_norm_l2, sort_res, indice, W_mask
    
    torch.cuda.empty_cache()

    return keep_indices, prune_indices

def _prune_sequential(args, layers, inputs):
    entropy = None
    keep_indices = []
    prune_indices = []

    for i in range(len(layers)):
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
                    _ = layer(**inputs[s]['prefill'])[0]
                elif args.hook_type == 'generate':
                    # n = len(inputs[s]['generate'])
                    _ = layer(**inputs[s]['generate'][0])[0]

        for h in handles:
            h.remove()

        if args.method in ['entropy', 'magent', 'esparse']:
            if args.hook_type == 'prefill':
                wrapped_layers['mlp.down_proj'].prepare_for_hist()
            elif args.hook_type == 'generate':
                wrapped_layers['mlp.down_proj'].prepare_for_kde()
            handles = []
            handles.append(sub_layers['mlp.down_proj'].register_forward_hook(hook_io('mlp.down_proj')))
            with torch.no_grad():
                for s in tqdm(range(len(inputs))):
                    _ = layer(**inputs[s])[0]
            for h in handles:
                h.remove()
            if args.hook_type == 'prefill':
                entropy = wrapped_layers['mlp.down_proj'].calculate_entropy()
            elif args.hook_type == 'generate':
                entropy = wrapped_layers['mlp.down_proj'].calculate_entropy_kde()

        imp = 0
        wanda = 0
        if args.structure_prune:
            for sub_layer in sub_layers:
                weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                x_norm_l2 = torch.sqrt(wrapped_layers[sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)
                wanda += get_importance('group_wanda', sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                
                if args.method == 'entropy':
                    imp = get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                    continue
                else:
                    imp += get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10) # (1, in)  

            wanda = norm_value(wanda)

            if args.method == 'magent':
                imp = (1-args.alpha/10) * wanda + args.alpha/10 * norm_value(entropy[i])                

            W_mask = (torch.zeros_like(imp) == 1)

            sort_res = torch.sort(imp, dim=0, stable=True)    # sort along row
            indice = sort_res.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
            prune_indices.append(indice)
            for sub_layer in sub_layers:
                # print(f'Prunning {i}.{sub_layer}') 
                W_mask.scatter_(0, indice, True)
                if 'up' in sub_layer or 'gate' in sub_layer:
                    sub_layers[sub_layer].weight.data[W_mask,:]=0
                elif 'down' in sub_layer:
                    sub_layers[sub_layer].weight.data[:,W_mask]=0
                else:
                    raise ValueError

            keep_indices.append(sort_res.indices[int(imp.shape[0]*(args.prune_ratio/10)):])

            del imp, weights, x_norm_l2, sort_res, indice, W_mask

            with torch.no_grad():
                for j in range(len(inputs)):
                    if args.hook_type == 'prefill':
                        inputs[j]['hidden_states'] = layer(**inputs[j]['prefill'])[0]
                    elif args.hook_type == 'generate':
                        inputs[j]['hidden_states'][-1] = layer(**inputs[j]['generate'][-1])[0]
        else:
            for sub_layer in sub_layers:
                weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                x_norm_l2 = torch.sqrt(wrapped_layers[sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)
                wanda += get_importance('group_wanda', sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                
                if args.method == 'entropy':
                    imp = get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                    continue
                else:
                    imp += get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10) # (1, in)  

            wanda = norm_value(wanda)

            if args.method == 'magent':
                imp = (1-args.alpha/10) * wanda + args.alpha/10 * norm_value(entropy[i])                

            W_mask = (torch.zeros_like(imp) == 1)

            sort_res = torch.sort(imp, dim=0, stable=True)    # sort along row
            indice = sort_res.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
            prune_indices.append(indice)
            for sub_layer in sub_layers:
                # print(f'Prunning {i}.{sub_layer}') 
                W_mask.scatter_(0, indice, True)
                if 'up' in sub_layer or 'gate' in sub_layer:
                    sub_layers[sub_layer].weight.data[W_mask,:]=0
                elif 'down' in sub_layer:
                    sub_layers[sub_layer].weight.data[:,W_mask]=0
                else:
                    raise ValueError

            keep_indices.append(sort_res.indices[int(imp.shape[0]*(args.prune_ratio/10)):])

            del imp, weights, x_norm_l2, sort_res, indice, W_mask

            with torch.no_grad():
                for j in range(len(inputs)):
                    if args.hook_type == 'prefill':
                        inputs[j]['hidden_states'] = layer(**inputs[j]['prefill'])[0]
                    elif args.hook_type == 'generate':
                        inputs[j]['hidden_states'][-1] = layer(**inputs[j]['generate'][-1])[0]
    
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
            if isinstance(module, nn.Linear) and 'down' in name:
                all_down_linears.append((f"layer.{i}.{name}", module))
            elif isinstance(module, nn.Linear) and  'up' in name:
                all_up_linears.append((f"layer.{i}.{name}", module))
            elif isinstance(module, nn.Linear) and  'gate' in name:
                all_gate_linears.append((f"layer.{i}.{name}", module))
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
            model.language_model.model.layers[i_linear].mlp.down_proj = new_down_linear
            model.language_model.model.layers[i_linear].mlp.gate_proj = new_gate_linear
            model.language_model.model.layers[i_linear].mlp.up_proj = new_up_linear

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
    model, tokenizer = load_model_tokenizer(f'/data/zige.wang/deploy/Tagbar_2B_ver_20250619FVOB')
    prompt = load_prompt('/data/template.yaml', prompt_type)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    img_path = '/data/Dataset/filtered/tagbar'

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.hook_type = 'prefill'
    args.hook_type = 'generate'
    # args.prune_type = 'sequential'
    args.prune_type = 'parrallel'
    args.method = 'magent'
    args.structure_prune = True
    args.prune_ratio = 3
    args.nsamples = 30
    args.alpha = 9
    args.plot_wanda_ent = False

    assert args.method in ['weight', 'wanda', 'entropy', 'esparse', 'magent', 'group_wanda', 'test']

    print(f'>>> Running: prune_ratio={args.prune_ratio}0%, method={args.method}, nsamples={args.nsamples}, entropy_ratio={args.alpha}0%')

    save_path = f'/data/prune/3/{args.method}_{args.hook_type}_{args.prune_type}_r{args.prune_ratio}_a{args.alpha}'
    
    keep_indices, prune_indices = prune(args, model, tokenizer, generation_config, img_path, prompt)
    # torch.save(torch.stack(prune_indices), f'./pt/{args.method}_{args.nsamples}_{args.prune_ratio}.pt')

    prune_model = apply_channel_prune(model, keep_indices)

    prune_model_size = get_model_size(prune_model, count_nonzero_only=True)
    print(f"Prune model has size={prune_model_size/MiB:.2f} MiB")
    prune_model_params = get_num_parameters(prune_model, count_nonzero_only=True)
    print(f"Prune model has {prune_model_params/1e9:.2f}B parameters")

    copy_all_files('/data/base_model/base2B', dst_dir=save_path)
    prune_model.save_pretrained(save_path)
    print(f"Pruned model saved to {save_path}")

if __name__ == "__main__":
    debug()
    # main()



