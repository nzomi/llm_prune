import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import shutil
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import random
import argparse
from icecream import ic
import warnings
import json
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

def get_llm_layers(model, model_type='vlm'):
    """根据模型类型获取正确的模型层"""
    if model_type == 'llm':
        return model.model.layers
    else:  # vlm
        return model.language_model.model.layers

def get_llm_data(data_path, nsamples):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    random.shuffle(data)
    return [item['instruction'] for item in data[:nsamples]]

def prepare_calib_inputs(model, tokenizer, data, generation_config, prompt, model_type='vlm'):
    llm_layers = get_llm_layers(model, model_type)
    cache = {'i':0, 'position_ids':None, 'attention_mask':None, 'position_embeddings':None}
    inputs = [None]*len(data)

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

    llm_layers[0] = Catcher(llm_layers[0])

    for item in data:
        try:
            if model_type == 'vlm':
                pixel_values = item
                if hasattr(model, 'chat'):
                    model.chat(tokenizer, pixel_values, prompt, generation_config)
                else:
                    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
                    model.generate(input_ids, images=pixel_values, **generation_config)
            else:
                text_prompt = item
                if hasattr(model, 'chat'):
                    model.chat(tokenizer, None, text_prompt, generation_config)
                else:
                    input_ids = tokenizer(text_prompt, return_tensors='pt').input_ids.to(model.device)
                    model.generate(input_ids, **generation_config)
        except ValueError:
            continue  

    llm_layers[0] = llm_layers[0].module

    return inputs

def prune(args, model, tokenizer, generation_config, data_path, prompt):
    if args.model_type == 'vlm':
        data = get_pixel_data(data_path, args.nsamples)
    else:
        data = get_llm_data(data_path, args.nsamples)
    if args.hook_type == 'prefill':
        return prune_prefill(args, model, tokenizer, data, generation_config, prompt)
    elif args.hook_type == 'generate':
        return prune_generate(args, model, tokenizer, data, generation_config, prompt)
    else:
        raise ValueError
            
def prune_prefill(args, model, tokenizer, data, generation_config, prompt):
    with torch.no_grad():
        inputs = prepare_calib_inputs(model, tokenizer, data, generation_config, prompt, args.model_type)

    llm_layer = get_llm_layers(model, args.model_type)
    if args.prune_type == 'sequential':
        return prune_sequential(args, llm_layer, inputs)
    else:
        return prune_parrallel(args, llm_layer, inputs)

def prune_generate(args, model, tokenizer, data, generation_config, prompt):
    keep_indices = []
    prune_indices = []
    llm_layers = get_llm_layers(model, args.model_type)
    wrapped_layers = defaultdict(dict)
    for i, layers in enumerate(llm_layers):
        sub_layers = find_sub_layers(layers)
        for sub_layer in sub_layers:
            wrapped_layers[i][sub_layer] = WrappedLayer(sub_layers[sub_layer], i, sub_layer)

    def hook_io(i, sub_layer):
        def tmp(_, input, output):
            if input[0].shape[1] == 1:
                wrapped_layers[i][sub_layer].add_batch(input[0].data, output.data)
        return tmp

    handles = []

    for i, layers in enumerate(llm_layers):
        sub_layers = find_sub_layers(layers)
        for sub_layer in sub_layers:
            handles.append(sub_layers[sub_layer].register_forward_hook(hook_io(i, sub_layer)))

    with torch.no_grad():
        for item in tqdm(data):
            try:
                if args.model_type == 'vlm':
                    pixel_values = item
                    if hasattr(model, 'chat'):
                        model.chat(tokenizer, pixel_values, prompt, generation_config)
                    else:
                        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
                        model.generate(input_ids, images=pixel_values, **generation_config)
                else:
                    text_prompt = item
                    if hasattr(model, 'chat'):
                        model.chat(tokenizer, None, text_prompt, generation_config)
                    else:
                        messages = [
                                    {"role": "user", "content": text_prompt}
                                ]
                        text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                        # input_ids = tokenizer(text_prompt, return_tensors='pt').input_ids.to(model.device)
                        model.generate(**model_inputs, max_new_tokens=1024)
            except:
                continue

    for h in handles:
        h.remove()


    if args.structure_prune:
        entropy = [None] * len(llm_layers)
        if args.method in ['entropy', 'magent', 'esparse']:
            for i, layers in tqdm(enumerate(llm_layers)):
                wrapped_layers[i]['mlp.down_proj'].prepare_for_kde()
                entropy[i] = wrapped_layers[i]['mlp.down_proj'].calculate_entropy_kde()
        if args.wanda_mode == 'global':
            all_wandas = []
            for i, layers in tqdm(enumerate(llm_layers)):
                sub_layers = find_sub_layers(layers)
                layer_wanda = 0
                for sub_layer in sub_layers:
                    weights = torch.abs(sub_layers[sub_layer].weight.data)
                    x_norm_l2 = torch.sqrt(wrapped_layers[i][sub_layer].x_norm_l2.reshape((1,-1)))
                    layer_wanda += get_importance('group_wanda', sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                all_wandas.append(layer_wanda)
            concatenated_wandas = torch.cat(all_wandas, dim=0)
            normalized_wandas = norm_value(concatenated_wandas)
            split_normalized_wandas = torch.split(normalized_wandas, [w.shape[0] for w in all_wandas], dim=0)
            global_wandas = split_normalized_wandas
        for i, layers in tqdm(enumerate(llm_layers)):
            sub_layers = find_sub_layers(layers)
            imp = 0
            wanda = 0
            if args.wanda_mode == 'global':
                wanda = global_wandas[i]
            else:
                for sub_layer in sub_layers:
                    weights = torch.abs(sub_layers[sub_layer].weight.data)
                    x_norm_l2 = torch.sqrt(wrapped_layers[i][sub_layer].x_norm_l2.reshape((1,-1)))
                    wanda += get_importance('group_wanda', sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                wanda = norm_value(wanda)
            for sub_layer in sub_layers:
                weights = torch.abs(sub_layers[sub_layer].weight.data)
                x_norm_l2 = torch.sqrt(wrapped_layers[i][sub_layer].x_norm_l2.reshape((1,-1)))
                if args.method == 'entropy':
                    imp = get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
                    continue
                else:
                    imp += get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], alpha=args.alpha/10)
            if args.method == 'magent':
                imp = (1-args.alpha/10) * wanda + args.alpha/10 * norm_value(entropy[i])
            W_mask = (torch.zeros_like(imp) == 1)
            sort_res = torch.sort(imp, dim=0, stable=True)
            indice = sort_res.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
            if args.plot_wanda_ent and args.method == 'magent':
                sort_wanda = torch.sort(wanda, dim=0, stable=True)
                sort_ent = torch.sort(norm_value(entropy[i]), dim=0, stable=True)
                indice_wanda = sort_wanda.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
                indice_ent = sort_ent.indices[:int(imp.shape[0]*(args.prune_ratio/10))]
                plot_wanda_ent(wanda, norm_value(entropy[i]), layer_idx=i, wanda_idx=indice_wanda, ent_idx=indice_ent, mag_ent_idx=indice, pr=args.prune_ratio, a=args.alpha)
            prune_indices.append(indice)
            for sub_layer in sub_layers:
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

def prune_parrallel(args, layers, inputs):
    keep_indices = []
    for i in range(len(layers)):
        layer = layers[i]
        sub_layers = find_sub_layers(layer)

        wrapped_layers = {}
        for sub_layer in sub_layers:
            wrapped_layers[sub_layer] = WrappedLayer(sub_layers[sub_layer])

        def hook_io(sub_layer):
            def tmp(_, input, output):
                wrapped_layers[sub_layer].get_data(input[0].data, output.data)
            return tmp

        handles = []

        for sub_layer in wrapped_layers:
            handles.append(sub_layers[sub_layer].register_forward_hook(hook_io(sub_layer)))

        with torch.no_grad():
            for s in range(len(inputs)):
                inputs[s].pop('i', None)
                _ = layer(**inputs[s])[0]

        for h in handles:
            h.remove()

        if args.get_entropy:
            pass

        imp = 0
        if args.structure_prune:
            for sub_layer in sub_layers:
                weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                x_norm_l2 = torch.sqrt(wrapped_layers[sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)

                imp += get_importance(args.method, weights, x_norm_l2, sub_layer) # (out, in)

            sort_res = torch.sort(imp, dim=0, stable=True)    # sort along row

            keep_indices.append(sort_res.indices[int(imp.shape[0]*(args.prune_ratio/10)):])

            del imp, weights, x_norm_l2, sort_res
    
    torch.cuda.empty_cache()

    return keep_indices

def prune_sequential(args, layers, inputs):
    entropy = None
    keep_indices = []
    prune_indices = []

    for i in range(len(layers)):
        layer = layers[i]
        sub_layers = find_sub_layers(layer)

        wrapped_layers = {}
        for sub_layer in sub_layers:
            wrapped_layers[sub_layer] = WrappedLayer(sub_layers[sub_layer])

        def hook_io(sub_layer):
            def tmp(_, input, output):
                wrapped_layers[sub_layer].add_batch(input[0].data, output.data)
            return tmp

        handles = []

        for sub_layer in wrapped_layers:
            handles.append(sub_layers[sub_layer].register_forward_hook(hook_io(sub_layer)))

        with torch.no_grad():
            for s in range(len(inputs)):
                inputs[s].pop('i', None)
                _ = layer(**inputs[s])[0]

        for h in handles:
            h.remove()

        if args.method in ['entropy', 'magent', 'esparse']:
            wrapped_layers['mlp.down_proj'].prepare_for_hist()
            handles = []
            handles.append(sub_layers['mlp.down_proj'].register_forward_hook(hook_io('mlp.down_proj')))
            with torch.no_grad():
                for s in tqdm(range(len(inputs))):
                    _ = layer(**inputs[s])[0]
            for h in handles:
                h.remove()

            entropy = wrapped_layers['mlp.down_proj'].calculate_entropy()

        imp = 0
        if args.structure_prune:
            for sub_layer in sub_layers:
                weights =  torch.abs(sub_layers[sub_layer].weight.data)  # (out, in)
                x_norm_l2 = torch.sqrt(wrapped_layers[sub_layer].x_norm_l2.reshape((1,-1))) #(1, in)

                if args.method == 'entropy':
                    imp = get_importance(args.method, sub_layer, weights, x_norm_l2, entropy)
                    continue

                if args.method in ['weight', 'wanda', 'esparse', 'magent', 'group_wanda']:
                    imp += get_importance(args.method, sub_layer, weights, x_norm_l2, entropy, args.alpha/10) # (1, in)                 

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
                    inputs[j]['hidden_states'] = layer(**inputs[j])[0]
        else:
            pass
    
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
def apply_channel_prune(model, idx, model_type='vlm'):
    model = copy.deepcopy(model)  
    all_down_linears = []
    all_up_linears = []
    all_gate_linears = []
    llm_layers = get_llm_layers(model, model_type)
    for i, decoder_layer in enumerate(llm_layers):
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
            llm_layers[i_linear].mlp.down_proj = new_down_linear
            llm_layers[i_linear].mlp.gate_proj = new_gate_linear
            llm_layers[i_linear].mlp.up_proj = new_up_linear

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
    parser.add_argument('--wanda_mode', type=str, default='local', choices=['local', 'global'])
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
    args.model_type = 'vlm'  # main函数默认使用vlm模式

    keep_indices, prune_indices = prune(args, model, tokenizer, generation_config, img_path, prompt)
    prune_model = apply_channel_prune(model, keep_indices, 'vlm')

    prune_model_size = get_model_size(prune_model, count_nonzero_only=True)
    print(f"Prune model has size={prune_model_size/MiB:.2f} MiB")
    prune_model_params = get_num_parameters(prune_model, count_nonzero_only=True)
    print(f"Prune model has {prune_model_params/1e9:.2f}B parameters")

    copy_all_files('/data/base_model/base2B', dst_dir=args.save_path)
    model.save_pretrained(args.save_path)
    print(f"Pruned model saved to {args.save_path}")

def debug():
    prompt_type = 'base'
    model, tokenizer = load_model_tokenizer(f'/data/zige.wang/deploy/Tagbar_2B_ver_20250619FVOB')
    prompt = load_prompt('/data/template.yaml', prompt_type)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    img_path = '/data/Dataset/filtered/tagbar'

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.wanda_mode = 'local'
    args.hook_type = 'generate'
    args.prune_type = 'sequential'
    args.method = 'magent'
    args.structure_prune = True
    args.prune_ratio = 2
    args.nsamples = 20
    args.alpha = 9
    args.plot_wanda_ent = False
    args.model_type = 'vlm'
    args.data_path = '/data/zige.wang/data/alpaca_data_cleaned.json'
    assert args.method in ['weight', 'wanda', 'entropy', 'esparse', 'magent', 'group_wanda']

    print(f'>>> Running: prune_ratio={args.prune_ratio}0%, method={args.method}, nsamples={args.nsamples}, entropy_ratio={args.alpha}0%')

    save_path = f'/data/zige.wang/0723_test_sample/{args.method}_{args.hook_type}_{args.prune_type}_r{args.prune_ratio}_a{args.alpha}_{args.wanda_mode}_n{args.nsamples}'
    
    # 根据model_type选择正确的数据路径
    if args.model_type == 'llm':
        data_path = args.data_path
    else:
        data_path = img_path
    
    keep_indices, prune_indices = prune(args, model, tokenizer, generation_config, data_path, prompt)
    # torch.save(torch.stack(prune_indices), f'./pt/{args.method}_{args.nsamples}_{args.prune_ratio}.pt')

    prune_model = apply_channel_prune(model, keep_indices, args.model_type)

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
    