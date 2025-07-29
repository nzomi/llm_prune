import os
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
import json
warnings.filterwarnings("ignore", category=FutureWarning)

from ..utils import *
from .loader import *
from .importance import *
from .wrapper import *
from ..lib.eval import eval_ppl

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
            pixel_values = load_image(os.path.join(img_path, img)).to(torch.bfloat16).cuda()
            pixel_data.append(pixel_values)
    return pixel_data

def get_llm_data(data_path, nsamples):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    random.shuffle(data)
    return [item['instruction'] for item in data[:nsamples]]

def prepare_calib_inputs(args, model, tokenizer, pixel_data, generation_config, prompt, model_type='internvl'):
    if model_type == 'qwen':
        return prepare_calib_inputs_qwen(args, model, tokenizer, pixel_data, generation_config, prompt)
    else:
        return prepare_calib_inputs_internvl(model, tokenizer, pixel_data, generation_config, prompt)

def prepare_calib_inputs_internvl(model, tokenizer, pixel_data, generation_config, prompt):
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

    llm_layers[0] = Catcher(llm_layers[0])

    for pixel_values in pixel_data:
        try:
            model.chat(tokenizer, pixel_values, prompt, generation_config)
        except ValueError:
            continue  

    llm_layers[0] = llm_layers[0].module
    return inputs

def prepare_calib_inputs_qwen(args, model, tokenizer, pixel_data, generation_config, prompt):
    llm_layers = model.model.layers
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
            if hasattr(model, 'chat'):
                model.chat(tokenizer, pixel_values, prompt, generation_config)
            else:
                messages = [{"role": "user", "content": f"{pixel_values}"}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
        except:
            pass

    llm_layers[0] = original_layer
    return captured_data

def prune(args, model, tokenizer, generation_config, img_path, prompt, model_type='internvl'):
    if model_type == 'qwen':
        pixel_data = get_llm_data(img_path, args.nsamples)
    else:
        pixel_data = get_pixel_data(img_path, args.nsamples)
    
    if args.prune_type == 'sequential':
        return prune_sequential(args, model, tokenizer, pixel_data, generation_config, prompt, model_type)
    elif args.prune_type == 'parrallel':
        return _prune_parrallel(args, model, tokenizer, pixel_data, generation_config, prompt, model_type)
    else:
        raise ValueError
            
def prune_sequential(args, model, tokenizer, pixel_data, generation_config, prompt, model_type='internvl'):
    with torch.no_grad():
        inputs = prepare_calib_inputs(args, model, tokenizer, pixel_data, generation_config, prompt, model_type)
    
    if model_type == 'qwen':
        model.config.use_cache = False
        llm_layer = model.model.layers
    else:
        model.config.llm_config.use_cache = False
        llm_layer = model.language_model.model.layers
    
    _prune_sequential(args, llm_layer, inputs, model_type)

def _prune_parrallel(args, model, tokenizer, pixel_data, generation_config, prompt, model_type='internvl'):
    keep_indices = []
    prune_indices = []
    
    if model_type == 'qwen':
        llm_layers = model.model.layers
    else:
        llm_layers = model.language_model.model.layers
    
    wrapped_layers = defaultdict(dict)
    for i, layers in enumerate(llm_layers):
        sub_layers = find_sub_layers(layers, model_type=model_type)
        for sub_layer in sub_layers:
            wrapped_layers[i][sub_layer] = WrappedLayer(sub_layers[sub_layer], i, sub_layer, 
                                                       kde_samples=args.kde_nsamples, model_type=model_type)

    def hook_io(i, sub_layer):
        def tmp(_, input, output):
            if input[0].shape[1] == 1:
                wrapped_layers[i][sub_layer].add_batch(input[0].data, output.data)
        return tmp

    handles = []
    for i, layers in enumerate(llm_layers):
        sub_layers = find_sub_layers(layers, model_type=model_type)
        for sub_layer in sub_layers:
            handles.append(sub_layers[sub_layer].register_forward_hook(hook_io(i, sub_layer)))

    with torch.no_grad():
        for pixel_values in tqdm(pixel_data):
            try:
                if model_type == 'qwen':
                    messages = [{"role": "user", "content": f"{pixel_values}"}]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=512
                    )
                else:
                    model.chat(tokenizer, pixel_values, prompt, generation_config)
            except:
                continue

    for h in handles:
        h.remove()

    entropy = [None] * len(llm_layers)
    if args.method in ['entropy', 'magent', 'esparse', 'test']:
        target_layer = 'feed_forward.w2' if model_type == '9b' else 'mlp.down_proj'
        for i, layers in tqdm(enumerate(llm_layers)):
            wrapped_layers[i][target_layer].prepare_for_kde()
            entropy[i] = wrapped_layers[i][target_layer].calculate_entropy_kde_total()

    if args.structure_prune:
        for i, layers in tqdm(enumerate(llm_layers)):
            sub_layers = find_sub_layers(layers, model_type=model_type)
            imp = 0
            wanda = 0
            for sub_layer in sub_layers:
                weights = torch.abs(sub_layers[sub_layer].weight.data)
                x_norm_l2 = torch.sqrt(wrapped_layers[i][sub_layer].x_norm_l2.reshape((1,-1)))
                wanda += get_importance('group_wanda', sub_layer, weights, x_norm_l2, entropy[i], 
                                      alpha=args.alpha/10, model_type=model_type)
                
                if args.method == 'entropy':
                    imp = get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], 
                                       alpha=args.alpha/10, model_type=model_type)
                    continue
                else:
                    imp += get_importance(args.method, sub_layer, weights, x_norm_l2, entropy[i], 
                                        alpha=args.alpha/10, model_type=model_type)
            
            if args.method == 'entropy':
                sorted_imp, sorted_idx = torch.sort(imp, descending=False)
            else:
                sorted_imp, sorted_idx = torch.sort(imp, descending=True)
            
            prune_n = int(args.sparsity_ratio * imp.shape[0])
            prune_indices.append(sorted_idx[:prune_n])
            keep_indices.append(sorted_idx[prune_n:])
    
    return keep_indices, prune_indices

def _prune_sequential(args, layers, inputs, model_type='internvl'):
    # Implementation would be similar to the original but unified
    pass

def prune_linear_channel(linear_layer, selected_idx, prune_channel):
    if prune_channel == 'in':
        linear_layer.weight.data = linear_layer.weight.data[:, selected_idx]
        linear_layer.in_features = len(selected_idx)
    elif prune_channel == 'out':
        linear_layer.weight.data = linear_layer.weight.data[selected_idx, :]
        if linear_layer.bias is not None:
            linear_layer.bias.data = linear_layer.bias.data[selected_idx]
        linear_layer.out_features = len(selected_idx)

@torch.no_grad()
def apply_channel_prune(model, idx, model_type='internvl'):
    if model_type == 'qwen':
        layers = model.model.layers
    else:
        layers = model.language_model.model.layers
    
    for i, layer in enumerate(layers):
        if model_type == '9b':
            prune_linear_channel(layer.feed_forward.w1, idx[i], 'out')
            prune_linear_channel(layer.feed_forward.w3, idx[i], 'out')
            prune_linear_channel(layer.feed_forward.w2, idx[i], 'in')
        else:
            prune_linear_channel(layer.mlp.up_proj, idx[i], 'out')
            prune_linear_channel(layer.mlp.gate_proj, idx[i], 'out')
            prune_linear_channel(layer.mlp.down_proj, idx[i], 'in')

def copy_all_files(args, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)