import os
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torchvision.transforms as T
import yaml
import math

from lib.eval import eval_ppl
import argparse


def load_model_tokenizer(path):
    # device_map = split_model(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        # use_flash_attn=False,
        trust_remote_code=True,
        device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model.seqlen = model.config.max_position_embeddings 
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='prefill')
    args = parser.parse_args()

    # path = '/data/prune/Qwen3-1.7B/group_wanda_generate_parrallel_r3_n1_a1_k60'
    model, tokenizer = load_model_tokenizer(args.path)
    device = torch.device("cuda:0")
    model.seqlen = model.config.max_position_embeddings 
    ppl_test = eval_ppl(args, model, tokenizer, device=model.device)
    print(f"wikitext perplexity {ppl_test}")

    prune_method = args.path.split("/")[-3].split("_")[0]
    prune_ratio = int(args.path.split("_r")[-1].split("_")[0])
    nsamples = int(args.path.split("_n")[-1].split("_")[0])

    save_filepath = "log_ppl_8B.txt"
    write_header = not os.path.exists(save_filepath)
    with open(save_filepath, "a") as f:
        if write_header:
            print(f"{'method':<10}{'actual_sparsity':<18}{'nsamples':<10}{'ppl_test':<10}", file=f)
        print(f"{prune_method:<10}{prune_ratio:<18}{nsamples:<10}{ppl_test:<10.4f}", file=f)

def debug():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    path = '/data/prune/Qwen3-1.7B/group_wanda_generate_parrallel_r3_n1_a1_k60'
    model, tokenizer = load_model_tokenizer(path)
    device = torch.device("cuda:0")
    model.seqlen = model.config.max_position_embeddings 
    ppl_test = eval_ppl(args, model, tokenizer, device=model.device)
    print(f"wikitext perplexity {ppl_test}")

    prune_method = path.split("/")[-3]
    prune_ratio = int(path.split("_r")[-1].split("_")[0])
    nsamples = int(path.split("_n")[-1].split("_")[0])

    save_filepath = "log_ppl.txt"
    write_header = not os.path.exists(save_filepath)
    with open(save_filepath, "a") as f:
        if write_header:
            print(f"{'method':<10}{'actual_sparsity':<18}{'nsamples':<10}{'ppl_test':<10}", file=f)
        print(f"{prune_method:<10}{prune_ratio:<18}{nsamples:<10}{ppl_test:<10.4f}", file=f)

if __name__ == "__main__":
    main()
    # debug()