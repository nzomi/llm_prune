import os
import argparse
import torch
from src.models.internvl import InternVLAdapter
from src.models.qwen import QwenAdapter
from src.lib.eval import eval_ppl
from src.core.pruner import apply_channel_prune, copy_all_files

def get_model_adapter(model_type, model_path, model_size='2b'):
    adapters = {
        'internvl': InternVLAdapter,
        'qwen': QwenAdapter
    }
    
    if model_type not in adapters:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if model_type == 'internvl':
        return adapters[model_type](model_path, model_size)
    else:
        return adapters[model_type](model_path)

def main():
    parser = argparse.ArgumentParser(description='LLM Pruning Framework')
    parser.add_argument('--model_type', type=str, choices=['internvl', 'qwen'], 
                       required=True, help='Model type to prune')
    parser.add_argument('--model_size', type=str, default='2b', choices=['2b', '9b'],
                       help='Model size for InternVL (2b or 9b), ignored for other models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save pruned model')
    parser.add_argument('--img_path', type=str, help='Path to images or data')
    parser.add_argument('--prompt_path', type=str, help='Path to prompt YAML file')
    parser.add_argument('--nsamples', type=int, default=40, help='Number of samples')
    parser.add_argument('--kde_nsamples', type=int, default=30, help='KDE samples')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity ratio')
    parser.add_argument('--method', type=str, default='group_wanda', 
                       choices=['weight', 'wanda', 'group_wanda', 'entropy', 'esparse', 'magent'],
                       help='Pruning method')
    parser.add_argument('--prune_type', type=str, default='parrallel', 
                       choices=['sequential', 'parrallel'], help='Pruning type')
    parser.add_argument('--structure_prune', action='store_true', help='Enable structure pruning')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter')
    parser.add_argument('--hook_type', type=str, default='prefill', help='Hook type for Qwen')
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate perplexity')
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device')
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # Get model adapter
    adapter = get_model_adapter(args.model_type, args.model_path, args.model_size)
    
    # Load model and tokenizer
    print(f"Loading {args.model_type} model from {args.model_path}")
    model, tokenizer = adapter.load_model()
    
    # Load prompt if provided
    prompt = None
    if args.prompt_path:
        prompt = adapter.load_prompt(args.prompt_path)
    
    # Set generation config
    generation_config = {
        'max_new_tokens': 512,
        'do_sample': False
    }
    
    # Perform pruning
    if args.img_path:
        print(f"Starting pruning with method: {args.method}")
        keep_indices, prune_indices = adapter.prune_model(
            args, model, tokenizer, generation_config, args.img_path, prompt
        )
        
        # Apply pruning
        if keep_indices:
            print("Applying channel pruning...")
            apply_channel_prune(model, keep_indices, args.model_type)
            
            # Save pruned model
            print(f"Saving pruned model to {args.save_path}")
            copy_all_files(args, args.model_path, args.save_path)
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
    
    # Evaluate perplexity if requested
    if args.eval_ppl:
        print("Evaluating perplexity...")
        ppl = eval_ppl(args, model, tokenizer)
        print(f"Perplexity: {ppl}")
    
    print("Done!")

if __name__ == "__main__":
    main()