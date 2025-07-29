from ..core.loader import load_model_tokenizer, load_prompt
from ..core.pruner import prune

class QwenAdapter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_type = 'qwen'
        
    def load_model(self):
        return load_model_tokenizer(self.model_path, self.model_type)
    
    def load_prompt(self, yaml_path, prompt_type='json'):
        return load_prompt(yaml_path, prompt_type, self.model_type)
    
    def prune_model(self, args, model, tokenizer, generation_config, img_path, prompt):
        return prune(args, model, tokenizer, generation_config, img_path, prompt, self.model_type)