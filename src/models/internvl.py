from ..core.loader import load_model_tokenizer, load_prompt
from ..core.pruner import prune

class InternVLAdapter:
    def __init__(self, model_path, model_size='2b'):
        self.model_path = model_path
        self.model_size = model_size
        # 根据模型尺寸确定模型类型
        if model_size == '9b':
            self.model_type = '9b'
        else:
            self.model_type = 'internvl'  # 默认为2b模型
        
    def load_model(self):
        return load_model_tokenizer(self.model_path, self.model_type)
    
    def load_prompt(self, yaml_path, prompt_type='json'):
        return load_prompt(yaml_path, prompt_type, self.model_type)
    
    def prune_model(self, args, model, tokenizer, generation_config, img_path, prompt):
        return prune(args, model, tokenizer, generation_config, img_path, prompt, self.model_type)