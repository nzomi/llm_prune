# LLM Pruning Framework

A unified framework for pruning Large Language Models (LLMs) with support for multiple model architectures and pruning methods.

[中文文档](README_zh.md)

## Features

- **Multi-Model Support**: Compatible with InternVL, InternVL-9B, and Qwen models
- **Multiple Pruning Methods**: Supports weight-based, WANDA, group WANDA, entropy-based, and other pruning techniques
- **Flexible Architecture**: Modular design with model-specific adapters
- **Easy Configuration**: YAML-based configuration management
- **Evaluation Tools**: Built-in perplexity evaluation capabilities

## Project Structure

```
llm_prune/
├── src/
│   ├── core/                 # Core modules
│   │   ├── __init__.py
│   │   ├── loader.py         # Unified data loading
│   │   ├── importance.py     # Importance calculation methods
│   │   ├── wrapper.py        # Layer wrapping utilities
│   │   └── pruner.py         # Main pruning logic
│   ├── models/               # Model-specific adapters
│   │   ├── __init__.py
│   │   ├── internvl.py       # InternVL adapter
│   │   ├── internvl_9b.py    # InternVL-9B adapter
│   │   └── qwen.py           # Qwen adapter
│   ├── lib/                  # Additional utilities
│   │   ├── ablate.py
│   │   ├── data.py
│   │   ├── eval.py
│   │   └── ...
│   └── utils.py              # General utilities
├── configs/
│   └── default.yaml          # Default configuration
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm_prune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
python main.py \
    --model_type internvl \
    --model_path /path/to/model \
    --save_path /path/to/output \
    --img_path /path/to/images \
    --sparsity_ratio 0.5 \
    --method group_wanda
```

### Using Configuration File

1. Edit `configs/default.yaml` with your settings
2. Run with configuration:

```bash
python main.py --config configs/default.yaml
```

## Supported Models

- **InternVL 2B**: `--model_type internvl --model_size 2b` (default)
- **InternVL 9B**: `--model_type internvl --model_size 9b`
- **Qwen**: `--model_type qwen`

## Pruning Methods

- `weight`: Weight magnitude-based pruning
- `wanda`: WANDA pruning method
- `group_wanda`: Group-wise WANDA pruning
- `entropy`: Entropy-based pruning
- `esparse`: E-sparse pruning
- `magent`: Magnitude-based pruning

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_type` | str | Required | Model type (internvl/9b/qwen) |
| `--model_path` | str | Required | Path to the model |
| `--save_path` | str | Required | Path to save pruned model |
| `--img_path` | str | Optional | Path to images or data |
| `--prompt_path` | str | Optional | Path to prompt YAML file |
| `--nsamples` | int | 40 | Number of samples |
| `--sparsity_ratio` | float | 0.5 | Sparsity ratio |
| `--method` | str | group_wanda | Pruning method |
| `--prune_type` | str | parrallel | Pruning type |
| `--eval_ppl` | flag | False | Evaluate perplexity |

## Examples

### Prune InternVL 2B Model

```bash
python main.py \
    --model_type internvl \
    --model_size 2b \
    --model_path ./models/internvl2b \
    --save_path ./output/internvl2b_pruned \
    --img_path ./data/images \
    --sparsity_ratio 0.3 \
    --method group_wanda \
    --eval_ppl
```

### Prune InternVL 9B Model

```bash
python main.py \
    --model_type internvl \
    --model_size 9b \
    --model_path ./models/internvl9b \
    --save_path ./output/internvl9b_pruned \
    --img_path ./data/images \
    --sparsity_ratio 0.5 \
    --method group_wanda \
    --eval_ppl
```

### Prune Qwen Model

```bash
python main.py \
    --model_type qwen \
    --model_path ./models/qwen \
    --save_path ./output/qwen_pruned \
    --sparsity_ratio 0.5 \
    --method entropy \
    --hook_type prefill
```

## Advanced Usage

### Custom Model Adapter

To add support for a new model, create a new adapter in `src/models/`:

```python
from ..core.loader import load_model_tokenizer, load_prompt
from ..core.pruner import prune

class CustomModelAdapter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_type = 'custom'
        
    def load_model(self):
        return load_model_tokenizer(self.model_path, self.model_type)
    
    def prune_model(self, args, model, tokenizer, generation_config, img_path, prompt):
        return prune(args, model, tokenizer, generation_config, img_path, prompt, self.model_type)
```

### Custom Pruning Method

Add new pruning methods in `src/core/importance.py`:

```python
def custom_importance(layer, sub_layer, weight, model_type='internvl'):
    # Your custom importance calculation
    return importance_scores
```

## Evaluation

The framework includes built-in evaluation tools:

```bash
# Evaluate perplexity after pruning
python main.py --model_type internvl --model_path ./pruned_model --eval_ppl
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{llm_pruning_framework,
    title={LLM Pruning Framework: A Unified Approach for Large Language Model Compression},
    author={Your Name},
    year={2024},
    url={https://github.com/your-username/llm_prune}
}
```

## Acknowledgments

- Thanks to the original authors of WANDA, InternVL, and Qwen models
- Built upon the transformers library by Hugging Face
- Inspired by various pruning techniques in the literature