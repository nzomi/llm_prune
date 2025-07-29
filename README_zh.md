# LLM 剪枝框架

一个统一的大语言模型(LLM)剪枝框架，支持多种模型架构和剪枝方法。

[English Documentation](README.md)

## 特性

- **多模型支持**: 兼容 InternVL、InternVL-9B 和 Qwen 模型
- **多种剪枝方法**: 支持基于权重、WANDA、组WANDA、熵等剪枝技术
- **灵活架构**: 模块化设计，包含模型特定适配器
- **简易配置**: 基于YAML的配置管理
- **评估工具**: 内置困惑度评估功能

## 项目结构

```
llm_prune/
├── src/
│   ├── core/                 # 核心模块
│   │   ├── __init__.py
│   │   ├── loader.py         # 统一数据加载
│   │   ├── importance.py     # 重要性计算方法
│   │   ├── wrapper.py        # 层包装工具
│   │   └── pruner.py         # 主要剪枝逻辑
│   ├── models/               # 模型特定适配器
│   │   ├── __init__.py
│   │   ├── internvl.py       # InternVL 适配器
│   │   ├── internvl_9b.py    # InternVL-9B 适配器
│   │   └── qwen.py           # Qwen 适配器
│   ├── lib/                  # 附加工具
│   │   ├── ablate.py
│   │   ├── data.py
│   │   ├── eval.py
│   │   └── ...
│   └── utils.py              # 通用工具
├── configs/
│   └── default.yaml          # 默认配置
├── main.py                   # 主入口点
├── requirements.txt          # 依赖项
└── README.md                 # 英文文档
```

## 安装

1. 克隆仓库:
```bash
git clone <repository-url>
cd llm_prune
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```bash
python main.py \
    --model_type internvl \
    --model_path /path/to/model \
    --save_path /path/to/output \
    --img_path /path/to/images \
    --sparsity_ratio 0.5 \
    --method group_wanda
```

### 使用配置文件

1. 编辑 `configs/default.yaml` 设置参数
2. 使用配置运行:

```bash
python main.py --config configs/default.yaml
```

## 支持的模型

- **InternVL 2B**: `--model_type internvl --model_size 2b` (默认)
- **InternVL 9B**: `--model_type internvl --model_size 9b`
- **Qwen**: `--model_type qwen`

## 剪枝方法

- `weight`: 基于权重幅度的剪枝
- `wanda`: WANDA 剪枝方法
- `group_wanda`: 组级 WANDA 剪枝
- `entropy`: 基于熵的剪枝
- `esparse`: E-sparse 剪枝
- `magent`: 基于幅度的剪枝

## 命令行参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_type` | str | 必需 | 模型类型 (internvl/9b/qwen) |
| `--model_path` | str | 必需 | 模型路径 |
| `--save_path` | str | 必需 | 剪枝模型保存路径 |
| `--img_path` | str | 可选 | 图像或数据路径 |
| `--prompt_path` | str | 可选 | 提示词YAML文件路径 |
| `--nsamples` | int | 40 | 样本数量 |
| `--sparsity_ratio` | float | 0.5 | 稀疏比率 |
| `--method` | str | group_wanda | 剪枝方法 |
| `--prune_type` | str | parrallel | 剪枝类型 |
| `--eval_ppl` | flag | False | 评估困惑度 |

## 示例

### 剪枝 InternVL 2B 模型

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

### 剪枝 InternVL 9B 模型

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

### 剪枝 Qwen 模型

```bash
python main.py \
    --model_type qwen \
    --model_path ./models/qwen \
    --save_path ./output/qwen_pruned \
    --sparsity_ratio 0.5 \
    --method entropy \
    --hook_type prefill
```

## 高级用法

### 自定义模型适配器

要添加新模型支持，在 `src/models/` 中创建新适配器:

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

### 自定义剪枝方法

在 `src/core/importance.py` 中添加新的剪枝方法:

```python
def custom_importance(layer, sub_layer, weight, model_type='internvl'):
    # 你的自定义重要性计算
    return importance_scores
```

## 评估

框架包含内置评估工具:

```bash
# 剪枝后评估困惑度
python main.py --model_type internvl --model_path ./pruned_model --eval_ppl
```

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 如适用，添加测试
5. 提交拉取请求

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 引用

如果您在研究中使用此框架，请引用:

```bibtex
@misc{llm_pruning_framework,
    title={LLM Pruning Framework: A Unified Approach for Large Language Model Compression},
    author={Your Name},
    year={2024},
    url={https://github.com/your-username/llm_prune}
}
```

## 致谢

- 感谢 WANDA、InternVL 和 Qwen 模型的原作者
- 基于 Hugging Face 的 transformers 库构建
- 受文献中各种剪枝技术启发