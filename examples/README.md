# 实验脚本使用说明

本目录包含了适配新统一框架的实验脚本，用于替代原有的分散脚本文件。

## 脚本说明

### 剪枝脚本

#### 1. InternVL 2B 模型剪枝
**文件**: `run_internvl_2b.sh`

```bash
# 使用前请修改脚本中的路径配置
# model_path: InternVL 2B模型路径
# img_path: 图像数据路径
# save_path_root: 剪枝结果保存根目录

bash run_internvl_2b.sh
```

**对应原脚本**: `run.sh`

#### 2. InternVL 9B 模型剪枝
**文件**: `run_internvl_9b.sh`

```bash
# 使用前请修改脚本中的路径配置
# model_path: InternVL 9B模型路径
# img_path: 图像数据路径
# save_path_root: 剪枝结果保存根目录

bash run_internvl_9b.sh
```

**对应原脚本**: `run9B.sh`

#### 3. Qwen 模型剪枝
**文件**: `run_qwen.sh`

```bash
# 使用前请修改脚本中的路径配置
# model_path: Qwen模型路径
# img_path: 数据路径
# save_path_root: 剪枝结果保存根目录

bash run_qwen.sh
```

**对应原脚本**: `run_qwen3_8.sh`, `run_qwen3_1_7.sh`

### 评估脚本

#### 困惑度评估
**文件**: `eval_ppl.sh`

```bash
# 使用前请修改脚本中的配置
# model_type: 模型类型 ('internvl' 或 'qwen')
# model_size: 模型尺寸 ('2b' 或 '9b'，仅对internvl有效)
# base_path: 剪枝模型的基础路径

bash eval_ppl.sh
```

**对应原脚本**: `ppl.sh`, `ppl8B.sh`

## 主要改进

### 1. 统一入口
- 所有脚本都调用 `main.py` 而不是分散的 `prune.py`, `prune9B.py`, `pruneQwen.py`
- 通过 `--model_type` 和 `--model_size` 参数区分不同模型

### 2. 参数映射
原有参数到新框架的映射关系：

| 原参数 | 新参数 | 说明 |
|--------|--------|------|
| `--prune_ratio` | `--sparsity_ratio` | 稀疏比率，需要转换为小数形式 |
| 脚本名称区分模型 | `--model_type` + `--model_size` | 统一通过参数指定 |
| `--kde_nsamples` | `--kde_nsamples` | 保持不变 |
| `--nsamples` | `--nsamples` | 保持不变 |

### 3. 路径配置
每个脚本开头都有清晰的配置区域，需要根据实际环境修改：

```bash
# Model configuration
model_type='internvl'
model_size='2b'
model_path='/path/to/model'  # 请修改为实际模型路径
img_path='/path/to/images'   # 请修改为实际图像路径
```

## 使用步骤

1. **修改配置**: 编辑对应脚本，设置正确的模型路径和数据路径
2. **设置权限**: `chmod +x *.sh`
3. **运行剪枝**: 执行对应的剪枝脚本
4. **评估结果**: 使用 `eval_ppl.sh` 评估剪枝后模型的困惑度

## 注意事项

1. **路径设置**: 所有脚本中的路径都需要根据实际环境进行修改
2. **CUDA设备**: 根据可用GPU修改 `cuda_device` 变量
3. **稀疏比率**: 新框架使用0-1之间的小数表示稀疏比率，而不是0-10的整数
4. **模型兼容性**: 确保模型路径指向正确的模型文件

## 故障排除

如果遇到问题，请检查：
1. 模型路径是否正确
2. 数据路径是否存在
3. CUDA设备是否可用
4. 依赖包是否已安装（参考根目录的 `requirements.txt`）