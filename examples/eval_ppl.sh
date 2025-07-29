#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 配置参数
model_type='qwen'  # 或者 'internvl'
model_size='2b'    # 仅对internvl有效，可选 '2b' 或 '9b'
base_path='/data/prune'  # 剪枝模型的基础路径

# 根据模型类型设置路径
if [ "$model_type" = "qwen" ]; then
    search_path="$base_path/Qwen3-1.7B/magent/a"
elif [ "$model_type" = "internvl" ] && [ "$model_size" = "2b" ]; then
    search_path="$base_path/internvl2B"
elif [ "$model_type" = "internvl" ] && [ "$model_size" = "9b" ]; then
    search_path="$base_path/internvl9B"
else
    echo "Error: Unsupported model configuration"
    exit 1
fi

echo "Evaluating models in: $search_path"
echo "Model type: $model_type, Model size: $model_size"
echo "==========================================="

# 遍历所有剪枝后的模型进行评估
for filepath in $(find "$search_path" -type d -name "*" | sort -V); do
    # 检查是否是有效的模型目录（包含config.json或pytorch_model.bin等文件）
    if [ -f "$filepath/config.json" ] || [ -f "$filepath/pytorch_model.bin" ] || [ -f "$filepath/model.safetensors" ]; then
        echo "Processing: $filepath"
        
        # 使用新的统一框架进行评估
        if [ "$model_type" = "internvl" ]; then
            python main.py \
                --model_type "$model_type" \
                --model_size "$model_size" \
                --model_path "$filepath" \
                --eval_ppl \
                --save_path "/tmp/dummy_save_path"  # eval模式不需要保存
        else
            python main.py \
                --model_type "$model_type" \
                --model_path "$filepath" \
                --eval_ppl \
                --save_path "/tmp/dummy_save_path"  # eval模式不需要保存
        fi
        
        echo "Completed: $filepath"
        echo "------------------------------------------"
    fi
done

echo "All evaluations completed!"