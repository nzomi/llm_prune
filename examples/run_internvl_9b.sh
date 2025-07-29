#!/bin/bash

cuda_device=6
export CUDA_VISIBLE_DEVICES=$cuda_device

# Model configuration
model_type='internvl'
model_size='9b'
model_path='/path/to/internvl9b/model'  # 请修改为实际模型路径
img_path='/path/to/images'  # 请修改为实际图像路径

# Experiment configuration
save_path_root='/data/prune/internvl9B'
hook_type='generate'
prune_type='parrallel'
methods1=('group_wanda' 'entropy' 'magent' 'esparse')
prune_ratios=(5)
nsamples=(30)
alphas1=(9)

run_python_command () {
    local prune_ratio=$1
    local method=$2
    local nsamples=30
    local alpha=$3
    local save_path="${save_path_root}/${method}_${hook_type}_${prune_type}_r${prune_ratio}_n${nsamples}_a${alpha}"

    echo ">>> Running: prune_ratio=${prune_ratio}0%, method=${method}, nsamples=${nsamples}, alpha=${alpha}"

    cmd="python main.py \
        --model_type $model_type \
        --model_size $model_size \
        --model_path $model_path \
        --save_path $save_path \
        --img_path $img_path \
        --sparsity_ratio 0.${prune_ratio} \
        --method $method \
        --nsamples $nsamples \
        --alpha $alpha \
        --prune_type $prune_type \
        --hook_type $hook_type \
        --cuda_device $cuda_device"

    eval $cmd
}

# 运行实验
for method in "${methods1[@]}"; do
    for prune_ratio in "${prune_ratios[@]}"; do
        for alpha in "${alphas1[@]}"; do
            run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
        done
    done
done

echo "All experiments completed!"