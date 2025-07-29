#!/bin/bash

cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device

# Model configuration
model_type='qwen'
model_path='/path/to/qwen/model'  # 请修改为实际模型路径
img_path='/path/to/data'  # 请修改为实际数据路径

# Experiment configuration
save_path_root='/data/prune/Qwen3-8B'
hook_type='generate'
prune_type='parrallel'
methods=('group_wanda' 'magent' 'esparse' 'entropy')
prune_ratios=(5)
nsamples=(128)
alphas=(1)
ksamples=(64)

run_python_command () {
    local method=$1
    local nsample=$2
    local k=$3
    local prune_ratio=$4
    local alpha=1
    local save_path="${save_path_root}/${method}_${hook_type}_${prune_type}_r${prune_ratio}_n${nsample}_a${alpha}_k${k}"

    echo ">>> Running: prune_ratio=${prune_ratio}0%, method=${method}, nsamples=${nsample}, alpha=${alpha}, ksamples=${k}"

    cmd="python main.py \
        --model_type $model_type \
        --model_path $model_path \
        --save_path $save_path \
        --img_path $img_path \
        --sparsity_ratio 0.${prune_ratio} \
        --method $method \
        --nsamples $nsample \
        --kde_nsamples $k \
        --alpha $alpha \
        --prune_type $prune_type \
        --hook_type $hook_type \
        --cuda_device $cuda_device"

    eval $cmd
}

# 运行实验
for prune_ratio in "${prune_ratios[@]}"; do
    for method in "${methods[@]}"; do
        for nsample in "${nsamples[@]}"; do
            for k in "${ksamples[@]}"; do
                run_python_command "$method" "$nsample" "$k" "$prune_ratio"
            done
        done
    done
done

echo "All experiments completed!"