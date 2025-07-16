#!/bin/bash

cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device


save_path_root='/data/prune/1'
hook_type='generate'
prune_type='sequential'
# prune_type='generate'
methods1=('wanda' 'group_wanda' 'entropy')
methods2=('magent' 'esparse')
prune_ratios=(1 2 3)
nsamples=(30)
alphas1=(0)
alphas2=(1 2 3 4 5 6 7 8 9)

run_python_command () {
    local prune_ratio=$1
    local method=$2
    local nsamples=$3
    local alpha=$4
    local save_path="${save_path_root}/${method}_${hook_type}_${prune_type}_r${prune_ratio}_n${nsample}_a${alpha}"

    echo ">>> Running: prune_ratio=${prune_ratio}0%, method=${method}, nsamples=${nsamples}, alpha=${alpha}"

    cmd="python prune.py \
        --prune_ratio $prune_ratio \
        --hook_type $hook_type \
        --prune_type $prune_type \
        --method $method \
        --nsamples $nsamples \
        --save_path $save_path"

    eval $cmd
}

for method in "${methods1[@]}"; do
    for prune_ratio in "${prune_ratios[@]}"; do
        for alpha in "${alphas1[@]}"; do
            run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
        done
    done
done

for method in "${methods2[@]}"; do
    for prune_ratio in "${prune_ratios[@]}"; do
        for alpha in "${alphas2[@]}"; do
            run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
        done
    done
done