#!/bin/bash

cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device


save_path_root='/data/prune/1'
hook_type='prefill'
prune_type='sequential'
method='wanda'
prune_ratios=(1 2 3)
nsamples=(10 30 50 70 100 150 200 300)

run_python_command () {
    local prune_ratio=$1
    local nsamples=$2
    local save_path="${save_path_root}/${hook_type}_${prune_type}_r${prune_ratio}_n${nsamples}"

    echo ">>> Running: prune_ratio=${prune_ratio}0%, method=${method}, nsamples=${nsamples}"

    cmd="python prune.py \
        --prune_ratio $prune_ratio \
        --hook_type $hook_type \
        --prune_type $prune_type \
        --method $method \
        --nsamples $nsamples \
        --save_path $save_path"

    eval $cmd
}
for nsample in "${nsamples[@]}"; do
    for prune_ratio in "${prune_ratios[@]}"; do
        run_python_command "$prune_ratio" "$nsample"
    done
done