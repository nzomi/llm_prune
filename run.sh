#!/bin/bash

cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device


save_path_root='./prune'
hook_type='prefill'
prune_type='sequential'
method='wanda'
prune_ratios=(1 2 3 4 5)
nsamples=(10 30 50 100 200)

run_python_command () {
    local prune_ratio=$1
    local nsamples=$2
    local save_path="${save_path_root}/${hook_type}_${prune_type}_r${prune_ratio}"

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