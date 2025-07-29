#!/bin/bash

cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device


save_path_root='/data/prune/internvl2B'
hook_type='generate'
# prune_type='sequential'
prune_type='parrallel'
methods1=('group_wanda' 'entropy' 'esparse' 'magent')
methods2=('magent')
methods3=('weight')
prune_ratios=(4 7)
# prune_ratios=(8)
nsamples=(30)
alphas1=(8)
alphas2=(9)

run_python_command () {
    local prune_ratio=$1
    local method=$2
    local nsamples=$3
    local alpha=$4
    local save_path="${save_path_root}/acc/${method}_${hook_type}_${prune_type}_r${prune_ratio}_n${nsample}_a${alpha}"

    echo ">>> Running: prune_ratio=${prune_ratio}0%, method=${method}, nsamples=${nsamples}, alpha=${alpha}"

    cmd="python prune.py \
        --prune_ratio $prune_ratio \
        --hook_type $hook_type \
        --prune_type $prune_type \
        --method $method \
        --nsamples $nsamples \
        --alpha $alpha \
        --save_path $save_path"

    eval $cmd
}

# for method in "${methods2[@]}"; do
#     for prune_ratio in "${prune_ratios[@]}"; do
#         for alpha in "${alphas2[@]}"; do
#             run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
#         done
#     done
# done

# for method in "${methods1[@]}"; do
#     for prune_ratio in "${prune_ratios[@]}"; do
#         for alpha in "${alphas1[@]}"; do
#             run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
#         done
#     done
# done

for method in "${methods2[@]}"; do
    for prune_ratio in "${prune_ratios[@]}"; do
        for alpha in "${alphas1[@]}"; do
            run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
        done
    done
done