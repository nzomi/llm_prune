#!/bin/bash

cuda_device=4
export CUDA_VISIBLE_DEVICES=$cuda_device


save_path_root='/data/prune/Qwen3-1.7B'
hook_type='generate'
prune_type='parrallel'
# prune_type='generate'
methods=('magent')
# methods2=('magent' 'esparse')
# methods3=('weight')
prune_ratios=(5)
nsamples=(128)
alphas=(1 2 3 4 5 6 7 8 9)
ksamples=(64)

run_python_command () {
    local prune_ratio=$4
    local method=$1
    local nsample=$2
    local alpha=$3
    local k=64
    local save_path="${save_path_root}/${method}/a5/${hook_type}_${prune_type}_r${prune_ratio}_n${nsample}_a${alpha}_k${k}"

    echo ">>> Running: prune_ratio=${prune_ratio}0%, method=${method}, nsamples=${nsample}, alpha=${alpha}, ksamples=${k}"

    cmd="python pruneQwen.py \
        --prune_ratio $prune_ratio \
        --hook_type $hook_type \
        --prune_type $prune_type \
        --method $method \
        --nsamples $nsample \
        --alpha $alpha \
        --kde_nsamples $k \
        --save_path $save_path"

    eval $cmd
}
for prune_ratio in "${prune_ratios[@]}"; do
    for method in "${methods[@]}"; do
        for nsample in "${nsamples[@]}"; do
            for alpha in "${alphas[@]}"; do
                run_python_command "$method" "$nsample" "$alpha" "$prune_ratio"
            done
        done
    done
done

# for method in "${methods1[@]}"; do
#     for prune_ratio in "${prune_ratios[@]}"; do
#         for alpha in "${alphas1[@]}"; do
#             run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
#         done
#     done
# done

# for method in "${methods3[@]}"; do
#     for prune_ratio in "${prune_ratios[@]}"; do
#         for alpha in "${alphas1[@]}"; do
#             run_python_command "$prune_ratio" "$method" "$nsamples" "$alpha"
#         done
#     done
# done