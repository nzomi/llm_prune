export CUDA_VISIBLE_DEVICES=0

for filepath in $(ls /data/prune/Qwen3-1.7B/magent/a | sort -V); do
    fullpath="/data/prune/Qwen3-1.7B/magent/a/$filepath"
    echo "Processing: $fullpath"
    python testppl.py --path "$fullpath"
done
