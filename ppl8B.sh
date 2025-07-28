export CUDA_VISIBLE_DEVICES=0

for filepath in $(ls /data/prune/Qwen3-8B | sort -V); do
    fullpath="/data/prune/Qwen3-8B/$filepath"
    echo "Processing: $fullpath"
    python testppl8B.py --path "$fullpath"
done
