#!/bin/sh

# . ./exp_asr.sh --method atome --perform_layer 0 --schedule constant
source /ocean/projects/cis210027p/ylu9/miniconda3/etc/profile.d/conda.sh
conda activate qwenaudio
cd /ocean/projects/cis210027p/ylu9/Qwen-Audio/eval_audio


ds="clotho"
perform_layer=0
method="merge_kv_evict"
schedule="none"
mem_reduce_rate=0.0

. ./parse_options.sh

echo "Running inference with mem_reduce_rate=$mem_reduce_rate on AAC task"
echo "perform_layer=$perform_layer"
echo "method=$method"
echo "schedule=$schedule"
python evaluate_caption.py \
    --checkpoint Qwen/Qwen-Audio \
    --dataset $ds \
    --batch-size 1 \
    --num-workers 1 \
    --mem-reduce-rate $mem_reduce_rate \
    --perform-layer $perform_layer \
    --method $method \
    --schedule $schedule \
    --dump-feats false \
    --dump-task aac \
    --dump-feat-layer 31 
