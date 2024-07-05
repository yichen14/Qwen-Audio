#!/bin/sh
source /ocean/projects/cis210027p/ylu9/miniconda3/etc/profile.d/conda.sh
conda activate qwenaudio
cd /ocean/projects/cis210027p/ylu9/Qwen-Audio/eval_audio

for ds in "librispeech" 
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env \
        --nproc_per_node 1 --nnodes 1 \
        evaluate_asr.py \
        --checkpoint Qwen/Qwen-Audio \
        --dataset $ds \
        --batch-size 1 \
        --num-workers 1 \
        --mem-reduce-rate 0.5 \
        --perform-layer 9 \
        --method merge_kv_evict \
        --schedule none \
        --dump-feats false \
        --dump-task asr \
        --dump-feat-layer 31 
done