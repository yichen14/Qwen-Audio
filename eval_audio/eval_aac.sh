#!/bin/sh
source /ocean/projects/cis210027p/ylu9/miniconda3/etc/profile.d/conda.sh
conda activate qwenaudio
cd /ocean/projects/cis210027p/ylu9/Qwen-Audio/eval_audio

ds="clotho"
python -m torch.distributed.launch --use-env \
    --nproc_per_node 1 --nnodes 1 \
    evaluate_caption.py \
    --checkpoint Qwen/Qwen-Audio \
    --dataset $ds \
    --batch-size 2 \
    --num-workers 2 \
