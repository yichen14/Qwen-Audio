#!/bin/sh

# srun --pty -p GPU-small -N 1 --gpus=v100-32:1 /bin/bash -l
# sbatch -t 4:00:00 -p GPU-small -N 1 --gpus=v100-32:1 eval_asr.sh

module load anaconda3/2022.10
conda activate fastsp
# cd /ocean/projects/cis210027p/ylu9/Qwen-Audio/eval_audio

for ds in "librispeech" 
# for ds in "test" 
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
        --method kv_evict \
        --schedule none \
        --dump-feats false \
        --dump-task asr \
        --dump-feat-layer 31 
done