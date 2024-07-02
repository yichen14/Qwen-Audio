#!/bin/sh

# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./dump_aac_feat.sh
source /ocean/projects/cis210027p/ylu9/miniconda3/etc/profile.d/conda.sh
conda activate qwenaudio
cd /ocean/projects/cis210027p/ylu9/Qwen-Audio/eval_audio

for merge_layer in {0..31};
do
    echo "Running inference with merge_layer=$merge_layer on AAC task"
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env \
        --nproc_per_node 1 --nnodes 1 \
        evaluate_caption.py \
        --checkpoint Qwen/Qwen-Audio \
        --dataset "layer_select_aac" \
        --batch-size 1 \
        --num-workers 1 \
        --mem-reduce-rate 0.0 \
        --merge-ratio 0.5 \
        --merge-layer $merge_layer \
        --dump-feats true \
        --dump-task aac \
        --dump-feat-layer 31 
done