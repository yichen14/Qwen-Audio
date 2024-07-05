# ASR
layer=9
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method random_evict --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method evicit --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method kv_evict --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method atome --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method random_merge --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_ave --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer $layer --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer $layer --schedule decay
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer 0 --schedule decay
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_kv_evict --perform_layer $layer --schedule none