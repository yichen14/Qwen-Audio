# ASR
layer=9
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method random_evict --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method evicit --perform_layer $layer --schedule none
# # sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method kv_evict --perform_layer 0 --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method atome --perform_layer 0 --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method random_merge --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_ave --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer $layer --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer $layer --schedule decay
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer 0 --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_weighted --perform_layer 0 --schedule decay
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_kv_evict --perform_layer $layer --schedule none

# AAC
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./run_aac.sh  # baseline (0.0 mem reduce)
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method random_evict --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method evicit --perform_layer $layer --schedule none
# # sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method kv_evict --perform_layer 0 --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method atome --perform_layer 0 --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method random_merge --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method merge_ave --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method merge_weighted --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method merge_weighted --perform_layer $layer --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method merge_weighted --perform_layer $layer --schedule decay
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method merge_weighted --perform_layer 0 --schedule constant
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aac.sh --method merge_weighted --perform_layer 0 --schedule decay
# # sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_kv_evict --perform_layer $layer --schedule none

# AQA
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./run_aqa.sh  # baseline (0.0 mem reduce)
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method random_evict --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method evicit --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method kv_evict --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method atome --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method random_merge --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method merge_ave --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method merge_weighted --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method merge_weighted --perform_layer $layer --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method merge_weighted --perform_layer $layer --schedule decay
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method merge_weighted --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_aqa.sh --method merge_weighted --perform_layer 0 --schedule decay
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_kv_evict --perform_layer $layer --schedule none

# ST
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./run_st.sh  # baseline (0.0 mem reduce)
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method random_evict --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method evicit --perform_layer $layer --schedule none
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method kv_evict --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method atome --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method random_merge --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method merge_ave --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method merge_weighted --perform_layer $layer --schedule none
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method merge_weighted --perform_layer $layer --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method merge_weighted --perform_layer $layer --schedule decay
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method merge_weighted --perform_layer 0 --schedule constant
sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_st.sh --method merge_weighted --perform_layer 0 --schedule decay
# sbatch -t 2-00:00:00 -p GPU-shared -N 1 --gpus=v100-32:1 ./exp_asr.sh --method merge_kv_evict --perform_layer $layer --schedule none