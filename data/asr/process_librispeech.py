import os
import csv
from tqdm import tqdm
import json

FILE_LIST = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/asr/ls_test_clean_list.txt"
GT_LIST = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/asr/ls_test_clean_gt.txt"
output_json = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/asr/librispeech_test_eval.jsonl"


# Read ground truth first
ground_truth = {}
with open(GT_LIST, 'r') as f:
    for line in tqdm(f):
        line = line.strip()
        audio_id = line.split(' ')[0]
        gt = ' '.join(line.split(' ')[1:])
        ground_truth[audio_id] = gt

with open(FILE_LIST, 'r') as f:
    for line in tqdm(f):
        line = line.strip()
        audio_path = line
        audio_id = os.path.basename(audio_path).split('.')[0]
        json_data = {"audio": line, "gt": ground_truth[audio_id], "source": "librispeech_test_clean"}
        with open(output_json, 'a') as f:
            f.write(json.dumps(json_data) + '\n')

# with open(output_tsv, 'w') as f:
#     writer = csv.writer(f, delimiter='\t')
#     for row in data:
#         writer.writerow(row)