import os
import csv
import json
from tqdm import tqdm


FILE_LIST = "./st_dev_clean_list.txt"
GT_LIST = "./st_dev_clean_gt.txt"
output_file = "./mustc_dev_clean_eval.jsonl"

# Read ground truth first
ground_truth = {}
with open(GT_LIST, 'r') as f:
    idx = 0
    for line in tqdm(f):
        gt = line.strip()
        audio_id = idx
        ground_truth[str(audio_id)] = gt
        idx += 1

with open(FILE_LIST, 'r') as f:
    for line in tqdm(f):
        line = line.strip()
        line = line.strip()
        audio_path = line
        audio_id = os.path.basename(audio_path).split('.')[0]
        json_data = {"audio": line, "gt": ground_truth[audio_id], "source": "mustc_en_de_dev"}
        with open(output_file, 'a') as f:
            f.write(json.dumps(json_data) + '\n')
