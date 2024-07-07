import json
import os

data_file = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/Clotho-AQA/clothoaqa_eval.jsonl"

with open(data_file) as f:
    data = f.readlines()
    json_data = [json.loads(d) for d in data]

path_set = []
for d in json_data:
    audio_path = d['audio']
    if not os.path.exists(audio_path):
        path_set.append(audio_path)
print(len(path_set))
