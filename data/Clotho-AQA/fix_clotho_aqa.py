import json
import os
import re

# many audio files are not align with its actual name 


def normalize_text(text):
    # Remove all characters except alphabets and numbers
    normalized_text = re.sub(r'[^a-zA-Z0-9]', '', text)
    # Convert the string to lowercase
    normalized_text = normalized_text.lower()
    return normalized_text

data_file = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/Clotho-AQA/clothoaqa_eval.jsonl"
cleaned_data_file = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/Clotho-AQA/clothoaqa_eval_cleaned.jsonl"

with open(data_file) as f:
    data = f.readlines()
    json_data = [json.loads(d) for d in data]

actual_audio_names = {}

for d in os.listdir('/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/Clotho-AQA/audio_files'):
    actual_audio_names[normalize_text(d.split('.wav')[0])] = d

path_set = set()
for d in json_data:
    audio_path = d['audio']
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
    
        audio_name = audio_path.split('audio_files/')[-1].split('.wav')[0]
        normalized_audio_name = normalize_text(audio_name)
        if normalized_audio_name in actual_audio_names.keys():
            d['audio'] = f"/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/Clotho-AQA/audio_files/{actual_audio_names[normalized_audio_name]}"
    with open(cleaned_data_file, 'a') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

# check all exists
with open(cleaned_data_file) as f:
    data = f.readlines()
    json_data = [json.loads(d) for d in data]

flag = False
for d in json_data:
    audio_path = d['audio']
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        flag = True

if not flag:
    print("All files found!")


    
    
