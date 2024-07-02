import csv
import json
import os

data_dir = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/clotho/evaluation"
gt_file = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/clotho/clotho_captions_evaluation.csv"
output_file = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/clotho/clotho_eval_.jsonl"

missing_count = 0
with open(gt_file, 'r') as f, open(output_file, 'a') as out_f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # file_name, caption_1, caption_2, caption_3, caption_4, caption_5
        file_name = row[0]
        captions = row[1:]
        if os.path.exists(f"{data_dir}/{file_name}"):
            json_data = {"audio": f"{data_dir}/{file_name}", "gt": captions, "source": "clotho_evaluation"}
            out_f.write(json.dumps(json_data) + '\n')
        else:
            missing_count += 1

            print(f"File {file_name} does not exist.")
print(f"Missing {missing_count} files.")
        