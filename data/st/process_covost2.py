import os
import csv
import json
from tqdm import tqdm

test_data_tsv = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/st/covost_v2.en_zh-CN.tsv"
data_dir = "/ocean/projects/cis210027p/shared/corpora/commonvoice/cv-corpus-4/en/clips/"
output_file = "/ocean/projects/cis210027p/ylu9/Qwen-Audio/data/st/covost2_eval.jsonl"

with open(test_data_tsv, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in tqdm(reader):
        if row["split"] != "test":
            continue
        audio_path = os.path.join(data_dir, row["path"])
        if os.path.exists(audio_path) is False:
            print(f"File not found: {audio_path}")
            continue
        source = "covost_v2_en_zh_test"
        gt = row["translation"]
        json_data = {"audio": audio_path, "gt": gt, "source": source}
        with open(output_file, "a") as f:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

