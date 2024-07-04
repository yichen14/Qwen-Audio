import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def calculate_entropy(features):
    """
    Calculate the entropy of the feature of a layer.

    Args:
        features (list of np.ndarray): layer features of shape (batch_size, num_tokens, hidden_dim)
    
    Returns:
        entropy (float): the entropy of the feature
    """

    return np.mean([np.sum([np.log(np.std(f_j)) for f_j in feature.squeeze(0)]) for feature in features])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

log_file = "/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/layer_distance.log"

# parse log file 
tasks = ["asr"]
full_data = {}
with open(log_file, 'r') as f:
    lines = f.readlines()
    n = len(lines)
    i = 0
    while i < n:
        if "------------" in lines[i]:
            i += 7
            continue
        if "============" in lines[i]:
            data = lines[i+1:i+7]
            task = data[0].split("Target Task: ")[1].split(",")[0]
            total_swd = float(data[4].split("SWD distance: ")[1])
            total_tsne = float(data[5].split("t-SNE distance: ")[1])
            for line in data[1:4]:
                split_line = line.split(", ")
                other_task = split_line[0].split("Other Task: ")[1]
                swd = float(split_line[1].split(": ")[1])
                tsne = float(split_line[2].split(": ")[1])
                if task not in full_data:
                    full_data[task] = {}
                if other_task not in full_data[task]:
                    full_data[task][other_task] = {}
                if "swd" not in full_data[task][other_task]:
                    full_data[task][other_task]["swd"] = []
                if "tsne" not in full_data[task][other_task]:
                    full_data[task][other_task]["tsne"] = []
                full_data[task][other_task]["swd"].append(swd)
                full_data[task][other_task]["tsne"].append(tsne)
            if "total_swd" not in full_data[task]:
                full_data[task]["total_swd"] = []
            if "total_tsne" not in full_data[task]:
                full_data[task]["total_tsne"] = []
            full_data[task]["total_swd"].append(total_swd)
            full_data[task]["total_tsne"].append(total_tsne)
        
        i += 7

for task in full_data.keys():
    swd_list_dict = {}
    swd_list_dict["total"] = full_data[task]["total_swd"]

    tsne_list_dict = {}
    tsne_list_dict["total"] = full_data[task]["total_tsne"]
    
    for other_task in full_data[task].keys():
        if other_task == "total_swd" or other_task == "total_tsne":
            continue
        swd_list_dict[other_task] = full_data[task][other_task]["swd"]
        tsne_list_dict[other_task] = full_data[task][other_task]["tsne"]
    
    # calculate transfer entropy
    task_feat_path = f"/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/features/{task}/layer_31_time_0_merge_31_output.pkl"
    with open(task_feat_path, 'rb') as f:
        task_feat = pickle.load(f)
    original_entropy = calculate_entropy(task_feat)
    print(f"Original entropy: {original_entropy}")

    transfer_entropy_list = []
    for layer in range(32):
        masked_task_feat_path = f"/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/features/{task}/layer_31_time_0_merge_{layer}_output.pkl"

        with open(masked_task_feat_path, 'rb') as f:
            masked_task_feat = pickle.load(f)
        
        layer_masked_entropy = calculate_entropy(masked_task_feat)
        print(f"Layer {layer} masked entropy: {layer_masked_entropy}")
        print(f"Entropy difference: {original_entropy - layer_masked_entropy}")
        transfer_entropy_list.append(original_entropy - layer_masked_entropy)    

    transfer_entropy_list = np.abs(transfer_entropy_list)

    # visualization
    # TE Graph
    fig, ax = plt.subplots()
    title = f'Transfer Entropy for {task}'
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Transfer Entropy')
    ax.plot(transfer_entropy_list)
    ax.grid()
    fig.savefig(f'{task}-transfer_entropy.png')

    # SWD vs TE Graph
    normalized_te = NormalizeData(transfer_entropy_list)
    # normalized_te[0], normalized_te[1] = 1.0, 1.0
    
    normalized_swd = NormalizeData(full_data[task]["total_swd"])
    fig, ax = plt.subplots()
    title = f'SWD vs Transfer Entropy for {task}'
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Value')
    ax.plot(normalized_te, label='Transfer Entropy')
    ax.plot(normalized_swd, label='SWD')
    ax.grid()
    ax.legend()
    fig.savefig(f'{task}-swd_vs_te.png')

    # SWD graph
    fig, ax = plt.subplots()
    title = f'SWD for {task}'
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('SWD')
    for other_task, swd_list in swd_list_dict.items():
        ax.plot(swd_list, label=f'SWD-{other_task}')
    ax.grid()
    ax.legend()
    # save figure
    fig.savefig(f'{task}-swd.png')

    # t-SNE graph
    fig, ax = plt.subplots()
    title = f't-SNE for {task}'
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('t-SNE')
    for other_task, tsne_list in tsne_list_dict.items():
        ax.plot(tsne_list, label=f't-SNE-{other_task}')
    ax.grid()
    ax.legend()
    # save figure
    fig.savefig(f'{task}-tsne.png')
