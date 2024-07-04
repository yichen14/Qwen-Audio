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

tasks = ["aac_2"]

for task in tasks:
    print(f"Task: {task}")

    task_feat_path = f"/ocean/projects/cis210027p/ylu9/Qwen-Audio/layer_selection/features/{task}/layer_31_merge_31_output.pkl"
    with open(task_feat_path, 'rb') as f:
        task_feat = pickle.load(f)
    original_entropy = calculate_entropy(task_feat)
    print(f"Original entropy: {original_entropy}")

    transfer_entropy_list = []
    for layer in range(32):
        masked_task_feat_path = f"/ocean/projects/cis210027p/ylu9/Qwen-Audio/layer_selection/features/{task}/layer_31_merge_{layer}_output.pkl"

        with open(masked_task_feat_path, 'rb') as f:
            masked_task_feat = pickle.load(f)
        
        layer_masked_entropy = calculate_entropy(masked_task_feat)
        print(f"Layer {layer} masked entropy: {layer_masked_entropy}")
        print(f"Entropy difference: {original_entropy - layer_masked_entropy}")
        transfer_entropy_list.append(original_entropy - layer_masked_entropy)
    
    # visualize the transfer entropy
    # wer_list = [0.5142, 0.1139, 0.1194, 0.0967, 0.0929, 0.0819, 0.0632, 0.0628, 0.0429, 0.0535, 0.2694, 0.1413, 0.1345, 0.1627, 0.1220, 0.1033, 0.0774, 0.0530, 0.0202, 0.0216, 0.0199, 0.0191, 0.0197, 0.0195, 0.0193, 0.0199, 0.0189, 0.0187, 0.0193, 0.0197, 0.0199, 0.0195]
    # wer_list = NormalizeData(wer_list)
    transfer_entropy_list = abs(np.array(transfer_entropy_list))
    transfer_entropy_list = NormalizeData(transfer_entropy_list)

    plt.plot(transfer_entropy_list, label="Transfer Entropy")
    # plt.plot(wer_list, label="WER")
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Transfer Entropy")
    plt.title(f"Transfer Entropy of {task}")
    plt.grid()
    plt.savefig(f"./transfer_entropy_{task}.png")

        