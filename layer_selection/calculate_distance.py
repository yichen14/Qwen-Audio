# modify from:
# How to Estimate Model Transferability of Pre-Trained Speech Models? https://arxiv.org/pdf/2306.01015
# https://github.com/virginiakm1988/LogME-CTC/blob/main/calculate_tsne_distance.py
# https://github.com/virginiakm1988/LogME-CTC/blob/main/calculate_swd_distance.py

import numpy as np
from geomloss import SamplesLoss
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import sys
import io
import ot as ot
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import logging
import pwcca

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
np.random.seed(42)

def compute_tsne_distance(feat1, feat2):
    """
    Compute the sliced Wasserstein distance and t-SNE distance between the means
    of two data distributions.

    Args:
        feat1 (np.ndarray): numpy array of shape (sample_num, m, n)
        feat2 (np.ndarray): numpy array of shape (sample_num, n)

    Returns:
        swd_distance (float): the sliced Wasserstein distance between the means of feat1 and feat2
        tsne_distance (float): the t-SNE distance between the means of feat1 and feat2
    """
    
    n_components = 50 # number of dimensions for the reduced features
    perplexity = 30 if feat1.shape[0] > 30 else (feat1.shape[0] // 2)
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=2, random_state=42, learning_rate="auto", n_iter=5000, perplexity=perplexity)

    feat1_reduced = tsne.fit_transform(feat1)
    feat2_reduced = tsne.fit_transform(feat2)

    tsne_distance = np.linalg.norm(np.median(feat1_reduced, axis=0) - np.median(feat2_reduced, axis=0))

    # logging('t-SNE distance:', tsne_distance)

    return  feat1_reduced, feat2_reduced, tsne_distance

def compute_distance(feat1, feat2):
    """
    Compute the sliced Wasserstein distance and t-SNE distance between the means
    of two data distributions.

    Args:
        feat1 (np.ndarray): numpy array of shape (sample_num, m, n)
        feat2 (np.ndarray): numpy array of shape (sample_num, n)

    Returns:
        swd_distance (float): the sliced Wasserstein distance between the means of feat1 and feat2
        tsne_distance (float): the t-SNE distance between the means of feat1 and feat2
    """
    n = len(feat1)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    seed = 42
    n_projections = 50
    swd_distance = ot.sliced.sliced_wasserstein_distance(feat1, feat2, a, b, n_projections, seed=seed)

    return  swd_distance

def compute_pwcca(feat1, feat2, epsilon=1e-10):
    feat1 = feat1.T
    feat2 = feat2.T
    print(feat1.shape, feat2.shape)
    pwcca_mean, w, _ = pwcca.compute_pwcca(feat1, feat2, epsilon=epsilon)
    return pwcca_mean


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("layer_distance.log"), logging.StreamHandler()])
tasks = ["asr", "st", "sum", "sv"]#sys.argv[1]

for task in tasks:
    total_swd_distance_list = []
    total_tsne_distance_list = []
    for layer in range(32):
        task_feat_path = f"/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/features/{task}/layer_{layer}_time_0_output.pkl"
        with open(task_feat_path, 'rb') as f:
            task_feat = pickle.load(f) # batch_size *
        
        # load other tasks' features
        other_tasks = [t for t in tasks if t != task]
        other_task_feats = {}
        for t in other_tasks:
            other_task_feat_path = f"/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/features/{t}/layer_{layer}_time_0_output.pkl"
            with open(other_task_feat_path, 'rb') as f:
                other_task_feat = pickle.load(f)
            other_task_feats[t] = other_task_feat
        
        swd_distance_total = 0.0
        tsne_distance_total = 0.0
        logging.info("=====================================")
        logging.info(f"Target Task: {task}, Layer: {layer}")
        for other_task, other_task_feat in other_task_feats.items():
            swd_distance_list = []
            tsne_distance_list = []
            for task_feat_sample, other_task_feat_sample in zip(task_feat, other_task_feat):
                task_feat_sample = task_feat_sample.squeeze(0)
                other_task_feat_sample = other_task_feat_sample.squeeze(0)

                swd_distance = compute_distance(task_feat_sample, other_task_feat_sample)
                _, _, tsne_distance = compute_tsne_distance(task_feat_sample, other_task_feat_sample)
                # tsne_distance = 0.0 # skip t-SNE distance for now

                swd_distance_list.append(swd_distance)
                tsne_distance_list.append(tsne_distance)
            avg_swd_distance = sum(swd_distance_list) / len(swd_distance_list)
            avg_tsne_distance = sum(tsne_distance_list) / len(tsne_distance_list)
            logging.info(f"Other Task: {other_task}, SWD distance: {avg_swd_distance}, t-SNE distance: {avg_tsne_distance}")
            swd_distance_total += avg_swd_distance
            tsne_distance_total += avg_tsne_distance

        logging.info(f"SWD distance: {swd_distance_total}")
        logging.info(f"t-SNE distance: {tsne_distance_total}")
        total_swd_distance_list.append(swd_distance_total)
        total_tsne_distance_list.append(tsne_distance_total)
    logging.info(f"------------- {task} ---------------")
    logging.info(f"SWD distance list: {total_swd_distance_list}")
    logging.info(f"t-SNE distance list: {total_tsne_distance_list}")
    logging.info(f"SWD rank: {np.argsort(total_swd_distance_list)}")
    logging.info(f"t-SNE rank: {np.argsort(total_tsne_distance_list)}")

        

# for task in tasks:
#     tsne_dist_list = []
#     swd_dist_list = []
#     logging(f"Compute distance between test and train features on {task}")
#     for layer in range(32):
#         test_feat_path = f"/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/features/{task}_test/layer_{layer}_time_0_output.pkl"
#         train_feat_path = f"/ocean/projects/cis210027p/ylu9/s2st-proj/SpeechT5/WavLLM/fairseq/examples/wavllm/layer_selections/features/{task}_train/layer_{layer}_time_0_output.pkl"
        
#         with open(test_feat_path, 'rb') as f:
#             test_feat = pickle.load(f)
#         with open(train_feat_path, 'rb') as f:
#             train_feat = pickle.load(f)
        
#         test_feat_list = []
#         for idx, features in enumerate(test_feat[:400]):
#             if idx % 2 == 1: # ignore system prompt
#                 test_feat_list.append(np.median(features.squeeze(), axis=0))
        
#         test_feat = np.array(test_feat_list)

#         train_feat_list = []
#         for idx, features in enumerate(train_feat):
#             train_feat_list.append(np.median(features.squeeze(), axis=0))
        
#         train_feat = np.array(train_feat_list)

#         distance = compute_distance(test_feat, train_feat)
#         _, _, tsne_distance = compute_tsne_distance(test_feat, train_feat)

#         logging(f"layer {layer}")
#         logging(f"SWD: {distance}, t-SNE: {tsne_distance}")
#         tsne_dist_list.append(tsne_distance)
#         swd_dist_list.append(distance)
    
#     # compute rank based on SWD and t-SNE distances
#     swd_rank = np.argsort(swd_dist_list)
#     tsne_rank = np.argsort(tsne_dist_list)
#     logging(f"SWD rank: {swd_rank}")
#     logging(f"t-SNE rank: {tsne_rank}")