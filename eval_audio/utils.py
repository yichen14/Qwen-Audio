import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mem-reduce-rate', type=float, default=0.0)
    parser.add_argument('--method',
                        type=str, 
                        default='merge_weighted', 
                        choices=["random_evict",    # baseline 1: random drop some tokens from from [selected layer]
                                "evicit",           # baseline 2: drop the tokens based on attention score(fastV approach) from [selected layer]
                                "kv_evict",         # baseline 3: drop the kv cache in attention based on attn score of each head(H2O) from layer 0
                                "atome",            # baseline 4: average merge the tokens based on key similarity two by two from layer 0 using [constant] schedule
                                "random_merge",     # baseline 5: random weighted merge the token cluster from [selected layer] based on only attention score
                                'merge_ave',        # our method 1: average merge the token cluster based on key similarity from [selected layer]
                                'merge_weighted',   # our method 2: weighted merge the token cluster based on key similarity and attention score 
                                                    #               from [selected layer] with different merge schedule: [none, constant, decay]
                                "merge_kv_evict"])  # our method 3: weighted merge + kv_evict the token cluster 
                                                    #               based on key similarity and attention score from [selected layer]
    parser.add_argument('--schedule', type=str, default='none', choices=['none', 'constant', 'decay'])
    parser.add_argument('--ratio', type=float, default=0.0)
    parser.add_argument('--perform-layer', type=int, default=33)
    parser.add_argument('--dump-feats', type=str, default="False")
    parser.add_argument('--dump-task', type=str, default=None)
    parser.add_argument('--dump-feat-layer', type=int, default=33)

    args = parser.parse_args()
    args.dump_feats = args.dump_feats.lower() == "true"
    # some restrictions
    if args.method == "kv_evict" or args.method == "atome": 
        # kv_evict and atome only support perform_layer = 0
        args.perform_layer = 0
    if args.method == "atome": 
        # atome only support constant schedule
        args.schedule = "constant"
    if args.method != "merge_weighted" and args.method != "atome":
        # other methods only support none schedule
        args.schedule = "none"
    return args

def calculate_ratio(mem_reduce_rate, method, perform_layer, schedule="none") -> float:
    # this function calculate drop ratio by memory reduce rate, method, and schedule
    model_layer_num = 32
    assert perform_layer < model_layer_num
    
    if method == "merge_kv_evict":
        return 1 - np.sqrt((((1 - mem_reduce_rate) * model_layer_num) - (perform_layer + 1)) / (model_layer_num - 1 - perform_layer))

    if method == "kv_evict" or method == "atome":
        perform_layer = 0
    if method == "atome":
        schedule = "constant"
    if method != "merge_weighted" and method != "atome":
        schedule = "none"

    max_mem_reduce_rate = 1.0 - ((perform_layer + 1) / model_layer_num)
    if not (0.0 <= mem_reduce_rate <= max_mem_reduce_rate):
        raise ValueError(f"mem_reduce_rate of {mem_reduce_rate} is not possible for perform_layer {perform_layer}.")

    if schedule == "none":
        ratio = 1.0 - (model_layer_num - model_layer_num*mem_reduce_rate - (perform_layer+1)) / (model_layer_num - (perform_layer+1))
    elif schedule == "constant":
        if mem_reduce_rate < 0.05:
            ratio = 0.0
        else:
            ratio = 0
            r = 0.001
            for _ in range(1000):
                result = 1 - (((perform_layer + 1) + (1 - r) * (1 - (1 - r)**(31 - perform_layer)) / (1 - (1 - r))) / 32)
                if abs(result - mem_reduce_rate) <= 0.005:
                    ratio = r
                    break
                r += 0.001
    elif schedule == "decay":
        if mem_reduce_rate < 0.05:
            ratio = 0.0
        else:
            ratio = 0
            r = 0.001
            for _ in range(1000):
                result = perform_layer + 1
                current = 1
                for l in range(perform_layer, 32):
                    current *= (1 - l * r / (perform_layer - 31) + 31 * r / (perform_layer - 31))
                    result += current
                result = 1 - result / 32   
                if abs(result - mem_reduce_rate) <= 0.005:
                    ratio = r
                    break
                r += 0.001
    else:
        raise ValueError(f"schedule {schedule} is not supported")
    
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Calculated ratio of {ratio} is out of bounds (0 to 1).")
    
    if method == "atome":
        # atome needs to be doubled because it merges two tokens at a time
        ratio *= 2
        if ratio > 1.0:
            ratio = 1.0
        atome_upper_bound = 0.5
        if mem_reduce_rate > atome_upper_bound:
            raise ValueError(f"mem_reduce_rate of {mem_reduce_rate} is not possible for atome.")
    
    return ratio