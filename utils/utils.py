import torch
import numpy as np
from sklearn.metrics import recall_score
mean = lambda l: sum(l)/len(l) if len(l) > 0 else 0.

def ignore_padding_flatten(preds, attention_mask):
    attention_mask[:,0] = 0
    attention_mask = attention_mask.bool()
    return torch.cat([x[mask] for (x, mask) in zip(preds, attention_mask)])

num_bins = 5
def calc_bal_acc(labels, preds, metric, name, ids_map):
    bins = np.histogram_bin_edges(metric, bins=num_bins)
    bin_ids = []
    for i in range(len(bins)):
        if i == (len(bins) - 1):
            ids = (metric >= bins[i])
        else:
            ids = (metric >= bins[i]) & (metric < bins[i+1])
        if sum(ids) == 0:
            continue
        bin_ids.append(ids)
    score_bins = []
    for ids in bin_ids:
        if len(ids_map) > 0:
            ids = [ids_map[idx] for idx, val in enumerate(ids) if val == True]
            ids = [x for y in ids for x in y]
            ids = np.array(ids, dtype=int)
        score = recall_score(labels[ids], preds[ids], average='macro')
        score_bins.append(score)
    return mean(score_bins)
