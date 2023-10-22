import sys
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from collections import Counter

np.random.seed(0)

dataset = sys.argv[1]
data = load_from_disk(dataset)
data.remove_columns_('ins_weight')
train_data = data['train']
train_data.shuffle(seed=0)
min_count = min(np.bincount(train_data['entropy_class']))

easy_ids, med_ids, hard_ids = [], [], []

for idx, x in tqdm(enumerate(train_data['entropy_class']), total=len(train_data)):
    if x == 0 and len(easy_ids) < min_count:
        easy_ids.append(idx)
    elif x == 1 and len(med_ids) < min_count:
        med_ids.append(idx)
    elif x == 2 and len(hard_ids) < min_count:
        hard_ids.append(idx)

ids = easy_ids + med_ids + hard_ids 

data['train'] = train_data.select(ids)

print(data)
data.save_to_disk(dataset + '_balanced')

"""
target_ids = [[idx for idx in easy_ids if train_data[idx]['label'] == t] for t in [0,1,2]]\
        + [[idx for idx in med_ids if train_data[idx]['label'] == t] for t in [0,1,2]]\
        + [[idx for idx in hard_ids if train_data[idx]['label'] == t] for t in [0,1,2]]
min_count = min([len(x) for x in target_ids])
target_ids = [x[:min_count] for x in target_ids]
target_ids = [x for l in target_ids for x in l]

data['train'] = train_data.select(target_ids)
data.save_to_disk('data/snli_super_balanced')
"""
