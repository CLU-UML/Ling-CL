from datasets import load_from_disk
from collections import Counter

data = load_from_disk('data/snli')
train_data = data['train']

# data = load_from_disk('data/snli_special')
# print(Counter(data['dev']['entropy_class']))
# print(Counter(data['test']['entropy_class']))
# exit()

import numpy as np
# tests = np.load('test_samples.npz')
# print(Counter(data['test'].select(tests['easy'])['entropy_class']))
# print(Counter(data['test'].select(tests['med'])['entropy_class']))
# print(Counter(data['test'].select(tests['hard'])['entropy_class']))

# data['train'] = data['train'].shuffle().select(range(55000))

train_ids = [idx for idx,x in enumerate(train_data) if len(x['labels']) >= 4]
data['train'] = train_data.select(train_ids)

# data.save_to_disk('data/snli_special')
