import sys
import numpy as np
from os.path import basename
from glob import glob
from datasets import load_from_disk

dataset = sys.argv[1]
data = load_from_disk(dataset)

losses = [np.load(fn) for fn in glob('/data/mohamed/losses/%s*'%basename(dataset))]
train_loss = [l['train'] for l in losses]
dev_loss = [l['dev'] for l in losses]

if train_loss[0].ndim == 3:
    for i in range(len(train_loss)):
        train_loss[i] = train_loss[i].sum(2) / (train_loss[i] != 0).sum(2)
    for i in range(len(dev_loss)):
        dev_loss[i] = dev_loss[i].sum(2) / (dev_loss[i] != 0).sum(2)

train_loss = np.mean(train_loss, (0,1))
dev_loss = np.mean(dev_loss, (0,1))
test_loss = [0] * len(data['test'])

data['train'] = data['train'].add_column('loss', train_loss)
data['dev'] = data['dev'].add_column('loss', dev_loss)
data['test'] = data['test'].add_column('loss', test_loss)
data = data.map(lambda x: x)

print(data)
data.save_to_disk(dataset)
