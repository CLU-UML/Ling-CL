from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd
from glob import glob
import string

lines = []
files = glob('/data/mohamed/data/fce-error-detection/tsv/*')
for fn in files:
    with open(fn) as f:
        lines.extend(f.readlines())

texts = []
labels = []
offset = 0
s = ''
sent_labels = []
for line in lines:
    line = line.strip()
    if line == '':
        texts.append(s.strip())
        labels.append(sent_labels)
        offset = 0
        s = ''
        sent_labels = []
    else:
        w, label = line.split('\t')
        if label == 'i':
            start = offset
            end = start + len(w)
            sent_labels.append((start, end))
        if not any(x in string.punctuation for x in w[:2]):
            s += ' '
            offset += 1
        s += w
        offset += len(w)

data = Dataset.from_dict({'sentence': texts, 'labels': labels})
train, dev, test = np.split(data.shuffle(), [int(.8*len(data)), int(.9*len(data))])

data = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame.from_records(train)),
    'dev': Dataset.from_pandas(pd.DataFrame.from_records(dev)),
    'test': Dataset.from_pandas(pd.DataFrame.from_records(test))
    })
# data.save_to_disk('/data/mohamed/data/fce-error-detection')
