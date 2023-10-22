from datasets import Dataset, DatasetDict
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

tree = ET.parse('/data/mohamed/data/an-dataset/an-dataset.xml')
root = tree.getroot()

def parse_label(label):
    correct, adj, noun = label.split('-')
    if 'i' in correct:
        return 0
    if adj != 'J' or noun != 'N':
        return 0
    return 1

texts = []
labels = []
addons = []
for i in range(len(root)):
    addon = root[i].attrib['lem'].split('_')[0].strip()
    text = ET.tostring(root[i][1], encoding='unicode', method='text').strip()
    label = parse_label(root[i][0].text.split()[0])

    texts.append(text)
    labels.append(label)
    addons.append(addon)


data = Dataset.from_dict({'sentence': texts, 'labels': labels, 'addon': addons})
train, dev, test = np.split(data.shuffle(), [int(.8*len(data)), int(.9*len(data))])

data = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame.from_records(train)),
    'dev': Dataset.from_pandas(pd.DataFrame.from_records(dev)),
    'test': Dataset.from_pandas(pd.DataFrame.from_records(test))
    })
data.save_to_disk('/data/mohamed/data/an-dataset')
