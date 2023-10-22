import json
from datasets import Dataset, DatasetDict

train = json.load(open('/data/mohamed/data/qanta_raw/qanta.train.json'))['questions']
dev = json.load(open('/data/mohamed/data/qanta_raw/qanta.dev.json'))['questions']
test = json.load(open('/data/mohamed/data/qanta_raw/qanta.test.json'))['questions']

train = Dataset.from_list(train).rename_columns({'category': 'label', 'text': 'sentence'})
train = train.remove_columns(set(train.column_names) - {'sentence', 'label', 'difficulty'})

dev = Dataset.from_list(dev).rename_columns({'category': 'label', 'text': 'sentence'})
dev = dev.remove_columns(set(dev.column_names) - {'sentence', 'label', 'difficulty'})

test = Dataset.from_list(test).rename_columns({'category': 'label', 'text': 'sentence'})
test = test.remove_columns(set(test.column_names) - {'sentence', 'label', 'difficulty'})

data = DatasetDict({'train': train, 'dev': dev, 'test': test})
print(data)
data.save_to_disk('/data/mohamed/data/qanta')
