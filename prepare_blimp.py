from glob import glob
from datasets import load_dataset, DatasetDict
from random import random

def process_labels(good, bad):
    if random() > 0.5:
        sent1 = good
        sent2 = bad
        label = 0
    else:
        sent1 = bad
        sent2 = good
        label = 1
    return {'sentence1': sent1, 'sentence2': sent2, 'label': label}

if __name__ == '__main__':
    files = glob('/data/mohamed/data/blimp_data/*')
    data = load_dataset('json', data_files=files)
    cols = ['sentence_good', 'sentence_bad']
    data = data.map(process_labels, input_columns=cols, remove_columns=cols)
    data = data['train'].shuffle()
    n = len(data)
    train = data.select(range(int(0.7*n)))
    val = data.select(range(int(0.7*n), int(0.8*n)))
    test = data.select(range(int(0.8*n), n))
    data = DatasetDict({
        'train': train, 'val': val, 'test': test
        })

    data.save_to_disk('/data/mohamed/data/blimp')
