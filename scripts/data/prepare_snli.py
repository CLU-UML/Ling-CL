from datasets import load_dataset
import numpy as np
from math import log, e
from collections import Counter

def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        ent = 0
    else:
        value,counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            ent = 0
        else:
            ent = 0.
            # Compute entropy
            base = e if base is None else base
            for i in probs:
                ent -= i * log(i, base)

    return ent

def process(labels, gold_label, base=None):
    """ Computes entropy of label distribution. """

    labels = [cls_map[l] for l in labels if l != '']
    label = Counter(labels).most_common(1)[0][0]
    gold_label = cls_map[gold_label]

    n_labels = len(labels)

    if n_labels <= 1:
        ent = 0
        cat = 2
    else:
        value,counts = np.unique(labels, return_counts=True)
        if n_labels == 5:
            if 5 in counts:
                cat = 0
            elif 4 in counts and 1 in counts:
                cat = 0
            elif 3 in counts and 2 in counts:
                cat = 1
            elif 3 in counts and 1 in counts:
                cat = 1
            elif 2 in counts and 1 in counts:
                cat = 2
        elif n_labels == 4:
            if 4 in counts:
                cat = 0
            elif 3 in counts:
                cat = 1
            elif 2 in counts:
                cat = 2
        elif n_labels == 3:
            if 3 in counts:
                cat = 0
            elif 2 in counts:
                cat = 1
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            ent = 0
            cat = 0
        else:
            ent = 0.
            # Compute entropy
            base = e if base is None else base
            for i in probs:
                ent -= i * log(i, base)

    weight1 = weights_target[cat][label]
    weight2 = weights_ent[cat]
    weight = weight1 * weight2

    diff = 1 - sum([l == label for l in labels])/len(labels)

    return dict(labels=labels,
            diff=diff,
            label=label,
            gold_label=gold_label,
            entropy=ent,
            entropy_class=cat,
            ins_weight=weight)

cls_map = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
        '-': 3}

if __name__ == '__main__':
    data = load_dataset('json', data_files={
        name: f'data/snli_1.0/snli_1.0_{name}.jsonl'
        for name in ['train', 'dev', 'test'] })

    weights = np.load('balance_weight.npz')
    weights_target = weights['target']
    weights_ent = weights['ent']

    data = data.map(process,
        input_columns=['annotator_labels', 'gold_label'],
        remove_columns=['annotator_labels', 'sentence1_binary_parse',
            'sentence1_parse', 'sentence2_binary_parse', 'sentence2_parse',
            'captionID'])

    data.save_to_disk('data/snli')

    train_ids = [idx for idx,x in enumerate(data['train']) if len(x['labels']) >= 4]
    data['train'] = data['train'].select(train_ids)

    data.save_to_disk('data/snli_special')
