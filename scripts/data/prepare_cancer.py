import pandas as pd
import numpy as np
from math import log, e
from collections import Counter
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight

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

def label_crit(row):
    relevant, cancer, desc = row[[
        'is_this_text_relevant_or_irrelevant_to_breast_brain_or_colon_cancer',
        'what_type_of_cancer_is_described_in_the_text',
        'the_text_describes'
        ]]

    if relevant == 'relevant':
        if cancer == 'breast_cancer':
            label = 0
        elif cancer == 'colon_cancer':
            label = 1
        elif cancer == 'brain_cancer':
            label = 2
        else:
            label = -1
    elif relevant == 'irrelevant':
        if desc == 'breast_brain_or_colon_cancer_in_passing_no_patient_experience':
            label = 3
        elif desc == 'none_of_the_above':
            label = 4
        elif desc == 'other_cancer_types':
            label = 5
        elif desc == 'scientific_publications_news_or_advertisements_related_to_cancer':
            label = 6
    return label

def process(labels):
    ent = entropy(labels)
    if ent < 0.5:
        ent_class = 0
    # elif ent < 0.563:
    elif ent < 0.637:
        ent_class = 1
    else:
        ent_class = 2
    return ent_class

def get_weights():
    weights_ent = compute_class_weight('balanced', classes=list(range(3)),
            y=data_grouped['entropy_class'])
    weights_target = [compute_class_weight('balanced', classes=list(range(7)),
        y=data_grouped[data_grouped.entropy_class == l]['label']) for l in range(3)]
    return weights_target, weights_ent

def weight(row):
    label, ent_cat = row[['label', 'entropy_class']]
    weight = weights_target[ent_cat][label] * weights_ent[ent_cat]
    return weight

def diff(labels):
    label = Counter(labels).most_common(1)[0][0]
    diff = 1 - sum([l == label for l in labels])/len(labels)

    return diff

if __name__ == '__main__':
    data = pd.read_csv('data/cancer/cancer_all.csv')
    data.set_index('_unit_id', inplace=True)
    data['label'] = data.apply(label_crit, axis = 1)
    data_grouped = data.groupby('_unit_id').first()
    ids = data.index.unique()
    labels = data.index.unique().map(lambda x: data.loc[x]['label'])
    data_grouped['labels'] = labels
    data_grouped['entropy'] = data_grouped['labels'].map(entropy)
    data_grouped['entropy_class'] = data_grouped['labels'].map(process)
    data_grouped['label'] = data_grouped['labels']\
        .map(lambda x: Counter(x).most_common(1)[0][0])
    data_grouped['diff'] = data_grouped['labels'].map(diff)
    data_grouped = data_grouped[data_grouped.label != -1]
    data_grouped = data_grouped[[
    'text_content',
    'label',
    'entropy_class',
    'entropy',
    'diff']]
    data_grouped.rename({'text_content': 't'},
        axis = 1, inplace=True)
    data_grouped.reset_index(inplace=True, drop=True)
    weights_target, weights_ent = get_weights()
    data_grouped['ins_weight'] = data_grouped.apply(weight, axis = 1)


    train, dev, test = np.split(data_grouped.sample(frac=1),
            [int(.6*len(data_grouped)), int(.8*len(data_grouped))])

    data = DatasetDict({
        'train': Dataset.from_dict(train),
        'dev': Dataset.from_dict(dev),
        'test': Dataset.from_dict(test)
        })

    data.save_to_disk('data/cancer')
