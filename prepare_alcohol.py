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

def process(labels):
    ent = entropy(labels)
    if ent == 0:
        ent_class = 0
    elif ent < 0.75:
        ent_class = 1
    else:
        ent_class = 2
    return ent_class

def label_crit(row):
    alcohol, intensity, social = row[[
        'is_this_text_about_a_person_or_a_group_of_people_drinking_alcohol',
        'what_is_the_intensity_of_alcohol_consumption_',
        'who_is_the_drinker_in_the_post_',
        ]]
    if alcohol == 'yes':
        if intensity == 'heavy_use_multiple_drinks_or_intoxication_mentioned':
            if social == 'only_author_of_the_post':
                label = 0
            elif social == 'a_group_of_people_including_author_of_the_post':
                label = 1
            elif social == 'others':
                label = 2
        elif intensity == 'light_use_single_drink_and_no_intoxication':
            if social == 'only_author_of_the_post':
                label = 3
            elif social == 'a_group_of_people_including_author_of_the_post':
                label = 4
            elif social == 'others':
                label = -1
        elif intensity == 'not_sure':
            if social == 'only_author_of_the_post':
                label = 5
            elif social == 'a_group_of_people_including_author_of_the_post':
                label = -1
            elif social == 'others':
                label = -1
    elif alcohol == 'no':
        label = 6
    return label



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

cls_map = {
        "yes": 0,
        "no": 1,
        }

def diff(labels):
    label = Counter(labels).most_common(1)[0][0]
    diff = 1 - sum([l == label for l in labels])/len(labels)

    return diff

if __name__ == '__main__':
    data = pd.read_csv('/data/mohamed/data/alcohol_raw/alcohol_all.csv',
            usecols = ['_unit_id',
                'is_this_text_about_a_person_or_a_group_of_people_drinking_alcohol',
                'what_is_the_intensity_of_alcohol_consumption_',
                'who_is_the_drinker_in_the_post_',
                't']
            )
    data.set_index('_unit_id', inplace=True)
    data_grouped = data.groupby('_unit_id').first()
    ids = data.index.unique()
    data['label'] = data.apply(label_crit, axis = 1)
    labels = data.index.unique().map(lambda x: data.loc[x]['label'])
    data_grouped['labels'] = labels
    data_grouped['entropy'] = data_grouped['labels'].map(entropy)
    data_grouped['entropy_class'] = data_grouped['labels'].map(process)
    data_grouped['label'] = data_grouped['labels']\
            .map(lambda x: Counter(x).most_common(1)[0][0])
    data_grouped['diff'] = data_grouped['labels'].map(diff)
    data_grouped.reset_index(inplace=True, drop=True)
    data_grouped = data_grouped[data_grouped.label != -1]
    weights_target, weights_ent = get_weights()
    data_grouped['ins_weight'] = data_grouped.apply(weight, axis = 1)
    data_grouped = data_grouped[['t', 'label', 'entropy_class', 'entropy', 'ins_weight', 'diff']]

    train, dev, test = np.split(data_grouped.sample(frac=1), [int(.6*len(data_grouped)), int(.8*len(data_grouped))])

    data = DatasetDict({
        'train': Dataset.from_dict(train),
        'dev': Dataset.from_dict(dev),
        'test': Dataset.from_dict(test)
        })

    data.save_to_disk('/data/mohamed/data/alcohol')
