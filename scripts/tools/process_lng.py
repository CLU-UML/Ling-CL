import sys
import numpy as np
import pandas as pd
from lng.lca.lc_anc import lca
from lng.L2SCA.analyzeText import sca
from datasets import load_from_disk
from sklearn.preprocessing import StandardScaler
from numpy import mean
from nltk import word_tokenize

dataset = sys.argv[1]
data = load_from_disk(dataset)

def process_l(x, standard1, standard2, standard):
    if 'sentence1_lca' in x and len(x['sentence1_lca']) != 1:
        return None
    if 'sentence1' in x:
        lca1 = lca(x['sentence1'], standard1)
        lca2 = lca(x['sentence2'], standard2)
        return {
                'sentence1_lca': lca1,
                'sentence2_lca': lca2,
                }
    elif 't' in x:
        lca_t = lca(x['t'], standard)
        return {'t_lca': lca_t}
    elif 'sentence' in x:
        lca_t = lca(x['sentence'], standard)
        return {'lca': lca_t}

def process_s(x):
    if 'sentence1_sca' in x and len(x['sentence1_sca']) != 1:
        return None
    if 'sentence1' in x:
        sca1 = sca(x['sentence1'])
        sca2 = sca(x['sentence2'])
        return {
                'sentence1_sca': sca1,
                'sentence2_sca': sca2,
                }
    elif 't' in x:
        sca_t = sca(x['t'])
        return {'t_sca': sca_t}
    elif 'sentence' in x:
        sca_t = sca(x['sentence'])
        return {'sca': sca_t}

standard1, standard2, standard = None, None, None
if 'sentence1' in data['train'].column_names:
    sample_size = 300000
    if len(data['train']) > sample_size:
        sents1 = np.random.choice(data['train']['sentence1'], sample_size, replace=False)
        sents2 = np.random.choice(data['train']['sentence2'], sample_size, replace=False)
    else:
        sents1 = data['train']['sentence1']
        sents2 = data['train']['sentence2']

    lens1 = [len(word_tokenize(x)) for x in sents1]
    lens2 = [len(word_tokenize(x)) for x in sents2]
    standard1 = int(np.percentile(lens1, 20))
    standard2 = int(np.percentile(lens2, 20))
elif 't' in data['train'].column_names:
    lens = [len(word_tokenize(x)) for x in data['train']['t']]
    standard = int(np.percentile(lens, 20))
elif 'sentence' in data['train'].column_names:
    lens = [len(word_tokenize(x)) for x in data['train']['sentence']]
    standard = int(np.percentile(lens, 20))

data = data.map(process_s); exit()

data = data.map(process_l,
        # num_proc=128,
        num_proc=30,
        fn_kwargs = {
            'standard': standard,
            'standard1': standard1,
            'standard2': standard2}
        )

data = data.map(process_s,
        num_proc = 128
        # num_proc=30,
        )

print(data)

#data.save_to_disk(dataset)
