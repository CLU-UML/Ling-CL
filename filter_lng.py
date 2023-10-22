from datasets import load_from_disk
from sklearn import metrics
from scipy.stats import describe 
import numpy as np
import pandas as pd
import os
from glob import glob
from const import *
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

data_dir = '/data/mohamed/data'
preds_dir = '/data/mohamed/preds'
datasets = ['anli', 'cola', 'rte', 'snli']
bin_method = 10
split = 'dev'

for dataset in datasets:
    print(dataset)
    data = load_from_disk(os.path.join(data_dir, dataset))
    df = pd.DataFrame(data[split])

    dataset_preds = glob(os.path.join(preds_dir, f'*{dataset}_none*dev*'))
    dates = [os.path.basename(pred).split('_')[0] for pred in dataset_preds]
    recent = np.argmax(dates)
    preds = np.load(dataset_preds[recent])

    if dataset in ['snli', 'anli', 'rte']:
        lng = np.concatenate([np.array(data[split]['sentence1_lca']),
            np.array(data[split]['sentence1_sca']),
            np.array(data[split]['sentence2_lca']),
            np.array(data[split]['sentence2_sca']),
            np.array(data[split]['sentence1_lingfeat']),
            np.array(data[split]['sentence2_lingfeat'])], axis=1)
        lng_names = lca_names + sca_names + lingfeat_names \
                + lca_names + sca_names + lingfeat_names
    else:
        lng = np.concatenate([np.array(data[split]['lca']),
            np.array(data[split]['sca']),
            np.array(data[split]['sentence_lingfeat'])], axis=1)
        lng_names = lca_names + sca_names + lingfeat_names

    slopes = []
    for idx in tqdm(range(lng.shape[1])):
        x = lng[:,idx]
        if len(set(x)) == 1:
            continue
        bins = np.histogram_bin_edges(x, bins=bin_method)
        score_bins = []
        bins_x = []
        for i in range(len(bins)):
            if i == (len(bins) - 1):
                ids = (x >= bins[i])
            else:
                ids = (x >= bins[i]) & (x < bins[i+1])
            if sum(ids) == 0:
                continue
            score = metrics.recall_score(df['label'][ids], preds[ids], average='macro')
            score_bins.append(score)
            bins_x.append(bins[i])

        z = np.polyfit(bins_x, score_bins, 1)
        slopes.append(z[0])
    slopes = np.array(slopes)
    ids = np.where(slopes >= 10)[0]
    slopes = abs(slopes)
    print(np.percentile(slopes, 10), np.percentile(slopes, 30),
            np.percentile(slopes, 50), np.percentile(slopes, 70),
            np.percentile(slopes, 90), np.percentile(slopes, 99))
