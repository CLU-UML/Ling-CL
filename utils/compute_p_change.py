import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import multiprocessing as mp

df = pd.read_csv('lng_combs_small.csv')

subdfs = {data: {text: df[(df.dataset == data) & (df.text == text)]
    for text in ('sentence1', 'sentence2')}
    for data in ('anli', 'chaosnli')}

def increase_of(irow):
    i, row = irow
    subdf = subdfs[row.dataset][row.text]
    vals = [subdf[subdf.metric == metric]['abs(corr)'].iat[0]
            for metric in row['metric'].split()]
    return row['abs(corr)'] - max(vals), ' '.join([str(round(x,3)) for x in vals])

pool = mp.Pool(mp.cpu_count())

res = list(tqdm(pool.imap(increase_of, df.iterrows(), chunksize=100), total=df.shape[0]))

inc, vals = zip(*res)
df['increase'] = inc
df['vals'] = vals

df.to_csv('lng_combs_small_inc.csv')
