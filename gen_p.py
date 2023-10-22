sca_names = "W,S,VP,C,T,DC,CT,CP,CN,MLS,MLT,MLC,C-S,VP-T,C-T,DC-C,DC-T,T-S,\
CT-T,CP-T,CP-C,CN-T,CN-C".split(',')
lca_names = "wordtypes,swordtypes,lextypes,slextypes,wordtokens,swordtokens,\
lextokens,slextokens,ld,ls1,ls2,vs1,vs2,cvs1,ndw,ndwz,ndwerz,ndwesz,ttr,\
msttr,cttr,rttr,logttr,uber,lv,vv1,svv1,cvv1,vv2,nv,adjv,advv,modv".split(',')

import pandas as pd
import numpy as np

df = pd.read_csv('lng_corr.csv')

ps = []

dataset = 'chaosnli'

subdf = df[(df['dataset'] == dataset) & (df['with'] == 'ent') & (df['split'] == 'train')
        & (df['text'] == 'sentence1')]
subdf = subdf.set_index('metric')

for n in lca_names + sca_names:
    ps.append(subdf.loc[n]['corr'])

subdf = df[(df['dataset'] == dataset) & (df['with'] == 'ent') & (df['split'] == 'train')
        & (df['text'] == 'sentence2')]
subdf = subdf.set_index('metric')

for n in lca_names + sca_names:
    ps.append(subdf.loc[n]['corr'])

np.save(f'{dataset}_p.npy', ps)
