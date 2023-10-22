import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sys
import numpy as np


def name(dataset, curr):
    d = dataset[0]
    if d == 'a': 
        d = 't'
    elif d == 'c':
        d = 'r'
    c = curr[0]
    b = 'B' if 'balanced' in dataset else 'F'

    if c == 'n':
        return 'none'
    else:
        return '%s-%s-%s'%(d.upper(),b,c.upper())

cfg_dir = {
        'none': 5,
        'S-B-E': 8,
        'S-B-L': 12,
        'S-F-E': 20,
        'S-F-L': 19,
        'T-B-E': 9,
        'T-B-L': 13,
        'T-F-E': 15,
        'T-F-L': 16,
        'R-B-E': 10,
        'R-B-L': 14,
        'R-F-E': 17,
        'R-F-L': 18,
        }

df = pd.read_csv(sys.argv[1])
df.rename(lambda x: x.split('.')[1] if '.' in x else x,
        axis = 1, inplace = True)
for col in ['data', 'curr', 'ent_cfg']:
    df[col] = df[col].apply(lambda x: x.replace('"', ''))

df = df[df['data'] != '-']
df['ent_cfg'] = df['ent_cfg'].astype('int')
datasets = df['data'].unique()
datasets = [(data, curr) for data in datasets for curr in df['curr'].unique() if curr != 'none']
cfgs = [6] + [cfg_dir[name(d, c)] for d,c in datasets]
df.drop(columns = ['run'], inplace = True)
df.acc = df.acc.astype('float')
df = df.groupby(['data', 'curr', 'ent_cfg']).mean()

full_datasets = datasets
datasets = [x for x in datasets if not 'balanced' in x[0]]
matrix = [[df.loc[data, curr, cfg].acc if (data, curr, cfg) in df.index else 0 for cfg in cfgs]
        for (data, curr) in datasets]
matrix = np.array(matrix)

none_acc = df.loc[:, :, 5].acc
nones = np.array([none_acc.loc[data].item() for data, curr in datasets])
matrix = np.concatenate([nones[:,np.newaxis], matrix], 1)
matrix = matrix / matrix.max(1)[:, np.newaxis] * 100

# names = [name(d,c) for d,c in datasets]
# col_names = ['No-CL', 'inc'] + names

# col_sort = matrix.mean(0).argsort()[::-1]
# matrix = matrix[:, col_sort]
# col_names = [col_names[i] for i in col_sort]

# row_sort = matrix.mean(1).argsort()[::-1]
# matrix = matrix[row_sort, :]
# row_names = [names[i] for i in row_sort]

# from collections import defaultdict
# d = defaultdict(list)
# for i,row in enumerate(row_names):
#     for j,col in enumerate(col_names):
#         if 'E' in row:
#             if 'E' in col:
#                 d['E>E'].append(matrix[i,j])
#             elif 'L' in col:
#                 d['L>E'].append(matrix[i,j])
#         elif 'L' in row:
#             if 'E' in col:
#                 d['E>L'].append(matrix[i,j])
#             elif 'L' in col:
#                 d['L>L'].append(matrix[i,j])
# d = {k: np.mean(v) for k,v in d.items()}
# print(d)



# fig, ax = plt.subplots(2,1, sharex = True,
#         figsize = (15,10), dpi = 300,
#         gridspec_kw={'height_ratios': [10, 1]})
# sns.heatmap(matrix,
#         ax = ax[0],
#         vmin = 95,
#         annot=True, fmt=".1f",
#         xticklabels = col_names,
#         yticklabels = row_names, 
#         cbar=False,
#         )
# ax[0].set_ylabel("Models")

# sns.heatmap(matrix.mean(0)[np.newaxis,:],
#         ax = ax[1],
#         annot=True, fmt=".1f",
#         vmin = 95,
#         xticklabels = col_names,
#         yticklabels = ['avg'],
#         cbar=False
#         )
# ax[1].set_xlabel("Configurations")
# plt.savefig('nxn_full.pdf')

matrix_avg = matrix.mean(0)[np.newaxis,:]

names = [name(d,c) for d,c in datasets]
full_names = ['No-CL', 'inc.'] + [name(d,c) for d,c in full_datasets]

sort_ids = matrix_avg.argsort()[0][::-1]
matrix = matrix[:, sort_ids]
matrix_avg = matrix_avg[:, sort_ids]
full_names = [full_names[i] for i in sort_ids]

matrix_avg_rows = matrix.mean(1)
sort_ids_rows = matrix_avg_rows.argsort()[::-1]
matrix = matrix[sort_ids_rows, :]
names = [names[i] for i in sort_ids_rows]

fig, ax = plt.subplots(2,1, sharex = True,
        # dpi = 300, figsize = (10, 6.2),
        dpi = 300, figsize = (10, 6),
        gridspec_kw={'height_ratios': [10, 1]})
g = sns.heatmap(matrix,
        square = True,
        ax = ax[0],
        vmin = 97,
        linewidths = 0.1,
        linecolor = 'black',
        annot=True, fmt=".1f",
        xticklabels = full_names,
        yticklabels = names, 
        cbar=False,
        )
ax[0].set_ylabel("Models")
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')

g = sns.heatmap(matrix_avg,
        ax = ax[1],
        annot=True, fmt=".1f",
        linewidths = 0.1,
        linecolor = 'black',
        vmin = 97,
        xticklabels = full_names, 
        yticklabels = ['Avg.'],
        cbar=False
        )
ax[1].set_xlabel("Configurations")
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')
plt.xticks(rotation = 45)
# plt.savefig('vis/nxn.pdf', bbox_inches='tight')
plt.savefig('vis/nxn.pdf', bbox_inches='tight')


# plt.show()

# matrix_c = [[], []]
# for i, name in enumerate(names):
#     if 'e' in name:
#         matrix_c[0] += matrix[:,i+1].tolist()
#     elif 'l' in name:
#         matrix_c[1] += matrix[:,i+1].tolist()
# matrix_c = np.array(matrix_c).mean(1)

# matrix_b = [[], []]
# for i, name in enumerate(names):
#     if 'b' in name:
#         matrix_b[0] += matrix[:,i+1].tolist()
#     elif 'f' in name:
#         matrix_b[1] += matrix[:,i+1].tolist()
# matrix_b = np.array(matrix_b).mean(1)

# matrix_d = [[], [], []]
# for i, name in enumerate(names):
#     if 'S' in name:
#         matrix_d[0] += matrix[:,i+1].tolist()
#     elif 'A' in name:
#         matrix_d[1] += matrix[:,i+1].tolist()
#     elif 'C' in name:
#         matrix_d[2] += matrix[:,i+1].tolist()
# matrix_d = np.array(matrix_d).mean(1)


# fig, ax = plt.subplots(3,1, 
#         gridspec_kw={'height_ratios': [1, 1, 1]})
# sns.heatmap(matrix_c[np.newaxis,:],
#         ax = ax[0],
#         annot=True, fmt=".2f",
#         xticklabels = ['e', 'l'],
#         yticklabels = [],
#         cbar=False
#         )

# sns.heatmap(matrix_d[np.newaxis,:],
#         ax = ax[1],
#         annot=True, fmt=".2f",
#         xticklabels = ['S', 'A', 'C'],
#         yticklabels = [],
#         cbar=False
#         )
# sns.heatmap(matrix_b[np.newaxis,:],
#         ax = ax[2],
#         annot=True, fmt=".2f",
#         xticklabels = ['b', 'f'],
#         yticklabels = [],
#         cbar=False
#         )
# plt.show()
