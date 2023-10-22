import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_ind, wilcoxon, ttest_rel
from matplotlib.ticker import FormatStrFormatter


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--vis', default='')
parser.add_argument('--data', default='bal')
parser.add_argument('--name')
args = parser.parse_args()

clrs = {
        'Ent (sp)': 'tab:orange',
        'Ent (inc)': 'tab:orange',
        'Loss (sp)': 'tab:blue',
        'Loss (inc)': 'tab:blue',
        'Ent (inc+loss)': 'tab:cyan',
        # 'ent_dec': 'orange',
        # 'No-CL': 'tab:gray',
        'No-CL': 'ghostwhite',
        'SL': 'tab:pink',
        'SL (avg+ent)': 'pink',
        'MentorNet': 'tab:red',
        'DP': 'tab:purple',
        'SPL': 'tab:olive',
        'SPL (hard)': 'tab:olive',
        'lca_class': 'lightblue',
        'sca_class': 'blue',
        'lns_class': 'darkblue'
        }

line_clrs = {
        'Ent (sp)': 'tab:orange',
        'Ent (inc)': 'tab:orange',
        'Loss (sp)': 'tab:blue',
        'Loss (inc)': 'tab:blue',
        'Ent (inc+loss)': 'tab:cyan',
        # 'ent_dec': 'orange',
        'No-CL': 'tab:gray',
        # 'No-CL': 'ghostwhite',
        'SL': 'tab:pink',
        'SL (avg+ent)': 'pink',
        'MentorNet': 'tab:red',
        'DP': 'tab:purple',
        'SPL': 'tab:green',
        'SPL (hard)': 'tab:olive',
        'lca_class': 'lightblue',
        'sca_class': 'blue',
        'lns_class': 'darkblue'
        }

def name(dataset, curr):
    if 'cancer' in dataset:
        d = 'c'
    elif 'snli' in dataset:
        d = 's'
    elif 'alcohol' in dataset:
        d = 'a'
    else:
        d = 'ch'
    # d = dataset[0]
    c = curr[0]
    b = 'B' if 'balanced' in dataset else 'F'

    if c == 'n':
        return 'none'
    else:
        return '%s-%s-%s'%(d.upper(),b,c.upper())

cfg_dir = {
        'none': 5,
        'S-B-E': [8],
        'S-B-L': [12],
        'S-F-E': [20],
        'S-F-L': [19],
        'A-B-E': [9],
        'A-B-L': [13, 21],
        'A-F-E': [15],
        'A-F-L': [16],
        'C-B-E': [10],
        'C-B-L': [14],
        'C-F-E': [17],
        'C-F-L': [18],
        }

def load_data():
    data = pd.read_csv(args.path)
    data = data[data.run != '--']
    data.rename(lambda x: x.split('.')[1] if '.' in x else x,
            axis = 1, inplace = True)
    def curr_names(row):
        # if row['lng'] and row['lng'] != 'None':
        #     row['curr'] = row['lng']

        if row['curr'] in ['ent', 'ent+', 'loss', 'loss+']:
            cfg_name = name(row['data'], row['curr'])
            if row['ent_cfg'] == '6':
                cfg = 'inc'
            elif int(row['ent_cfg']) in cfg_dir[cfg_name]:
                cfg = 'sp'
            else:
                # cfg = 'sp-'
                cfg = int(row['ent_cfg'])

        if row['curr'] == 'ent':
            row['curr'] = 'Ent (%s)'%cfg
        if row['curr'] == 'loss':
            row['curr'] = 'Loss (%s)'%cfg
        elif row['curr'] == 'ent+':
            row['curr'] = 'Ent+ (%s)'%cfg
        elif row['curr'] == 'loss+':
            row['curr'] = 'Loss+ (%s)'%cfg
        elif row['curr'] == 'spl':
            row['curr'] = 'SPL'
            # row['curr'] = 'SPL (easy)' if row['spl_mode'] == 'easy' else 'SPL (hard)'
        elif row['curr'] == 'none':
            row['curr'] = 'No-CL'
        elif row['curr'] == 'sl':
            row['curr'] = 'SL'
        elif row['curr'] == 'mentornet':
            row['curr'] = 'MentorNet'
        elif row['curr'] == 'dp':
            row['curr'] = 'DP'

        return row
    # data['lng'] = data['lng'].fillna('')
    if 'sl_mode' in data:
        data['sl_mode'] = data['sl_mode'].fillna('')
    for col in data.columns:
        data[col] = data[col].apply(lambda x: x.replace('"', '') if isinstance(x,str) else x)
    # data = data[data.curr != "ent+"]
    if 'sl_mode' in data:
        data = data[data.sl_mode != "avg+ent"]
    data = data[data.curr != "spl"]
    if 'spl_mode' in data:
        data = data[data.spl_mode != "hard"]
    data = data.apply(curr_names, axis = 1)
    data = data[data.curr != 'Ent (dec)']
    if 'data_fraction' in data:
        data = data[data.data_fraction != 0.05]
    if 'noise' in data:
        data = data[data.noise != 0.9]
    # data = data.drop(['ent_cfg', 'spl_mode', 'sl_mode'], axis = 1)
    data['acc_easy'] = data['acc_easy'].astype('float32')
    data['acc_med'] = data['acc_med'].astype('float32')
    data['acc_hard'] = data['acc_hard'].astype('float32')
    if 'acc_easy' in data:
        data['bal_acc'] = data[data.ent_classes == "3"]\
                [['acc_easy','acc_med','acc_hard']].mean(1)
    data.reset_index(inplace=True)

    if not 'epochs' in data:
        data['epochs'] = 10
    if not 'noise' in data:
        data['noise'] = 0
    if not 'data_fraction' in data:
        data['data_fraction'] = 1

    # data = data[~data.curr.isin(['Loss (sp)', 'Loss (inc)'])]

    data = data[data.acc != '-']
    data.acc = data.acc.astype('float32')
    return data

def vis1_1():
    sns.set(font_scale=0.5)
    fig, ax = plt.subplots(len(datasets),1, figsize = (3,4), sharex = True, dpi = 300)
    subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1) & (data.data == default_data)]
    order = subdata.groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs.get(x, 'lightblue') for x in order]
    fig.suptitle("Accuracy")
    ax = ax.ravel()
    for idx, dataset in enumerate(datasets):
        subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1) & (data.data == dataset)]
        bar = sns.barplot(data=subdata, x = 'curr', y = 'acc', order = order, hue_order = order, palette = pal, ax = ax[idx], capsize = 0.01, errwidth = 1, ci=68)
        ax[idx].set_ylim(subdata['acc'].min() + 0.001, subdata['acc'].max() - 0.001)
        ax[idx].set_title(dataset)
        if idx == 3:
            ax[idx].set_xlabel("curriculum")
            ax[idx].set_ylabel("acc (%)")
        else:
            ax[idx].set_xlabel(None)
            ax[idx].set_ylabel(None)

        ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        for idx in ent_ids:
            bar.containers[0][idx]._edgecolor = (0,0,0,1)
            bar.containers[0][idx]._linewidth = 1
    plt.xticks(rotation = 45)
    plt.savefig('vis/bar_acc_all1.png', bbox_inches='tight')
    plt.close()
    sns.set()

def vis1_2():
    plt.figure(figsize = (10, 10), dpi = 300)
    subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1)]
    order = subdata[subdata.data == default_data].groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs[x] for x in order]
    bar = sns.barplot(data=subdata, x = 'data', y = 'acc', hue = 'curr', hue_order = order, linewidth = 0, palette = pal, capsize = 0.01, ci=68)
    ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
    for idx in ent_ids:
        for i in range(3):
            bar.containers[idx][i]._edgecolor = (0,0,0,1)
            bar.containers[idx][i]._linewidth = 2
    plt.ylim(subdata['acc'].min() - 0.1)
    plt.title('acc')
    plt.savefig('vis/bar_acc_all2.png', bbox_inches='tight')
    plt.close()

def vis1_3():
    sns.set(font_scale=0.4)
    fig, ax = plt.subplots(1, len(datasets), figsize=(15, 5), dpi = 300)
    fig.suptitle("Accuracy")
    subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1)]
    order = subdata[subdata.data == default_data].groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs.get(x, 'lightblue') for x in order]
    for idx, dataset in enumerate(datasets):
        bar = sns.barplot(ax = ax[idx], data=subdata[subdata.data == dataset], x = 'curr', y = 'acc', order = order, hue_order = order, linewidth = 0, palette = pal, capsize = 0.01, errwidth = 1, ci=68)
        ax[idx].set_title(dataset)
        if idx == 0:
            ax[idx].set_xlabel("curriculum")
            ax[idx].set_ylabel("acc (%)")
        else:
            ax[idx].set_xlabel(None)
            ax[idx].set_ylabel(None)
        ax[idx].set_ylim(subdata[subdata.data == dataset]['acc'].min() + 0.001, subdata[subdata.data == dataset]['acc'].max())

        ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        for idx in ent_ids:
            bar.containers[0][idx]._edgecolor = (0,0,0,1)
            bar.containers[0][idx]._linewidth = 1
    plt.savefig('vis/bar_acc_all3.png', bbox_inches='tight')
    plt.close()
    sns.set()

def vis1_4():
    plt.figure(figsize = (5, 5), dpi = 300)
    subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1)]
    order = subdata.groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs.get(x, 'lightblue') for x in order]
    pal = sns.color_palette("Greys")
    bar = sns.barplot(data=subdata, x = 'curr', y = 'acc', order = order,
            hue_order = order, linewidth = 0, palette = pal, capsize = 0.1,
            ci=68
            )
    ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
    for idx in ent_ids:
        bar.containers[0][idx]._edgecolor = (0,0,0,1)
        bar.containers[0][idx]._linewidth = 3
    # plt.ylim(0.78, 0.798)
    subdata_avg = subdata.groupby('curr').mean()
    plt.ylim(subdata_avg.acc.min() - 0.01, subdata_avg.acc.max() + 0.01)
    # plt.xticks(fontsize = 15, rotation = 45)
    plt.xticks(fontsize = 25, rotation = 90)
    plt.locator_params(axis="y", nbins=4)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.yticks(
            # ticks,
            fontsize = 20)
    # plt.title('Accuracy')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.savefig('vis/bar_acc_all4.pdf', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        plt.figure(figsize = (5, 5), dpi = 300)
        subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1)
                & (data.data == dataset)]
        order = subdata.groupby('curr').mean().sort_values('acc', ascending = False).index
        pal = [clrs.get(x, 'lightblue') for x in order]
        pal = sns.color_palette("Greys")
        bar = sns.barplot(data=subdata, x = 'curr', y = 'acc', order = order,
                hue_order = order, linewidth = 0, palette = pal, capsize = 0.1,
                ci=68
                )
        ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        for idx in ent_ids:
            bar.containers[0][idx]._edgecolor = (0,0,0,1)
            bar.containers[0][idx]._linewidth = 3
        # plt.ylim(0.78, 0.798)
        subdata_avg = subdata.groupby('curr').mean()
        plt.ylim(subdata_avg.acc.min() - 0.01, subdata_avg.acc.max() + 0.01)
        # plt.xticks(fontsize = 15, rotation = 45)
        plt.xticks(fontsize = 25, rotation = 90)
        plt.locator_params(axis="y", nbins=4)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.yticks(
                # ticks,
                fontsize = 20)
        # plt.title('Accuracy')
        plt.ylabel(None)
        plt.xlabel(None)
        plt.savefig('vis/bar_acc_all4_%s.pdf'%dataset, bbox_inches='tight')
        plt.close()

def vis2_1():
    plt.figure(figsize = (15, 10), dpi = 300)
    subdata = data[(data.noise == 0) & (data.data_fraction == 1) & (data.data.isin(datasets))]
    subdata = subdata[subdata.epochs != 50]
    order = subdata[subdata.epochs == 10].groupby('curr').mean().sort_values('acc', ascending = False).index[:top_lines]
    if not 'No-CL' in order:
        order = order.insert(top_lines, 'No-CL')
    sizes = [3 if x in ('Ent (sp)', 'Ent (inc)', 'No-CL') else 1 for x in order]
    pal = [line_clrs.get(x, 'lightblue') for x in order]
    subdata = subdata[subdata.curr.isin(order)]
    sns.lineplot(data=subdata, x = 'epochs', y = 'acc', size = 'curr', hue = 'curr', style = 'curr',
            dashes = False, err_style = None, palette = pal, hue_order = order, markers = True,
            sizes = sizes)
    plt.title('shorter train (avg)')
    plt.gca().invert_xaxis()
    # plt.savefig('vis/line_epochs_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        # plt.figure(figsize = (15, 10), dpi = 300)
        plt.figure(figsize = (4.2, 3.3), dpi = 300)
        subdata = data[(data.data == dataset) & (data.noise == 0) & (data.data_fraction == 1)]
        subdata = subdata[subdata.epochs != 50]
        order = subdata[subdata.epochs == 10].groupby('curr').mean().sort_values('acc', ascending = False).index[:top_lines]
        if not 'No-CL' in order:
            order = order.insert(top_lines, 'No-CL')
        sizes = [2 if x in ('Ent (sp)', 'Ent (inc)', 'No-CL') else 1 for x in order]
        pal = [line_clrs.get(x, 'lightblue') for x in order]
        subdata = subdata[subdata.curr.isin(order)]
        sns.lineplot(data=subdata, x = 'epochs', y = 'acc', hue = 'curr', style = 'curr', dashes = False, err_style = None, palette = pal, hue_order = order, markers = True, sizes = sizes, size = 'curr')
        plt.gca().invert_xaxis()
        # plt.title('shorter train (%s)'%dataset)
        plt.legend(prop={'size': 13}, framealpha = 0.8,
            handlelength = 0.8,)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.locator_params(axis="y", nbins=4)
        plt.xlabel('Epochs', fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.savefig('vis/line_epochs_%s.pdf'%dataset, bbox_inches='tight')
        plt.close()

def vis2_2():
    plt.figure(figsize = (20, 10), dpi = 300)
    subdata = data[(data.noise == 0) & (data.data_fraction == 1) & (data.data.isin(datasets))]
    order = subdata[subdata.epochs == 10].groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs[x] for x in order]
    bar = sns.barplot(data=subdata, x = 'epochs', y = 'acc', hue = 'curr', hue_order = order, palette = pal, linewidth = 2, order = [10, 7, 5, 3, 1], capsize = 0.01)
    ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
    for idx in ent_ids:
        for i in range(5):
            bar.containers[idx][i]._edgecolor = (0,0,0,1)
    plt.title('shorter train (avg)')
    plt.ylim(subdata.acc.min() - 0.01, subdata.acc.max())
    plt.savefig('vis/bar_epochs_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        plt.figure(figsize = (16, 6.3), dpi = 300)
        subdata = data[(data.data == dataset) & (data.noise == 0) & (data.data_fraction == 1)]
        order = subdata[subdata.epochs == 10].groupby('curr').mean().sort_values('acc', ascending = False).index
        pal = [clrs[x] for x in order]
        bar = sns.barplot(data=subdata, x = 'epochs', y = 'acc', hue = 'curr', hue_order = order, palette = pal, linewidth = 4, order = [10, 7, 5, 3, 1], capsize = 0.01)
        ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        # for idx in ent_ids:
        #     for i in range(5):
        #         bar.containers[idx][i]._edgecolor = (0,0,0,1)
        for idx in range(len(bar.containers)):
            for i in range(5):
                if idx in ent_ids:
                    bar.containers[idx][i]._edgecolor = (0,0,0,1)
                else:
                    bar.containers[idx][i]._edgecolor = (0,0,0,0)
        # plt.title('shorter train (%s)'%dataset)
        plt.legend(prop={'size': 25})
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.xlabel('Epochs', fontsize = 25)
        plt.ylabel('Accuracy', fontsize = 25)
        # plt.legend()
        plt.legend(prop={'size': 25}, bbox_to_anchor = (0.3,0.5), framealpha = 0.8)
        plt.ylim(subdata.acc.min() - 0.01, subdata.acc.max())
        plt.savefig('vis/bar_epochs_%s.png'%dataset, bbox_inches='tight')
        plt.close()

def vis3_1():
    plt.figure(figsize = (15, 10), dpi = 300)
    subdata = data[(data.data_fraction == 1) & (data.epochs == 10) & (data.data.isin(datasets))]
    order = subdata[subdata.noise == 0.0].groupby('curr').mean().sort_values('acc', ascending = False).index[:top_lines]
    if not 'No-CL' in order:
        order = order.insert(top_lines, 'No-CL')
    sizes = [3 if x in ('Ent (sp)', 'Ent (inc)', 'No-CL') else 1 for x in order]
    pal = [line_clrs.get(x, 'lightblue') for x in order]
    subdata = subdata[subdata.curr.isin(order)]
    sns.lineplot(data=subdata, x = 'noise', y = 'acc', size = 'curr', style = 'curr', hue = 'curr', dashes = False, err_style = None,
            palette = pal, hue_order = order, markers = True,
            sizes = sizes)
    plt.title('noisy data (avg)')
    # plt.savefig('vis/line_noise_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        # plt.figure(figsize = (15, 10), dpi = 300)
        plt.figure(figsize = (4.2, 3.3), dpi = 300)
        subdata = data[(data.data == dataset) & (data.data_fraction == 1) & (data.epochs == 10)]
        order = subdata[subdata.noise == 0.0].groupby('curr').mean().sort_values('acc', ascending = False).index[:top_lines]
        if not 'No-CL' in order:
            order = order.insert(top_lines, 'No-CL')
        sizes = [2 if x in ('Ent (sp)', 'Ent (inc)', 'No-CL') else 1 for x in order]
        pal = [line_clrs.get(x, 'lightblue') for x in order]
        subdata = subdata[subdata.curr.isin(order)]
        sns.lineplot(data=subdata, x = 'noise', y = 'acc', style = 'curr', hue = 'curr', dashes = False, err_style = None, palette = pal, hue_order = order, markers = True, sizes = sizes, size = 'curr')
        # plt.title('noisy data (%s)'%dataset)
        plt.legend(prop={'size': 14}, framealpha = 0.8,
            handlelength = 0.8,)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.locator_params(axis="y", nbins=4)
        plt.xlabel('Noise Fraction', fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.savefig('vis/line_noise_%s.pdf'%dataset, bbox_inches='tight')
        plt.close()

def vis3_2():
    plt.figure(figsize = (20, 10), dpi = 300)
    subdata = data[(data.data_fraction == 1) & (data.epochs == 10) & (data.data.isin(datasets))]
    order = subdata[subdata.noise == 0].groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs[x] for x in order]
    bar = sns.barplot(data=subdata, x = 'noise', y = 'acc', hue = 'curr', hue_order = order, linewidth = 2, palette = pal, capsize = 0.01)
    ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
    for idx in ent_ids:
        for i in range(5):
            bar.containers[idx][i]._edgecolor = (0,0,0,1)
    plt.title('noisy data (avg)')
    plt.ylim(subdata.acc.min() - 0.1)
    plt.savefig('vis/bar_noise_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        plt.figure(figsize = (12.8, 6.3), dpi = 300)
        subdata = data[(data.data == dataset) & (data.data_fraction == 1) & (data.epochs == 10) & (data.noise < 0.8)]
        order = subdata[subdata.noise == 0].groupby('curr').mean().sort_values('acc', ascending = False).index
        pal = [clrs[x] for x in order]
        bar = sns.barplot(data=subdata, x = 'noise', y = 'acc', hue = 'curr', hue_order = order, linewidth = 4, palette = pal, capsize = 0.01)
        ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        # for idx in ent_ids:
        #     for i in range(4):
        #         bar.containers[idx][i]._edgecolor = (0,0,0,1)
        for idx in range(len(bar.containers)):
            for i in range(4):
                if idx in ent_ids:
                    bar.containers[idx][i]._edgecolor = (0,0,0,1)
                else:
                    bar.containers[idx][i]._edgecolor = (0,0,0,0)
        # plt.title('noisy data (%s)'%dataset)
        plt.legend(prop={'size': 25}, bbox_to_anchor = (0.5,0.5), framealpha = 0.8)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.xlabel('Noise Fraction', fontsize = 25)
        plt.ylabel('Accuracy', fontsize = 25)

        # plt.legend(prop={'size': 20})
        plt.ylim(subdata.acc.min() - 0.1)
        plt.savefig('vis/bar_noise_%s.png'%dataset, bbox_inches='tight')
        plt.close()

def vis4_1():
    plt.figure(figsize = (15, 10), dpi = 300)
    subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1) & (data.data.isin(datasets))].melt('curr', ['acc', 'acc_easy', 'acc_med', 'acc_hard', 'bal_acc'], 'cols', 'vals')
    order = subdata[subdata.cols == 'acc'].groupby('curr').mean().sort_values('vals', ascending = False).index
    pal = [clrs[x] for x in order]
    bar = sns.barplot(data=subdata, x = 'cols', y = 'vals', hue = 'curr',
            hue_order = order, linewidth = 2, palette = pal, capsize = 0.01)
    ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
    for idx in ent_ids:
        for i in range(5):
            bar.containers[idx][i]._edgecolor = (0,0,0,1)
    plt.ylim(subdata.vals.min() - 0.1)
    plt.title('acc breakdown (avg)')
    plt.savefig('vis/bar_bd_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        plt.figure(figsize = (15, 10), dpi = 300)
        subdata = data[(data.data == dataset) & (data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1)].melt('curr', ['acc', 'acc_easy', 'acc_med', 'acc_hard', 'bal_acc'], 'cols', 'vals')
        order = subdata[subdata.cols == 'acc'].groupby('curr').mean().sort_values('vals', ascending = False).index
        pal = [clrs[x] for x in order]
        bar = sns.barplot(data=subdata, x = 'cols', y = 'vals', hue = 'curr',
                hue_order = order, linewidth = 2, palette = pal, capsize = 0.01)
        ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        for idx in ent_ids:
            for i in range(5):
                bar.containers[idx][i]._edgecolor = (0,0,0,1)
        plt.ylim(subdata.vals.min() - 0.1)
        plt.title('acc breakdown (%s)'%dataset)
        plt.savefig('vis/bar_bd_%s.png'%dataset, bbox_inches='tight')
        plt.close()

def vis4_2():
    # sns.set(font_scale=2)
    # subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1) & (data.data.isin(datasets))]
    subdata = data[(data.data.isin(datasets))]
    for i, group in enumerate(['acc_easy', 'acc_med', 'acc_hard', 'bal_acc']):
        plt.figure(figsize = (4.2, 3.3), dpi = 300)
        order = subdata.groupby('curr').mean().sort_values(group, ascending = False).index
        pal = [clrs.get(x, 'lightblue') for x in order]
        pal = sns.color_palette("Greys")
        bar = sns.barplot(data=subdata, x = 'curr', y = group,
                hue_order = order, order = order, linewidth = 3, palette = pal, capsize = 0.01, ci = 68)
        ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        for idx in ent_ids:
            bar.containers[0][idx]._edgecolor = (0,0,0,1)
        subdata_avg = subdata.groupby('curr').mean()[group]
        plt.ylim(subdata_avg.min() - 0.01, subdata_avg.max() + 0.01)
        # plt.xticks(rotation = 45)
        # plt.xticks(rotation = 45, fontsize = 13)
        plt.xticks(rotation = 90, fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.locator_params(axis="y", nbins=4)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.ylabel(None)
        plt.xlabel(None)
        plt.savefig('vis/bd_%s.pdf'%group, bbox_inches='tight')
        plt.close()
        # ax[i].set_xticks([])
    # plt.title('acc breakdown (avg)')

    for dataset in datasets:
        subdata = data[data.data == dataset]
        for i, group in enumerate(['acc_easy', 'acc_med', 'acc_hard', 'bal_acc']):
            plt.figure(figsize = (4.2, 3.3), dpi = 300)
            order = subdata.groupby('curr').mean().sort_values(group, ascending = False).index
            pal = [clrs.get(x, 'lightblue') for x in order]
            pal = sns.color_palette("Greys")
            bar = sns.barplot(data=subdata, x = 'curr', y = group,
                    hue_order = order, order = order, linewidth = 3, palette = pal, capsize = 0.01, ci = 68)
            ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
            for idx in ent_ids:
                bar.containers[0][idx]._edgecolor = (0,0,0,1)
            subdata_avg = subdata.groupby('curr').mean()[group]
            plt.ylim(subdata_avg.min() - 0.01, subdata_avg.max() + 0.01)
            # plt.xticks(rotation = 45)
            # plt.xticks(rotation = 45, fontsize = 13)
            plt.xticks(rotation = 90, fontsize = 25)
            plt.yticks(fontsize = 25)
            plt.locator_params(axis="y", nbins=4)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.ylabel(None)
            plt.xlabel(None)
            plt.savefig('vis/bd_%s_%s.pdf'%(dataset, group), bbox_inches='tight')
            plt.close()
            # ax[i].set_xticks([])
        # plt.title('acc breakdown (avg)')

    sns.set()

def vis4_3():
    sns.set(font_scale=1)
    """
    subdata = data[(data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1) & (data.data.isin(datasets))]
    limits = {'acc_easy': (0.83, 0.88),
            'acc_med': (0.62, 0.64),
            'acc_hard': (0.49, 0.52),
            'bal_acc': (0.653, 0.68)
            }
    for i, group in enumerate(['acc_easy', 'acc_med', 'acc_hard', 'bal_acc']):
        plt.figure(dpi = 300)
        order = subdata.groupby('curr').mean().sort_values(group, ascending = False).index
        pal = [clrs[x] for x in order]
        bar = sns.barplot(data=subdata, x = 'curr', y = group,
                hue_order = order, order = order, linewidth = 1, palette = pal, capsize = 0.01)
        ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx]]
        for idx in ent_ids:
            bar.containers[0][idx]._edgecolor = (0,0,0,1)
        plt.ylim(limits[group][0], limits[group][1])
        plt.ylabel('acc')
        plt.savefig('vis/bar_bd_avg_%s.png'%group, bbox_inches='tight')
        plt.close()
        # ax[i].set_xticks([])
    # plt.title('acc breakdown (avg)')
    """

    limits_s = {'acc_easy': (0.75, 0.80),
            'acc_med': (0.55, 0.578),
            'acc_hard': (0.475, 0.52),
            'bal_acc': (0.595, 0.628)
            }
    limits_a = {'acc_easy': (0.2, 0.88),
            'acc_med': (0.2, 0.64),
            'acc_hard': (0.2, 0.52),
            'bal_acc': (0.2, 0.68)
            }
    limits_c = {'acc_easy': (0.2, 0.88),
            'acc_med': (0.2, 0.64),
            'acc_hard': (0.2, 0.52),
            'bal_acc': (0.2, 0.68)
            }
    limits = {'snli_balanced': limits_s,
            'alcohol_7_balanced': limits_a,
            'cancer_balanced': limits_c
            }
    for dataset in datasets:
        fig, ax = plt.subplots(4, 1, figsize = (3.2, 10.5), sharex = True, dpi = 300)
        subdata = data[(data.data == dataset) & (data.epochs == 10) & (data.noise == 0) & (data.data_fraction == 1)]
        for i, group in enumerate(['acc_easy', 'acc_med', 'acc_hard', 'bal_acc']):
            order = subdata.groupby('curr').mean().sort_values(group, ascending = False).index
            pal = [clrs[x] for x in order]
            bar = sns.barplot(data=subdata, x = 'curr', y = group,
                    hue_order = order, order = order, linewidth = 1, palette = pal, capsize = 0.01 , ax = ax[i])
            ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
            for idx in ent_ids:
                bar.containers[0][idx]._edgecolor = (0,0,0,1)
            ax[i].set_ylim(limits[dataset][group][0], limits[dataset][group][1])
            if i == 0:
                ax[i].legend(bar.containers[0], order.tolist())
            # plt.xticks(fontsize = 6)
            plt.xticks([])
            ax[i].set_ylabel(None)
            ax[i].set_xlabel(None)
        plt.savefig('vis/bar_bd_%s.png'%(dataset), bbox_inches='tight')
        plt.close()
    sns.set()

def vis5_1():
    plt.figure(figsize = (15, 10), dpi = 300)
    subdata = data[(data.noise == 0) & (data.epochs == 10) & (data.data.isin(datasets))]
    order = subdata[subdata.data_fraction == 1.0].groupby('curr').mean().sort_values('acc', ascending = False).index[:top_lines]
    if not 'No-CL' in order:
        order = order.insert(top_lines, 'No-CL')
    sizes = [3 if x in ('Ent (sp)', 'Ent (inc)', 'No-CL') else 1 for x in order]
    pal = [line_clrs.get(x, 'lightblue') for x in order]
    subdata = subdata[subdata.curr.isin(order)]
    sns.lineplot(data=subdata, x = 'data_fraction', y = 'acc', size = 'curr', style = 'curr', hue = 'curr', dashes = False,
            err_style = None, palette = pal, hue_order = order, markers = True,
            sizes = sizes)
    plt.title('smaller data (avg)')
    plt.gca().invert_xaxis()
    # plt.savefig('vis/line_small_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        if not 'balanced' in dataset:
            continue
        # fig = plt.figure(figsize = (15, 10), dpi = 300)
        fig = plt.figure(figsize = (4.2, 3.3), dpi = 300)
        ax = plt.gca()
        subdata = data[(data.data == dataset) & (data.noise == 0) & (data.epochs == 10)]
        order = subdata[subdata.data_fraction == 1.0].groupby('curr').mean().sort_values('acc', ascending = False).index[:top_lines]
        if not 'No-CL' in order:
            order = order.insert(top_lines, 'No-CL')
        sizes = [2 if x in ('Ent (sp)', 'Ent (inc)', 'No-CL') else 1 for x in order]
        pal = [line_clrs.get(x, 'lightblue') for x in order]
        subdata = subdata[subdata.curr.isin(order)]
        sns.lineplot(data=subdata, x = 'data_fraction', y = 'acc', style = 'curr', hue = 'curr', dashes = False, err_style = None, palette = pal, hue_order = order, markers = True, sizes = sizes, size = 'curr')
        plt.gca().invert_xaxis()
        plt.legend(prop={'size': 13}, framealpha = 0.8,
            handlelength = 0.8,)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xlabel('Data Fraction', fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.locator_params(axis="y", nbins=4)
        # plt.title('smaller data (%s)'%dataset)

        # for line, name in zip(ax.lines, order.tolist()):
        #     y = line.get_ydata()[-1]
        #     x = line.get_xdata()[-1]
        #     if not np.isfinite(y):
        #         y=next(reversed(line.get_ydata()[~line.get_ydata().mask]),float("nan"))
        #     if not np.isfinite(y) or not np.isfinite(x):
        #         continue     
        #     text = ax.annotate(name,
        #                    xy=(x, y),
        #                    xytext=(0, 0),
        #                    color=line.get_color(),
        #                    xycoords=(ax.get_xaxis_transform(),
        #                              ax.get_yaxis_transform()),
        #                    textcoords="offset points")
        #     text_width = (text.get_window_extent(
        #     fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
        #     if np.isfinite(text_width):
        #             ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 1.05)

        plt.savefig('vis/line_small_%s.pdf'%dataset, bbox_inches='tight')
        plt.close()

def vis5_2():
    plt.figure(figsize = (20, 10), dpi = 300)
    subdata = data[(data.noise == 0) & (data.epochs == 10) & (data.data.isin(datasets))]
    order = subdata[subdata.data_fraction == 1.0].groupby('curr').mean().sort_values('acc', ascending = False).index
    pal = [clrs[x] for x in order]
    bar = sns.barplot(data=subdata, x = 'data_fraction', y = 'acc', hue = 'curr', hue_order = order, linewidth = 2, palette = pal, order = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1], capsize = 0.01)
    ent_ids = [idx for idx in range(len(subdata.curr.unique())) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
    for idx in ent_ids:
        for i in range(6):
            bar.containers[idx][i]._edgecolor = (0,0,0,1)
    plt.title('smaller data (avg)')
    plt.ylim(subdata.acc.min() - 0.1)
    plt.savefig('vis/bar_small_avg.png', bbox_inches='tight')
    plt.close()

    for dataset in datasets:
        if not 'balanced' in dataset:
            continue
        plt.figure(figsize = (16, 6.3), dpi = 300)
        subdata = data[(data.data == dataset) & (data.noise == 0) & (data.epochs == 10)]
        order = subdata[subdata.data_fraction == 1.0].groupby('curr').mean().sort_values('acc', ascending = False).index
        pal = [clrs[x] for x in order]
        bar = sns.barplot(data=subdata, x = 'data_fraction', y = 'acc', hue = 'curr', hue_order = order, linewidth = 4, palette = pal, order = [1.0, 0.8, 0.6, 0.4, 0.2], capsize = 0.01)
        ent_ids = [idx for idx in range(len(order)) if 'Ent' in order[idx] or 'No' in order[idx] or 'Loss' in order[idx]]
        # for idx in ent_ids:
        #     for i in range(5):
        #         bar.containers[idx][i]._edgecolor = (0,0,0,1)
        for idx in range(len(bar.containers)):
            for i in range(5):
                if idx in ent_ids:
                    bar.containers[idx][i]._edgecolor = (0,0,0,1)
                else:
                    bar.containers[idx][i]._edgecolor = (0,0,0,0)
        # plt.title('smaller data (%s)'%dataset)
        # plt.legend(prop={'size': 20})

        plt.legend(prop={'size': 25}, bbox_to_anchor = (0.3,0.5), framealpha = 0.8)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.xlabel('Data Fraction', fontsize = 25)
        plt.ylabel('Accuracy', fontsize = 25)

        plt.ylim(subdata.acc.min() - 0.1)
        plt.savefig('vis/bar_small_%s.png'%dataset, bbox_inches='tight')
        plt.close()
def print_for_latex_1():
    subdata = data[(data.noise == 0) & (data.epochs == 10) & (data.data_fraction == 1)]
    means = subdata.groupby(['data', 'curr'])['acc'].mean().apply(lambda x: round(x*100, 3))
    stds = subdata.groupby(['data', 'curr'])['acc'].sem().apply(lambda x: round(x*100, 3))
    import scipy
    print(" ".join(data.curr.unique()))
    for dataset in data.data.unique():
        s = '%s & '%dataset
        for curr in ['No-CL', 'Ent (sp)', 'Ent (inc)', 'DP', 'MentorNet', 'SL', 'SPL']:
            m, e = means[dataset, curr], stds[dataset, curr]
            s += "%.2f $\pm$ %.2f & "%(m, e)
        print(s[:-2])
        print()
    means = subdata.groupby(['curr'])['acc'].mean().apply(lambda x: round(x*100, 3))
    stds = subdata.groupby(['curr'])['acc'].sem().apply(lambda x: round(x*100, 3))
    s = 'Average & '
    for curr in ['Ent (sp)', 'Ent (inc)', 'No-CL', 'DP', 'MentorNet', 'SL', 'SPL']:
        m, e = means[curr], stds[curr]
        s += "%.2f $\pm$ %.2f & "%(m, e)
    print(s[:-2])
    print()

def print_for_latex_2():
    subdata = data[(data.noise == 0) & (data.epochs == 10) & (data.data_fraction == 1)]
    means = subdata.groupby(['data', 'curr'])['acc'].mean().apply(lambda x: round(x*100, 3))
    stds = subdata.groupby(['data', 'curr'])['acc'].sem().apply(lambda x: round(x*100, 3))
    import scipy
    datasets_order = data.data.unique()
    datasets_order = ['snli_balanced', 'snli_special', 'alcohol_7_balanced', 'alcohol_7', 'cancer_balanced', 'cancer']
    print(" ".join(datasets_order))
    for curr in data.curr.unique():
        s = '\\textbf{%s} & '%curr
        for dataset in datasets_order:
            m, e = means[dataset, curr], stds[dataset, curr]
            if m == means[dataset].max():
                s += "\\textbf{%.2f} $\pm$ %.2f & "%(m, e)
            else:
                s += "%.2f $\pm$ %.2f & "%(m, e)
        s = s[:-3]
        s += '\\\\'
        print(s)
        print()
    means = subdata.groupby(['data'])['acc'].mean().apply(lambda x: round(x*100, 3))
    stds = subdata.groupby(['data'])['acc'].sem().apply(lambda x: round(x*100, 3))
    s = '\\textbf{Average} & '
    for dataset in datasets_order:
        m, e = means[dataset], stds[dataset]
        if m == means.max():
            s += "\\textbf{%.2f} $\pm$ %.2f & "%(m, e)
        else:
            s += "%.2f $\pm$ %.2f & "%(m, e)
    print(s[:-2])
    print()

def vis7():
    query = ['Loss (sp)', 'Ent (sp)', 'Loss (inc)', 'Ent (inc)']
    query = ['No-CL']
    query = ['Ent (inc)']
    f = open('vis/pvals.csv', 'w')
    f.write("data, curr1, curr2, acc1, acc2, pval, stars\n")
    for dataset in datasets:
        subdata = data[data.data == dataset]
        subdata_grouped = subdata.groupby('curr').mean()
        for q in query:
            for curr in subdata_grouped.sort_values('acc', ascending = False).index:
                if curr in query:
                    continue
                s1 = subdata[subdata.curr == q].acc
                s2 = subdata[subdata.curr == curr].acc
                if len(s1) > len(s2):
                    s1 = s1[:len(s2)]
                elif len(s2) > len(s1):
                    s2 = s2[:len(s1)]
                # pval = ttest_ind(s1, s2,
                #         equal_var=False, alternative = 'greater')[1]
                pval = ttest_rel(s1, s2,
                        alternative = 'greater')[1]
                # pval = wilcoxon(s1, s2,
                #         alternative = 'greater')[1]
                acc1 = subdata_grouped.loc[q].acc
                acc2 = subdata_grouped.loc[curr].acc
                if pval < 0.0005:
                    stars = "***"
                elif pval < 0.005:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                else:
                    stars = ""
                s = f"{dataset}, {q}, {curr}, {acc1*100:.3f}, {acc2*100:.3f}, {pval:.4f}, {stars}\n"
                f.write(s)
    f.close()

def vis8():
    fig = plt.figure(figsize = (4.2, 3.3), dpi = 300)
    subdata = data
    subdata.ent_classes = data.ent_classes.astype('int')
    sns.lineplot(data = subdata, x = 'ent_classes', y = 'acc', hue = 'curr',
                err_style = 'bars', size = 'curr', ci = 68)
    plt.xlabel("Number of classes", fontsize = 15)
    plt.ylabel("Accuracy", fontsize = 15)
    plt.legend(prop={'size': 13}, framealpha = 0.8,
        handlelength = 0.8,)
    xticks = [2,4,6,8,16]
    # plt.xticks(xticks, fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.locator_params(axis="y", nbins=6)
    # plt.savefig('vis/ent_fg_all.pdf', bbox_inches='tight')
    plt.savefig('vis/ent_fg_all.png', bbox_inches='tight')

    for dataset in data.data.unique():
        fig = plt.figure(figsize = (4.2, 3.3), dpi = 300)
        subdata = data[data.data == dataset].sort_values('ent_classes')
        subdata.ent_classes = data.ent_classes.astype('int')
        sns.lineplot(data = subdata, x = 'ent_classes', y = 'acc', hue = 'curr',
                err_style = 'bars', size = 'curr', ci = 68
                )
        plt.xlabel("Number of classes", fontsize = 15)
        plt.ylabel("Accuracy", fontsize = 15)
        plt.legend(prop={'size': 13}, framealpha = 0.8,
            handlelength = 0.8,)
        # plt.xticks(xticks, fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.locator_params(axis="y", nbins=6)
        # plt.savefig('vis/ent_fg_%s.pdf'%dataset, bbox_inches='tight')
        plt.savefig('vis/ent_fg_%s.png'%dataset, bbox_inches='tight')
                


top_lines = 4
if args.data == 'bal':
    datasets = ['snli_balanced', 'alcohol_7_balanced', 'cancer_balanced']
    default_data = 'snli_balanced'
elif args.data == 'full':
    datasets = ['snli_special', 'alcohol_7', 'cancer']
    default_data = 'snli_special'
elif args.data == 'all':
    datasets = ['snli_balanced', 'alcohol_7_balanced', 'cancer_balanced', 'snli_special', 'alcohol_7', 'cancer']
    default_data = 'snli_balanced'
else:
    datasets = [args.data]
    default_data = 'snli_balanced'
datasets += ['chaosnli']

data = load_data()
data = data[data.data.isin(datasets)]

if '1' in args.vis:
    # vis1_1()
    # vis1_2()
    # vis1_3()
    vis1_4()
if '2' in args.vis:
    vis2_1()
    # vis2_2()
if '3' in args.vis:
    vis3_1()
    # vis3_2()
if '4' in args.vis:
    # vis4_1()
    vis4_2()
    # vis4_3()
if '5' in args.vis:
    vis5_1()
    # vis5_2()
if '6' in args.vis:
    print_for_latex_2()
if '7' in args.vis:
    vis7()
if '8' in args.vis:
    vis8()
