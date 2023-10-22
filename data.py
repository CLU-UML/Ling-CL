import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from sklearn.preprocessing import StandardScaler, PowerTransformer
from datasets import load_from_disk

def noise_reduce_permute(data, args):
    n = len(data['train'])
    if args.data_fraction < 1:
        ids = np.random.choice(n, int(args.data_fraction*n), replace=False)
        data['train'] = data['train'].select(ids)

    if args.noise > 0:
        noisy_ids = np.random.choice(n, int(args.noise*n), replace=False)
        noisy_labels = {idx: l for idx,l in zip(noisy_ids,
            np.random.permutation(data['train'][noisy_ids]['label']))}
        def process(sample, idx):
            if idx in noisy_ids:
                sample['label'] = noisy_labels[idx]
            return sample
        data['train'] = data['train'].map(process, with_indices = True)

    if args.diff_permute:
        diff = np.random.permutation(data['train']['difficulty_class'])
        def process(sample, idx):
            sample['difficulty_class'] = diff[idx]
            return sample
        data['train'] = data['train'].map(process, with_indices = True)
    return data

def filter_columns(data, args):
    cur_cols = data['train'].column_names
    columns = ['label']
    if args.curr == 'dp' and 'diff' in cur_cols:
        columns.append('diff')

    if 'difficulty_class' in cur_cols:
        columns.append('difficulty_class')
    if 'difficulty_score' in cur_cols:
        columns.append('difficulty_score')

    if 'sentence1' in cur_cols:
        text = ['sentence1', 'sentence2']
    elif 't' in cur_cols:
        text = ['t']
    elif 'sentence' in cur_cols and 'addon' in cur_cols:
        text = ['sentence', 'addon']
    elif 'sentence' in cur_cols:
        text = ['sentence']
    columns += text

    if args.diff_score == 'lng_w' or args.add_feats == 'lng' or args.multiview:
        columns.append('lng')

    if args.eval_class is not None:
        columns.append(args.eval_class)

    # if 'lng' in args.data:
    #     for t in text:
    #         columns.extend(['%s_lca'%t, '%s_sca'%t])

    for split in data:
        remove_cols = [k for k in data[split].features if not k in columns]
        data[split] = data[split].remove_columns(remove_cols)
    return data, text

def partition(data, diff, args):
    thresholds = [np.percentile(diff, i/args.diff_classes*100) for i in range(args.diff_classes)]
    def assign_class(row):
        diff_class = 0
        val = row['difficulty_score']
        if isinstance(val, list):
            val = val[args.lng_id]
        for i in range(args.diff_classes - 1, -1, -1):
            if val >= thresholds[i]:
                diff_class = i
                break
        return {'difficulty_class': diff_class}
    data = data.map(assign_class)
    return data

def process_diff(data, args):
    if args.diff_score is not None:
        data = data.rename_column(args.diff_score, 'difficulty_score')

        scaler = StandardScaler()
        diff = np.array(data['train']['difficulty_score']).reshape(-1,1)
        scaler.fit(diff)
        def scale(row):
            diff = np.array(row['difficulty_score']).reshape(-1,1)
            return {'difficulty_score': scaler.transform(diff).ravel()}
        data = data.map(scale, batched=True)

        diff = data['train']['difficulty_score']
        data = partition(data, diff, args)
    elif args.diff_class is not None:
        data = data.rename_column(args.diff_class, 'difficulty_class')

    return data

def combine_lng(data, lng_ids, lng_idx = None):
    def process(row):
        if 'sentence1' in row:
            all_lng = row['sentence1_lca'] \
                    + row['sentence1_sca'] \
                    + row['sentence1_lingfeat'] \
                    + row['sentence2_lca'] \
                    + row['sentence2_sca'] \
                    + row['sentence2_lingfeat']
        elif 'sentence' in row:
            all_lng = row['lca'] + row['sca'] + row['sentence_lingfeat']
        if lng_ids is not None:
            all_lng = [x for idx,x in enumerate(all_lng) if idx in lng_ids]
        elif lng_idx is not None:
            all_lng = all_lng[lng_idx]
        return {'lng': all_lng}

    data = data.map(process)
    if not lng_idx:
        scaler = StandardScaler()
        scaler.fit(data['train']['lng'])
        def scale(row):
            return {'lng': scaler.transform(row['lng'])}
        data = data.map(scale, batched=True)
    return data

def weight_lng(data, ps, n, multiview=False):
    if ps is None:
        ps = np.random.rand(n)*2 - 1
    idx = abs(ps).argmax()
    order = 1 if ps[idx] >= 0 else -1
    def weighted_lng(example):
        if multiview:
            lng_w = np.array(example['lng'])[:,idx] * order
        else:
            lng_w = (example['lng'] * ps).sum(1) / np.sqrt(np.square(ps).sum())
        return {'lng_w': lng_w}
    data = data.map(weighted_lng, batched=True)
    return data

def update_dataloader(dataloader, ps, args):
    data = dataloader.dataset
    data = weight_lng(data, ps, len(data['lng'][0]), args.multiview)
    if args.diff_score is not None:
        data = data.remove_columns('difficulty_score')
        data = data.rename_column(args.diff_score, 'difficulty_score')
    diff = data['difficulty_score']
    data = partition(data, diff, args)

    dataloader = DataLoader(data, dataloader.batch_size,
            sampler=dataloader.sampler, collate_fn = dataloader.collate_fn)
    return dataloader


def process_labels(sample, n):
    labels = sample['labels']
    np.random.shuffle(labels)
    labels = labels[:n]
    label = np.argmax(np.bincount(labels))
    return {'label': label}


def create_token_labels(labels, encoded):
    new_labels = []
    for i in range(len(labels)):
        offsets = encoded['offset_mapping'][i]
        orig_len = sum(encoded['attention_mask'][i])
        new_label = np.zeros_like(encoded['input_ids'][i], dtype=float)
        for j, (cs, ce) in enumerate(offsets):
            if (cs, ce) == (0,0):
                continue
            if any([(cs >= s and cs < e) or (ce >= s and ce < e) for s, e in labels[i]]):
                new_label[j] = 1
        new_labels.append(new_label)
    return new_labels

def get_dataloaders(args, tokenizer, ps=None, lng_names=None):
    def tokenize(x):
        if args.data == 'fce-error-detection':
            encoded = tokenizer.batch_encode_plus(x['sentence'], truncation=True, max_length=args.max_length,
                    padding = 'max_length', return_offsets_mapping=True)
            labels = create_token_labels(x['label'], encoded)
            return {'input_ids': encoded.input_ids, 'attention_mask': encoded.attention_mask, 'label': labels}
        if 'sentence1' in x:
            return tokenizer(x['sentence1'], x['sentence2'],
                    truncation=True, max_length = args.max_length, padding='max_length'
                    )
        elif 't' in x:
            return tokenizer(x['t'], 
                    truncation=True, max_length = args.max_length, padding='max_length')
        elif 'sentence' in x and 'addon' in x:
            return tokenizer(x['sentence'], x['addon'],
                    truncation=True, max_length = args.max_length, padding='max_length')
        elif 'sentence' in x:
            return tokenizer(x['sentence'], 
                    truncation=True, max_length = args.max_length, padding='max_length')

    def collate_fn(batch):
        return {k: torch.tensor([x[k] for x in batch]) for k in batch[0].keys()}

    data = load_from_disk(os.path.join(args.data_dir, args.data))

    if args.n_annots > 0:
        labels = data['train']['label']
        data['train'] = data['train'].map(process_labels, fn_kwargs={'n': args.n_annots})
        labels_after = data['train']['label']

    if args.diff_score == 'lng_w' or args.diff_score == 'lng' or args.add_feats == 'lng':
        if args.lng_ids:
            lng_ids = np.load(os.path.join(args.data_dir, 'lng_ids', args.data) + '.npy')
        elif args.lng_ids_path:
            lng_ids = np.load(args.lng_ids_path)
            if len(lng_names) > 241:
                lng_ids = [x for x in lng_ids] + [x + 241 for x in lng_ids]
        else:
            lng_ids = None

        lng_idx = lng_names.index(args.lng_name) if args.lng_name else None
        data = combine_lng(data, lng_ids, lng_idx)
        # if args.data in ('snli', 'anli', 'rte'):
        #     args.lng_n = 482
        # else:
        #     args.lng_n = 241
        args.lng_n = len(data['train']['lng'][0]) if not lng_idx else 1
    if args.diff_score == 'lng_w':
        data = weight_lng(data, ps, len(data['train']['lng'][0]))
    data = process_diff(data, args)

    data = noise_reduce_permute(data, args)

    data, text = filter_columns(data, args)

    data = data.map(tokenize, batched = True, remove_columns = text)

    if args.overfit:
        data['train'] = data['train'].select(range(1))

    sampler = SubsetRandomSampler([]) if args.curr in ['sampling', 'competence', 'datasel']\
            else RandomSampler(data['train'])

    train_dataloader = DataLoader(data['train'], args.batch_size,
            sampler=sampler, collate_fn = collate_fn, num_workers=0)
    dev_dataloader = DataLoader(data['dev'], args.batch_size, collate_fn = collate_fn)
    test_dataloader = DataLoader(data['test'], args.batch_size, collate_fn = collate_fn)
    train_dataloader_ns = DataLoader(data['train'], args.batch_size, collate_fn=collate_fn)\
            if args.save_losses or args.curr == 'datasel' else None

    if args.curr == 'dp':
        args.dp_tao = float(np.percentile(data['train']['diff'], 50))

    return train_dataloader, dev_dataloader, test_dataloader,\
            data['dev'], train_dataloader_ns
