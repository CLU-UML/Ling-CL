import datasets
import pandas as pd
import numpy as np
from collections import Counter

if __name__ == '__main__':
    # data = load_dataset('json', data_files={
    #     'train': [
    #         'data/snli_1.0/snli_1.0_train.jsonl',
    #         'data/multinli_1.0/multinli_1.0_train.jsonl'
    #         ]
    #     })

    train_files = [
            'data/snli_1.0/snli_1.0_train.jsonl',
            'data/multinli_1.0/multinli_1.0_train.jsonl',
            'data/nli_fever/train_fitems.jsonl',
            'data/anli_v1.0/R1/train.jsonl',
            'data/anli_v1.0/R2/train.jsonl',
            'data/anli_v1.0/R3/train.jsonl'
            ]

    dev_files = [
            'data/anli_v1.0/R1/dev.jsonl',
            'data/anli_v1.0/R2/dev.jsonl',
            'data/anli_v1.0/R3/dev.jsonl'
            ]

    test_files = [
            'data/anli_v1.0/R1/test.jsonl',
            'data/anli_v1.0/R2/test.jsonl',
            'data/anli_v1.0/R3/test.jsonl'
            ]

    anli_labels = {'c': 'contradiction', 'n': 'neutral', 'e': 'entailment'}
    fever_labels = {'SUPPORTS': 'entailment', 'REFUTES': 'contradiction',
            'NOT ENOUGH INFO': 'neutral'}
    labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    train_dfs = []
    for fn in train_files:
        with open(fn) as f:
            df = pd.io.json.read_json(f, lines = True)
            df.rename({'context': 'sentence1',
            'hypothesis': 'sentence2',
            'query': 'sentence2'
            }, inplace = True, axis = 1)

            if 'fever' in fn:
                df['label'] = df['label'].map(lambda x: fever_labels[x])
            elif 'anli' in fn:
                df['label'] = df['label'].map(lambda x: anli_labels[x])
            else:
                df['label'] = df['annotator_labels'].map(
                        lambda x: Counter(x).most_common(1)[0][0])
            df['label'] = df['label'].map(lambda x: labels[x])

            df['entropy_class'] = np.ones(len(df), dtype = 'int') * 0

            if 'multi' in fn:
                df['mnli_genre'] = df['genre']
            elif 'anli' in fn:
                df['anli_genre'] = df['genre']

            if 'anli' in fn:
                if 'R1' in fn:
                    df['anli_class'] = np.ones(len(df), dtype = 'int') * 0
                elif 'R2' in fn:
                    df['anli_class'] = np.ones(len(df), dtype = 'int') * 1
                elif 'R3' in fn:
                    df['anli_class'] = np.ones(len(df), dtype = 'int') * 2

            if 'anli' in fn:
                if 'R1' in fn:
                    train_dfs.extend([df] * 10)
                elif 'R2' in fn:
                    train_dfs.extend([df] * 20)
                elif 'R3' in fn:
                    train_dfs.extend([df] * 10)
            else:
                train_dfs.append(df)
    train_df = pd.concat(train_dfs)
    train_df['anli_class'].fillna(-1, axis = 0, inplace = True)
    train_df = train_df[['entropy_class', 'sentence1', 'sentence2',
        'mnli_genre', 'anli_genre', 'anli_class', 'label']]

    dev_dfs = []
    for fn in dev_files:
        with open(fn) as f:
            df = pd.io.json.read_json(f, lines = True)
            df.rename({'context': 'sentence1',
            'hypothesis': 'sentence2',
            'query': 'sentence2'
            }, inplace = True, axis = 1)

            df['label'] = df['label'].map(lambda x: anli_labels[x])
            df['label'] = df['label'].map(lambda x: labels[x])

            df['anli_genre'] = df['genre']

            if 'R1' in fn:
                df['anli_class'] = np.ones(len(df), dtype = 'int') * 0
            elif 'R2' in fn:
                df['anli_class'] = np.ones(len(df), dtype = 'int') * 1
            elif 'R3' in fn:
                df['anli_class'] = np.ones(len(df), dtype = 'int') * 2

            df['entropy_class'] = np.ones(len(df), dtype = 'int') * 0


            dev_dfs.append(df)
    dev_df = pd.concat(dev_dfs)
    dev_df = dev_df[['entropy_class', 'sentence1', 'sentence2', 'anli_genre', 'anli_class', 'label']]

    test_dfs = []
    for fn in test_files:
        with open(fn) as f:
            df = pd.io.json.read_json(f, lines = True)
            df.rename({'context': 'sentence1',
            'hypothesis': 'sentence2',
            'query': 'sentence2'
            }, inplace = True, axis = 1)

            df['label'] = df['label'].map(lambda x: anli_labels[x])
            df['label'] = df['label'].map(lambda x: labels[x])

            df['anli_genre'] = df['genre']

            if 'R1' in fn:
                df['anli_class'] = np.ones(len(df), dtype = 'int') * 0
            elif 'R2' in fn:
                df['anli_class'] = np.ones(len(df), dtype = 'int') * 1
            elif 'R3' in fn:
                df['anli_class'] = np.ones(len(df), dtype = 'int') * 2

            df['entropy_class'] = np.ones(len(df), dtype = 'int') * 0


            test_dfs.append(df)
    test_df = pd.concat(test_dfs)
    test_df = test_df[['entropy_class', 'sentence1', 'sentence2', 'anli_genre', 'anli_class', 'label']]

    data = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(train_df),
        'dev': datasets.Dataset.from_pandas(dev_df),
        'test': datasets.Dataset.from_pandas(test_df),
        })

    data.save_to_disk('data/anli_upscaled')
