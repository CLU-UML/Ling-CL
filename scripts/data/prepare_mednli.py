from datasets import load_dataset


data = load_dataset('json', data_files={
    name: f'/data/mohamed/data/mednli_raw/mli_{name}_v1.jsonl'
    for name in ['train', 'dev', 'test'] })

# data = data.rename_column('gold_label', 'label')

label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
data = data.map(lambda x: {'label': label_map[x['gold_label']]})

data.save_to_disk('/data/mohamed/data/mednli')
