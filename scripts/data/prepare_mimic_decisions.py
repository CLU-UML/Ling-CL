import os
import json
from glob import glob
from collections import Counter

data_dir = '/data/mohamed/data'
files = glob(os.path.join(data_dir, 'mimic_decisions/data/**/*'))

for fn in files:
    ids = []
    with open(fn) as f:
        data = json.load(f, strict=False)
        annots = data[0]['annotations']
    txt_candidates = glob(os.path.join(data_dir, f'mimic_decisions/raw_text/{os.path.basename(fn).split("-")[0]}*.txt'))
    txt_fn = txt_candidates[0]
    s = open(txt_fn).read()

    for annot in annots:
        start = int(annot['start_offset'])
        end = int(annot['end_offset'])
    break
