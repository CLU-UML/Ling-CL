import lftk
import spacy
import sys, os
from datasets import load_from_disk

dataset = sys.argv[1]
data = load_from_disk(dataset)

nlp = spacy.load("en_core_web_sm")

def extract_lftk(row, input_col, output_col):
    text = row[input_col]
    doc = nlp(text)
    LFTK = lftk.Extractor(doc)
    
    feats = LFTK.extract()
    
    return {output_col: list(feats.values())}

if 'sentence1' in data['train'].column_names:
    data = data.map(extract_lftk, num_proc=40,
            fn_kwargs={'input_col': 'sentence1', 'output_col': 'sentence1_lftk'})
    data = data.map(extract_lftk, num_proc=40,
            fn_kwargs={'input_col': 'sentence2', 'output_col': 'sentence2_lftk'})
elif 'sentence' in data['train'].column_names:
    data = data.map(extract_lftk, num_proc=40,
            fn_kwargs={'input_col': 'sentence', 'output_col': 'sentence_lftk'})

print(data)
data.save_to_disk(dataset)
