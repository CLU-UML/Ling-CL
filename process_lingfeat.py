import sys, os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
from lingfeat import extractor
from datasets import load_from_disk
import logging
# import torch
# torch.multiprocessing.set_start_method('spawn')

dataset = sys.argv[1]
data = load_from_disk(dataset)

# LF = extractor.pass_text(data['train']['sentence1'][0])
# LF.preprocess()
# LF.PhrF_()
# logging.getLogger('stanza').setLevel(30)

def extract_lingfeat(row, input_col, output_col):
    if output_col in row and len(row[output_col]) != 1:
        return {output_col: row[output_col]}
    text = row[input_col]
    LingFeat = extractor.pass_text(text)
    LingFeat.preprocess()
    
    d = {}
    d.update(LingFeat.WoKF_()) # Wikipedia Knowledge Features
    d.update(LingFeat.WBKF_()) # WeeBit Corpus Knowledge Features
    d.update(LingFeat.OSKF_()) # OneStopEng Corpus Knowledge Features

    # Discourse (Disco) Features
    d.update(LingFeat.EnDF_()) # Entity Density Features
    d.update(LingFeat.EnGF_()) # Entity Grid Features

    # Syntactic (Synta) Features
    # d.update(LingFeat.PhrF_()) # Noun/Verb/Adj/Adv/... Phrasal Features (logging stanza)
    # d.update(LingFeat.TrSF_()) # (Parse) Tree Structural Features (logging stanza)
    d.update(LingFeat.POSF_()) # Noun/Verb/Adj/Adv/... Part-of-Speech Features

    # Lexico Semantic (LxSem) Features
    d.update(LingFeat.TTRF_()) # Type Token Ratio Features
    d.update(LingFeat.VarF_()) # Noun/Verb/Adj/Adv Variation Features 
    d.update(LingFeat.PsyF_()) # Psycholinguistic Difficulty of Words (AoA Kuperman)
    d.update(LingFeat.WorF_()) # Word Familiarity from Frequency Count (SubtlexUS)

    # Shallow Traditional (ShTra) Features
    d.update(LingFeat.ShaF_()) # Shallow Features (e.g. avg number of tokens)
    d.update(LingFeat.TraF_()) # Traditional Formulas 
    
    return {output_col: list(d.values())}

if 'sentence1' in data['train'].column_names:
    data = data.map(extract_lingfeat, num_proc=30,
            fn_kwargs={'input_col': 'sentence1', 'output_col': 'sentence1_lingfeat'})
    data = data.map(extract_lingfeat, num_proc=30,
            fn_kwargs={'input_col': 'sentence2', 'output_col': 'sentence2_lingfeat'})
elif 'sentence' in data['train'].column_names:
    data = data.map(extract_lingfeat, num_proc=100,
            fn_kwargs={'input_col': 'sentence', 'output_col': 'sentence_lingfeat'})
print(data)

data.save_to_disk(dataset)
