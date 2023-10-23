import torch
def ignore_padding_flatten(preds, attention_mask):
    attention_mask[:,0] = 0
    attention_mask = attention_mask.bool()
    return torch.cat([x[mask] for (x, mask) in zip(preds, attention_mask)])
