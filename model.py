import os
import torch
from datetime import datetime
from torch import nn
from torch.optim import SGD
from transformers import AutoModel, AutoConfig
from curricula.superloss import SuperLoss
from curricula.spl import SPL
from curricula.lng_curr import LngCurriculum
from curricula.mentornet import MentorNet
from curricula.glf_curr import GLFCurriculum
from curricula.sig_curr import SigmoidCurriculum
from curricula.gauss_curr import GaussCurriculum
from curricula.sampling_curr import SamplingCurriculum
from curricula.competence_curr import CompetenceCurriculum
from curricula.diff_pred_weighting import DPWeighting
from curricula.data_sel_curr import DataSelCurriculum
from transformers import AutoModel, AutoTokenizer, AdamW, AutoConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_name, num_labels, ckpt = args.model_name, args.num_labels, args.ckpt

        if ckpt is not None:
            self.backbone = AutoModel.from_pretrained(ckpt.replace('meta.pt', 'model'))
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        if 't5' in model_name:
            self.backbone = self.backbone.encoder

        config = AutoConfig.from_pretrained(model_name)
        self.config = config
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        hidden_size = config.hidden_size
        if args.add_feats == 'lng':
            hidden_size += args.lng_n

        if args.data in ('fce-error-detection'):
            self.token_classification = True
        else:
            self.token_classification = False

        if self.token_classification:
            out_logits = 1
        else:
            out_logits = num_labels
        self.classifier = nn.Linear(hidden_size, out_logits)
        self.classifier.apply(self.init_weights)

        if args.diff_score == 'lng_w' and args.lng_method == 'train':
            self.p = nn.Parameter(torch.rand(args.lng_n)*2 -1)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()

    def forward(self, x, add_feats = None):
        model_out = self.backbone(**x)
        if not self.token_classification:
            h = model_out.pooler_output
        else:
            h = model_out.last_hidden_state

        if add_feats is not None:
            if h.ndim == 3:
                add_feats = add_feats.unsqueeze(1).expand(-1,h.shape[1], -1)
            h = torch.cat([h, add_feats], -1)
        h = self.dropout(h)

        logits = self.classifier(h)
        if self.token_classification:
            logits = logits.squeeze(-1)

        return logits

def get_name(args):
    if args.curr == 'none':
        curr = 'none'
    elif args.curr == 'sl':
        curr = 'sl_%s'%args.sl_mode 
    elif args.curr == 'ent' or args.curr == 'ent+':
        curr = 'ent_%s'%args.glf_cfg
    elif args.curr == 'loss':
        curr = 'loss_%s'%args.glf_cfg
    elif args.curr == 'spl':
        curr = 'spl_%s'%args.spl_mode
    elif args.curr == 'mentornet':
        curr = 'mentornet'
    elif args.curr == 'dp':
        curr = 'dp'
    elif args.curr == 'datasel':
        curr = 'datasel'
    elif args.curr == 'lng':
        curr = 'lng'
    elif args.curr == 'competence':
        curr = 'comp'
    elif args.curr == 'sampling':
        curr = 'samp'
    elif args.curr == 'gauss':
        curr = 'gauss'
    elif args.curr == 'sigmoid':
        curr = 'sig'
    elif args.curr == 'neg-sigmoid':
        curr = 'negsig'
    else:
        curr = ''
    if args.diff_score == 'lng_w':
        lng_ids = args.lng_ids or args.lng_ids_path is not None
        curr += '_%s_%s_%s'%(args.lng_method, args.multiview, lng_ids)
    if args.add_feats == 'lng':
        curr = 'concat_%s'%args.lng_ids
    diff_score = args.lng_name if args.lng_name else args.diff_score
    curtime = datetime.now().strftime('%y%m%d_%H-%M-%S')
    name = f'{curtime}_{args.data}_{diff_score}_{curr}_{args.seed}'
    return name

def init_model(args, device, dataloader, glf_cfg=None):
    if args.curr == 'sl':
        curr = SuperLoss(args.num_labels, mode=args.sl_mode, lam=args.sl_lam).to(device)
    elif args.curr == 'glf' or args.curr == 'glf+':
        curr = GLFCurriculum(args.glf_cfg, args.epochs, avgloss='+' in args.curr,
                cfg=glf_cfg, diff_classes = args.diff_classes, anti=args.anti_curr)
    elif args.curr == 'sigmoid':
        curr = SigmoidCurriculum('pos', alpha=args.sig_alpha)
    elif args.curr == 'neg-sigmoid':
        curr = SigmoidCurriculum('neg', alpha=args.sig_alpha)
    elif args.curr == 'gauss':
        curr = GaussCurriculum(alpha=args.gauss_alpha)
    elif args.curr == 'spl':
        curr = SPL(mode = args.spl_mode)
    elif args.curr == 'mentornet':
        curr = MentorNet(args.num_labels, args.epochs)
    elif args.curr == 'dp':
        curr = DPWeighting(args.dp_tao, args.dp_alpha)
    elif args.curr == 'datasel':
        curr = DataSelCurriculum(dataloader.sampler, len(dataloader.dataset))
    elif args.curr in ['sampling', 'competence']:
        if args.multiview:
            diff_col = 'lng'
        elif 'difficulty_score' in dataloader.dataset.column_names:
            diff_col = 'difficulty_score'
        else:
            diff_col = 'difficulty_class'
        if args.curr == 'competence':
            curr = CompetenceCurriculum(dataloader.dataset[diff_col], dataloader.sampler,
                    p=args.comp_p, c0=args.comp_c0)
        elif args.curr == 'sampling':
            curr = SamplingCurriculum(dataloader.dataset[diff_col], dataloader.sampler,
                    multiview=args.multiview)
    elif args.curr == 'none':
        curr = None
    else:
        raise NameError('Invalid curriculum name')

    if args.ckpt:
        print('[Resuming]')
        state = torch.load(args.ckpt)
        step = state['step']
        name = os.path.basename(args.ckpt)
        str_end = name.rfind('_', 0, -8)
        name = name[:str_end]
        model = Model(args).to(device)
        print(next(model.backbone.parameters()))
        model.load_state_dict(state['model'], strict=False)
        print(next(model.backbone.parameters()))
    else:
        step = 0
        name = get_name(args)
        model = Model(args).to(device)

    if isinstance(curr, nn.Module):
        curr.to(device)
    return model, curr, name, step

def init_opt(model, total_steps, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay) and n != 'p'], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    if args.lr_decay:
        scheduler = get_linear_schedule_with_warmup(optimizer, -1, total_steps)
    else:
        scheduler = None

    if args.ckpt:
        state = torch.load(args.ckpt)
        optimizer.load_state_dict(state['optimizer'])
    return optimizer, scheduler
