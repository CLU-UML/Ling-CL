import torch
import aim
import os, shutil, json
import numpy as np
import torch.nn.functional as F
from torch import nn
from datetime import datetime
from options import parse_args
from model import init_model, init_opt
from data import get_dataloaders, update_dataloader
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, fbeta_score
from transformers import AutoModel, AutoTokenizer, logging
from scipy.stats import pearsonr
from tqdm import tqdm
from math import ceil
from const import *
from utils.utils import ignore_padding_flatten, calc_bal_acc
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

mean = lambda l: sum(l)/len(l) if len(l) > 0 else 0.

class Trainer():
    def __init__(self, args, writer, device, dataloader, lng_names=None, lng_ids=None, name=None, glf_cfg=None):

        self.args = args
        self.writer = writer
        self.epochs = args.epochs
        self.epoch_size = args.epoch_size
        self.best_acc = 0
        self.best_step = 0
        self.device = device
        self.total_steps = self.epochs * self.epoch_size
        self.save_losses = args.save_losses
        if self.save_losses:
            self.losses = {'train': [], 'dev': []}

        self.loss_bn = nn.BatchNorm1d(1, affine = False).to(self.device)

        model, curr, model_name, step = init_model(args, device, dataloader, glf_cfg)
        optimizer, scheduler = init_opt(model, self.total_steps, args)
        if model.token_classification:
            crit = nn.BCEWithLogitsLoss(reduction='none')
        else:
            crit = nn.CrossEntropyLoss(reduction='none')

        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.curr = curr
        self.name = name if name is not None else model_name
        self.step = step
        self.lng_names = lng_names
        self.lng_ids = lng_ids

    def get_loss(self, batch):
        x = {'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)}
        add_feats = batch[self.args.add_feats].to(self.device) if self.args.add_feats else None

        logits = self.model(x, add_feats)
        labels = batch['label'].to(self.device)
        loss_unw = self.crit(logits, labels)

        if self.model.token_classification:
            loss_unw *= x['attention_mask']
            loss_unw[:,0] = 0
            loss_unw = loss_unw.sum(-1) / (loss_unw != 0).sum(-1)

        training_progress = self.step/self.total_steps
        relative_progress = (training_progress-self.args.burn_in)\
                /(1-self.args.burn_in-self.args.burn_out)

        if relative_progress >= 0 and relative_progress < 1 and isinstance(self.curr, nn.Module):
            if self.args.curr == 'glf' or self.args.curr == 'glf+':
                confs = self.curr(loss_unw, relative_progress, batch['difficulty_class'])
            elif self.args.curr in ['sigmoid', 'neg-sigmoid', 'gauss']:
                diff = batch['difficulty_score']

                confs = self.curr(relative_progress, diff.to(self.device))
            elif self.args.curr == 'sl':
                confs = self.curr(loss_unw, None)
            elif self.args.curr == 'spl':
                confs = self.curr(loss_unw)
            elif self.args.curr == 'mentornet':
                confs = self.curr(loss_unw, labels, self.step // self.epoch_size)
            elif self.args.curr == 'dp':
                confs = self.curr(loss_unw, batch['diff'].to(self.device))
            else:
                raise NotImplementedError()
        else:
            confs = torch.ones_like(loss_unw)

        # confs = confs.reshape(-1)
        eps = 1e-5

        loss_w = confs * loss_unw
        if self.args.sel_bp:
            total_loss = loss_w[confs > 1e-5].sum() / max(eps, confs[confs > 1e-5].sum())
        else:
            total_loss = loss_w.sum() / max(eps, confs.sum())

        return logits, confs, total_loss, loss_unw.mean(), loss_unw, loss_w

    def evaluate(self, dataloader, count=None, return_preds = False, return_loss = False, return_scores=False):
        losses = []
        losses_unw = []
        accs = []
        self.model.eval()
        if isinstance(self.curr, nn.Module):
            self.curr.eval()
        acc_class = [[] for i in range(3)]
        loss_unw_class = [[] for i in range(3)]
        loss_w_class = [[] for i in range(3)]
        confs = [[] for i in range(3)]
        trues, preds, scores = [], [], []
        full_loss = torch.tensor([])
        full_acc = []
        ids_map = {}
        offset = 0
        counter = 0
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                logits, conf, loss, loss_unw, \
                        all_loss_unw, all_loss_w = self.get_loss(batch)
            true = batch['label']
            logits_max = logits.max(-1)
            if self.model.token_classification:
                pred = torch.where(logits > 0, 1, 0).cpu()
            else:
                pred = logits_max.indices.cpu()
                score = logits_max.values.cpu()

            batch_accs = (true == pred).float()
            if self.model.token_classification:
                batch_errors = 1 - batch_accs
                score = batch_errors.mean(1)
            

            if self.model.token_classification:
                batch_accs *= batch['attention_mask']
                batch_accs[:,0] = 0
                batch_accs = batch_accs.sum(1) / (batch['attention_mask'].sum(1) - 1)

                true = ignore_padding_flatten(true, batch['attention_mask'])
                pred = ignore_padding_flatten(pred, batch['attention_mask'])
                for j in range(batch['input_ids'].shape[0]):
                    len_x = sum(batch['attention_mask'][j]) - 1
                    ids_map[counter] = list(range(offset, offset + len_x))
                    offset += len_x
                    counter += 1

            full_loss = torch.cat([full_loss, all_loss_unw.cpu()])
            if return_loss:
                full_acc += batch_accs.int().tolist()
            trues.extend(true.tolist())
            preds.extend(pred.tolist())
            scores.extend(score.tolist())
            losses_unw.append(loss_unw.detach().item())
            losses.append(loss.detach().item())
            if self.args.eval_class in batch:
                for c in range(3):
                    class_ids = batch[self.args.eval_class] == c
                    class_accs = batch_accs[class_ids]
                    acc_class[c].extend(class_accs.tolist())

                    class_loss_unw = all_loss_unw[class_ids]
                    loss_unw_class[c].extend(class_loss_unw.tolist())

                    class_loss_w = all_loss_w[class_ids]
                    loss_w_class[c].extend(class_loss_w.tolist())

                    class_conf = conf[batch[self.args.eval_class] == c]
                    if class_conf.numel() != 0:
                        confs[c].append(class_conf.mean().item())
            if count and i > 0 and i % count == 0:
                break
        if self.args.diff_score == 'lng_w' or self.args.multiview:
            lng = torch.tensor(dataloader.dataset['lng']).cpu()
            if self.args.lng_method == 'corr':
                lng_loss = torch.cat([lng, full_loss.view(-1,1)], 1)
                p = torch.corrcoef(lng_loss.T)[-1, :-1]
                p = torch.nan_to_num(p)
            elif self.args.lng_method == 'opt':
                p = torch.linalg.lstsq(lng, full_loss.view(-1,1)).solution[:,0]

            if not self.args.eval_only:
                if self.args.curr in ['sampling', 'competence'] and self.args.multiview:
                    self.curr.update_p(p.numpy())
                else:
                    self.train_dataloader = update_dataloader(self.train_dataloader, p.numpy(), self.args)
                    self.train_iter = iter(self.train_dataloader)
                    if self.args.curr in ['sampling', 'competence']:
                        diff = self.train_dataloader.dataset['difficulty_score']
                        self.curr.update_indices(diff)

            if self.writer is not None and self.lng_names is not None:
                for i, (idx,n) in enumerate(zip(self.lng_ids, self.lng_names)):
                    sent = 'p' if idx < self.args.lng_n//2 else 'h'
                    self.writer.track(p[i], step = self.step,
                            epoch = self.step // self.epoch_size,
                            name = 'lng-weight', context = {'name': n, 'sent': sent})
                    bal_acc = calc_bal_acc(np.array(trues), np.array(preds), lng[:,idx], n+sent, ids_map = ids_map)
                    self.writer.track(bal_acc, step = self.step,
                            epoch = self.step // self.epoch_size,
                            name = 'bal-acc', context = {'name': n, 'sent': sent})

        if self.model.token_classification:
            mean_acc = fbeta_score(trues, preds, beta=self.args.fbeta)
        elif 'cola' in self.args.data:
            mean_acc = matthews_corrcoef(trues, preds)
        else:
            mean_acc = accuracy_score(trues, preds)

        res = [mean(losses_unw), mean(losses), 
                mean_acc, confs, 
                [mean(a) for a in acc_class],
                [mean(l) for l in loss_w_class],
                [mean(l) for l in loss_unw_class]
                ]

        if return_scores:
            res.append(scores)
        if return_preds:
            res.append(preds)
        if return_loss:
            res.append(full_loss.tolist())
        if return_loss:
            res.append(full_acc)

        self.model.train()
        if isinstance(self.curr, nn.Module):
            self.curr.train()

        return res

    def save(self):
        with open('{}/{}_args'.format(self.args.ckpt_dir, self.name), 'w') as f:
            json.dump(self.args.__dict__, f)
        self.model.backbone.save_pretrained('{}/{}_best_model'.format(self.args.ckpt_dir,
            self.name))
        torch.save({
            'model': {k: v for k,v in self.model.state_dict().items()
                if 'backbone' not in k},
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'curr': self.curr,
            'best_step': self.best_step,
            'best_acc': self.best_acc},
            '{}/{}_best_meta.pt'.format(self.args.ckpt_dir, self.name))

    def load(self, name):
        state = torch.load("%s_meta.pt"%name)
        self.model.load_state_dict(state['model'], strict=False)
        self.step = state['step']
        self.curr = state['curr']
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_step = state['best_step']
        self.best_acc = state['best_acc']

        self.model.backbone = AutoModel.from_pretrained("%s_model"%name).to(self.device)

    def load_best(self):
        ckpt_path = '{}/{}_best'.format(self.args.ckpt_dir, self.name)
        if os.path.exists('%s_model'%ckpt_path):
            print("[Loading Best] Current: %d -> Best: %d (%.4f) [%s]"%(self.step, self.best_step,
                self.best_acc, ckpt_path))
            self.load(ckpt_path)
        return self.best_acc, self.best_step

    def cleanup(self):
        ckpt_path = '{}/{}_best'.format(self.args.ckpt_dir, self.name)
        if os.path.exists('%s_model'%ckpt_path):
            os.remove("%s_meta.pt"%ckpt_path)
            shutil.rmtree("%s_model"%ckpt_path)

    def train(self, train_dataloader, dev_dataloader, dev_dataset, train_ns = None):
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        while self.step < self.total_steps:
            self.model.train()

            if self.args.curr == 'sampling' or self.args.curr == 'competence':
                training_progress = self.step/self.total_steps
                relative_progress = (training_progress-self.args.burn_in)\
                        /(1-self.args.burn_in-self.args.burn_out)
                relative_progress = max(0, min(1, relative_progress))
                if self.args.curr == 'sampling':
                    progress = (1 + int(relative_progress*self.args.sampling_int))\
                            / self.args.sampling_int
                elif self.args.curr == 'competence':
                    progress = relative_progress
                self.curr.update_sampler(progress)
                self.train_iter = iter(self.train_dataloader)
            if self.args.curr == 'datasel':
                if self.step == 0:
                    self.curr.update_sampler(difficulty=None, burn_in=True)
                elif self.step >= (self.total_steps*self.args.burn_in) \
                        and self.step % self.epoch_size == 0:
                    train_scores = self.evaluate(train_ns, return_scores = True)[-1]
                    self.curr.update_sampler(difficulty=train_scores)
                self.train_iter = iter(self.train_dataloader)

            batch = next(self.train_iter, None)
            if batch is None:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter, None)

            logits, conf, loss, loss_unw, \
                    all_loss_unw, all_loss_w = self.get_loss(batch)

            loss /= self.args.grad_accumulation
            loss.backward()

            if (self.step+1) % self.args.grad_accumulation == 0:
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                            self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            if self.step % 50 == 0 and self.writer is not None:
                if self.model.token_classification:
                    pred = torch.where(logits > 0, 1, 0).cpu()
                else:
                    pred = logits.argmax(-1).cpu()


                true = batch['label']
                batch_accs = (true == pred).float()
                if self.model.token_classification:
                    batch_accs *= batch['attention_mask']
                    batch_accs[:,0] = 0
                    batch_accs = batch_accs.sum(1) / (batch['attention_mask'].sum(1) - 1)

                    true = ignore_padding_flatten(true, batch['attention_mask'])
                    pred = ignore_padding_flatten(pred, batch['attention_mask'])
                    acc = fbeta_score(true, pred, beta=self.args.fbeta)
                elif 'cola' in self.args.data:
                    acc = matthews_corrcoef(true, pred)
                else:
                    acc = accuracy_score(true, pred)

                self.writer.track(loss.detach().item(), step = self.step,
                        epoch = self.step // self.epoch_size,
                        name = 'loss_weighted', context = {'split': 'train'})
                self.writer.track(loss_unw.detach().item(), step = self.step,
                        epoch = self.step // self.epoch_size,
                        name = 'loss_unweighted', context = {'split': 'train'})
                self.writer.track(acc, step = self.step,
                        epoch = self.step // self.epoch_size,
                        name = 'acc', context = {'split': 'train'})

                if self.args.eval_class in batch:
                    for c,c_name in enumerate(['easy', 'med', 'hard']):
                        class_ids = batch[self.args.eval_class] == c
                        class_conf = conf[class_ids]
                        class_acc = batch_accs[class_ids]
                        class_loss_w = all_loss_w[class_ids]
                        class_loss_unw = all_loss_unw[class_ids]
                        if class_conf.numel() != 0:
                            self.writer.track(class_conf.mean().item(), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'conf', context = {'split': 'train', 'subset': c_name})
                            self.writer.track(mean(class_acc), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'acc_bd',
                                    context = {'split': 'train', 'subset': c_name})
                            self.writer.track(class_loss_w.mean().item(), step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'loss_w_bd',
                                    context = {'split': 'train', 'subset': c_name})
                            self.writer.track(class_loss_unw.mean().item(),
                                    step = self.step,
                                epoch = self.step // self.epoch_size,
                                    name = 'loss_unw_bd',
                                    context = {'split': 'train', 'subset': c_name})

            if (self.step + 1) % (self.epoch_size // self.args.val_freq) == 0:
                res = self.evaluate(dev_dataloader, return_loss = self.save_losses,
                        count=self.args.val_count, return_preds=True)
                loss_unw, loss, acc, conf,\
                        class_accs, class_loss_w, class_loss_unw = res[:7]
                if self.save_losses:
                    res_train = self.evaluate(train_ns, return_loss = True)
                    self.losses['train'].append(res_train[-2])
                    self.losses['dev'].append(res[-2])

                if self.writer is not None:
                    self.writer.track(loss_unw, name = 'loss_unweighted', step = self.step,
                        epoch = self.step // self.epoch_size,
                            context = {'split': 'val'})
                    self.writer.track(loss, name = 'loss_weighted', step = self.step,
                        epoch = self.step // self.epoch_size,
                            context = {'split': 'val'})
                    self.writer.track(acc, name = 'acc', step = self.step,
                        epoch = self.step // self.epoch_size,
                            context = {'split': 'val'})

                    self.writer.track(class_accs[0], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'acc_bd', context = {'split': 'val', 'subset': 'easy'})
                    self.writer.track(class_accs[1], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'acc_bd', context = {'split': 'val', 'subset': 'med'})
                    self.writer.track(class_accs[2], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'acc_bd', context = {'split': 'val', 'subset': 'hard'})

                    self.writer.track(class_loss_w[0], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'loss_w_bd', context = {'split': 'val', 'subset': 'easy'})
                    self.writer.track(class_loss_w[1], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'loss_w_bd', context = {'split': 'val', 'subset': 'med'})
                    self.writer.track(class_loss_w[2], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'loss_w_bd', context = {'split': 'val', 'subset': 'hard'})

                    self.writer.track(class_loss_unw[0], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'loss_unw_bd',
                            context = {'split': 'val', 'subset': 'easy'})
                    self.writer.track(class_loss_unw[1], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'loss_unw_bd',
                            context = {'split': 'val', 'subset': 'med'})
                    self.writer.track(class_loss_unw[2], step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'loss_unw_bd',
                            context = {'split': 'val', 'subset': 'hard'})

                    self.writer.track(mean(conf[0]), step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'conf', context = {'split': 'val', 'subset': 'easy'})
                    self.writer.track(mean(conf[1]), step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'conf', context = {'split': 'val', 'subset': 'med'})
                    self.writer.track(mean(conf[2]), step = self.step,
                        epoch = self.step // self.epoch_size,
                            name = 'conf', context = {'split': 'val', 'subset': 'hard'})

                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_step = self.step
                    self.save()
                    if not self.args.debug:
                        np.save(f'{self.args.preds_dir}/{self.name}_dev.npy', res[-1])

            self.step += 1
def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    if (args.diff_score and 'lng' in args.diff_score) or args.add_feats == 'lng':
        if args.data in ('cola', 'sst2', 'an-dataset', 'fce-error-detection'):
            lng_names = lca_names + sca_names + lingfeat_names
        elif args.data in ('snli', 'anli', 'rte', 'chaosnli'):
            lng_names = lca_names + sca_names + lingfeat_names + lca_names + sca_names + lingfeat_names
        else:
            raise ValueError
    else:
        lng_names = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, return_token_type_ids=False)
    torch.manual_seed(0)
    np.random.seed(0)
    train_dataloader, dev_dataloader, test_dataloader,\
            dev_dataset, train_dataloader_ns = get_dataloaders(args, tokenizer, lng_names=lng_names)

    if args.lng_ids:
        lng_ids = np.load(os.path.join(args.data_dir, 'lng_ids', args.data) + '.npy')
        lng_names = [x for idx, x in enumerate(lng_names) if idx in lng_ids]
    elif args.lng_n is not None:
        lng_ids = list(range(args.lng_n))
    else:
        lng_ids = None

    for seed in args.seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed = seed

        args.epoch_size = ceil(len(train_dataloader.dataset) / args.batch_size)

        writer = aim.Run(experiment=args.aim_exp, repo=args.aim_repo, 
                system_tracking_interval=30) if not args.debug else None

        if writer is not None:
            writer['hparams'] = args.__dict__
            writer['meta'] = {'time': datetime.now().strftime('%y%m%d/%H:%M:%S')}

        trainer = Trainer(args, writer, device, train_dataloader,
                lng_names=lng_names, lng_ids=lng_ids)

        if not args.eval_only:
            print('[Starting Training]', trainer.name)
            trainer.train(train_dataloader, dev_dataloader,
                    dev_dataset, train_dataloader_ns)

        if args.save_losses:
            np.savez('/data/mohamed/losses/%s_%d.npz'%(args.data, seed), **trainer.losses)

        best_acc, best_step = trainer.load_best()
        if writer is not None:
            writer.track(best_acc, name = 'best_val_acc')
            writer.track(best_step, name = 'best_step')


        if args.data not in ['cola', 'sst2', 'rte']:
            print('[Testing]')
            res = trainer.evaluate(test_dataloader, return_loss = False,
                    return_preds=True)
            if not args.debug:
                np.save(f'{args.preds_dir}/{trainer.name}_test.npy', res[-1])
            acc, confs, class_acc  = res[2:5]
            print('Acc:', acc)
            print("0: {:.4f}\n1: {:.4f}\n2: {:.4f}".format(*class_acc))
            if writer is not None:
                writer.track(acc, name = 'test_acc', context = {'subset': 'overall'})
                writer.track(class_acc[0], name = 'test_acc', context = {'subset': 'easy'})
                writer.track(class_acc[1], name = 'test_acc', context = {'subset': 'med'})
                writer.track(class_acc[2], name = 'test_acc', context = {'subset': 'hard'})
        trainer.cleanup()

if __name__ == '__main__':
    # import cProfile, pstats
    # profile = cProfile.Profile()
    # profile.runcall(main)
    # ps = pstats.Stats(profile).sort_stats('time')
    # ps.print_stats(100)
    main()
