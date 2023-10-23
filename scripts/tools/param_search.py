import os, torch, optuna, joblib
import tempfile
import numpy as np
from torch import nn
from train import Trainer, mean
from transformers import AutoTokenizer
from math import ceil
from options import parse_args
from data import get_dataloaders, update_dataloader
from const import *

def objective(trial):
    p = trial.suggest_int('p', 1, 10, step=1)
    c0 = trial.suggest_float('c0', 0, 1, step=0.1)
    args.comp_p = p
    args.comp_c0 = c0

    bests, steps = [], []
    for seed in range(2):
        torch.manual_seed(seed)
        train_dataloader, dev_dataloader, test_dataloader,\
                dev_dataset, train_dataloader_ns = get_dataloaders(args, tokenizer, lng_names=lng_names)
        args.epoch_size = ceil(len(train_dataloader.dataset) / args.batch_size)

        writer = None

        name = next(tempfile._get_candidate_names())
        print(name)
        trainer = Trainer(args, writer, device, train_dataloader,
                lng_names=lng_names, lng_ids=lng_ids, name=name)

        print('[Starting Training]')
        trainer.train(train_dataloader, dev_dataloader,
                dev_dataset, train_dataloader_ns)
        best_acc, best_step = trainer.load_best()
        print('Acc:', best_acc)
        trainer.cleanup()

        bests.append(best_acc)
        steps.append(best_step/args.epoch_size)

    trial.set_user_attr('best_steps', steps)
    trial.set_user_attr('dev_accs', bests)
    return mean(bests)

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data in ('cola', 'sst2'):
        lng_names = lca_names + sca_names + lingfeat_names
    elif args.data in ('snli', 'anli', 'rte'):
        lng_names = lca_names + sca_names + lingfeat_names + lca_names + sca_names + lingfeat_names
    else:
        raise ValueError

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, return_token_type_ids=False)
    torch.manual_seed(0)
    np.random.seed(0)

    if args.lng_ids:
        lng_ids = np.load(os.path.join(args.data_dir, 'lng_ids', args.data) + '.npy')
        lng_names = [x for idx, x in enumerate(lng_names) if idx in lng_ids]
    elif args.lng_n is not None:
        lng_ids = list(range(args.lng_n))
    else:
        lng_ids = None

    study_path = os.path.join(args.study_dir, args.study_name + '.pkl')
    saver = lambda study, _: joblib.dump(study, study_path)

    if os.path.isfile(study_path):
        study = joblib.load(study_path)
        print('[Loaded study] %s'%study_path)
    else:
        study = optuna.create_study(study_name = args.study_name, direction='maximize')

    study.optimize(objective, n_trials=20, callbacks = [saver], n_jobs = 1)
