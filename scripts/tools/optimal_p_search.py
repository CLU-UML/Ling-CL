import aim
import os, torch, optuna, joblib
import tempfile
import numpy as np
from torch import nn
from options import parse_args
from model import init_model, init_opt
from data import get_dataloaders
from train import Trainer, mean
from transformers import AutoTokenizer
from lng_curr import LngCurriculum
import multiprocessing as mp

sca_names = "W,S,VP,C,T,DC,CT,CP,CN,MLS,MLT,MLC,C-S,VP-T,C-T,DC-C,DC-T,T-S,\
CT-T,CP-T,CP-C,CN-T,CN-C".split(',')
lca_names = "wordtypes,swordtypes,lextypes,slextypes,wordtokens,swordtokens,\
lextokens,slextokens,ld,ls1,ls2,vs1,vs2,cvs1,ndw,ndwz,ndwerz,ndwesz,ttr,\
msttr,cttr,rttr,logttr,uber,lv,vv1,svv1,cvv1,vv2,nv,adjv,advv,modv".split(',')
all_names = ['sent1_%s'%x for x in lca_names] + ['sent1_%s'%x for x in sca_names]\
        + ['sent2_%s'%x for x in lca_names] + ['sent2_%s'%x for x in sca_names]

def run(args, device, tokenizer, bests, steps, ps, n):
    if n % 2 == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif n % 2 == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    torch.manual_seed(0)
    np.random.seed(0)
    train_dataloader, dev_dataloader, test_dataloader,\
            dev_dataset, train_dataloader_ns = get_dataloaders(args, tokenizer, ps)

    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed = seed

        args.epoch_size = len(train_dataloader)

        name = next(tempfile._get_candidate_names())
        trainer = Trainer(args, writer=None, device=device)
        trainer.name = name

        print('[Starting Training]', trainer.name)
        trainer.train(train_dataloader, dev_dataloader, dev_dataset, train_dataloader_ns)

        best_acc, best_step = trainer.best_acc, trainer.best_step
        print('Acc:', best_acc)
        trainer.cleanup()

        bests.append(best_acc)
        steps.append(best_step/args.epoch_size)

def objective(trial):
    manager = mp.Manager()
    bests = manager.list()
    steps = manager.list()
    ps = np.array([trial.suggest_float(name, -1, 1, step=0.05) for name in all_names])
    p = mp.Process(target=run, args=(args, device, tokenizer, bests, steps, ps, trial.number))
    p.start()
    p.join()

    bests = list(bests)
    steps = list(steps)
    trial.set_user_attr('best_steps', steps)
    trial.set_user_attr('accs', bests)
    return mean(bests)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    mp.set_start_method('spawn')

    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    study_path = os.path.join(args.study_dir, args.study_name + '.pkl')
    saver = lambda study, _: joblib.dump(study, study_path)

    if os.path.isfile(study_path):
        print('[Resuming Study]')
        study = joblib.load(study_path)
    else:
        study = optuna.create_study(study_name = args.study_name, direction='maximize')

    study.optimize(objective, n_trials=400, callbacks = [saver], n_jobs = 4)
