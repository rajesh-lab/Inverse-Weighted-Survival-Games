import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import warnings
import time
from pathlib import Path
import torch.nn.functional as F
import torch.utils.data as data
import pdb
import argparse
import pickle
from datetime import datetime
import copy

# local
import data_utils
import util
import models
import _saver

parser = argparse.ArgumentParser(description='survival analysis langone data')
parser.add_argument('--K',type=int,default=20)
parser.add_argument('--lr',type=float, default=0.001)
parser.add_argument('--train_batch_sz',type=int,default=1024)
parser.add_argument('--test_batch_sz',type=int,default=1024)
parser.add_argument('--loss_fn',type=str,default='nll')
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--N_train',type=int,default=1024)
parser.add_argument('--N_val',type=int,default=4096)
parser.add_argument('--N_test',type=int,default=4096)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for the DataLoader.')
parser.add_argument('--dropout_rate',type=float,default=0.0)
parser.add_argument('--save_dir',type=str,default='ckpts')
parser.add_argument('--clip_min',type=float,default=0.001)
parser.add_argument('--dataset',type=str,default='gamma')
parser.add_argument('--logeps',type=float,default=1e-4)
parser.add_argument('--ckpt_basename',type=str,default='tmp')
parser.add_argument('--m',type=int,default=1)
parser.add_argument('--kminusone',type=int,default=0)
args = parser.parse_args()
print(args)


def round3(x):
    return round(x,3)

if __name__ == '__main__':

    # print date
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

    args.realsets = ['metabric','support','gbsg','flchain','nwtco','gbm','gbmlgg','nacd','nacdcol','brca','read']
    args.game_losses = ['bs_game','bll_game']

    # catch nans
    torch.set_anomaly_enabled(True)

    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model fn
    if args.dataset=='mnist':
        args.model_fn = models.CatNNConv
    else:
        args.model_fn = models.CatNN
    
    # data
    if args.dataset=='gamma':
        loaders = data_utils.make_gamma_loaders(args)
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders
        args.D_in = trainloader.dataset.X.shape[1]
    elif args.dataset=='mnist':
        loaders = data_utils.make_mnist_loaders(args)
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders
        args.D_in1 = trainloader.dataset.X.shape[-2]
        args.D_in2 = trainloader.dataset.X.shape[-1]
    elif args.dataset in args.realsets:
        loaders = data_utils.make_real_loaders(args)
        trainloader,valloader,testloader = loaders
        args.D_in = trainloader.dataset.X.shape[1]
    else:
        assert False
     
    # model
    Fmodel = args.model_fn(args).to(args.device)
    Gmodel = args.model_fn(args).to(args.device)

    # optimizer
    Foptimizer = torch.optim.Adam(Fmodel.parameters(),lr=args.lr)
    Goptimizer = torch.optim.Adam(Gmodel.parameters(),lr=args.lr)

    # saver
    Fsaver = _saver.ModelSaver(args,is_g=False)
    Gsaver = _saver.ModelSaver(args,is_g=True)
        
    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    metrics = {}
    metrics['datetime'] = dt_string
    args_copy = copy.deepcopy(vars(args))
    del args_copy['model_fn']
    metrics['args'] = args_copy
    metrics['during_training'] = {'F_train_loss':[], 
                                  'G_train_loss':[],
                                  'F_val_loss':[],
                                  'G_val_loss':[],
                                  'saved_F':[],
                                  'saved_G':[]
                                 }


    # start train  
    for epoch in range(args.epochs):
        print("*"*69)
        print("EPOCH {}".format(epoch))
        print("*"*69)

        Fmodel.train()
        Gmodel.train()

        floss_tr,gloss_tr = util.train_or_val('train',trainloader,Fmodel,Gmodel,args,Foptimizer,Goptimizer)

        Fmodel.eval()
        Gmodel.eval()

        fbs_km,_ = util.game('bs_game','test',testloader,Fmodel,Gmodel,args,mode='kmG')
        print("fbs km",fbs_km)

        if args.loss_fn=='nll':
            # val
            floss_va,gloss_va = util.train_or_val('val',valloader,Fmodel,Gmodel,args)
        else:
            floss_va,gloss_va = 0.0,0.0

        metrics['during_training']['F_train_loss'].append(round3(floss_tr))
        metrics['during_training']['G_train_loss'].append(round3(gloss_tr))
        metrics['during_training']['F_val_loss'].append(round3(floss_va))
        metrics['during_training']['G_val_loss'].append(round3(gloss_va))

        saved_F = Fsaver.maybe_save(epoch, Fmodel, floss_va)
        saved_G = Gsaver.maybe_save(epoch, Gmodel, gloss_va)
        metrics['during_training']['saved_F'].append(1 if saved_F else 0)
        metrics['during_training']['saved_G'].append(1 if saved_G else 0)

        if args.loss_fn in args.game_losses:
            Fsaver.always_save(epoch, Fmodel)          
            Gsaver.always_save(epoch, Gmodel)
    #end train
    # start test

    with torch.no_grad():

        if args.loss_fn=='nll':
            Fmodel = Fsaver.load_best().to(args.device)
            Gmodel = Gsaver.load_best().to(args.device)
            Fmodel.eval()
            Gmodel.eval()
            ret = util.eval_nll(loaders,Fmodel,Gmodel,args)
        else:
            ret,bestF,bestG = util.eval_game(loaders,args)
            metrics['bestF']=bestF
            metrics['bestG']=bestG
        metrics['best_val'] = ret
    if not os.path.exists('results/'):
        os.makedirs('results')
    metric_save_path = os.path.join('results',args.ckpt_basename+'.pkl')
    print("Writing results to",metric_save_path)
    with open(metric_save_path, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
