import models
import os
import copy
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter as KMFitter
import numpy as np

# local
import catdist
import data_utils
import _concordance
import _nll
import _km

def str_to_bool(arg):
    """Convert an argument string into its boolean value.
    Args:
        arg: String representing a bool.
    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def isnan(x):
    return torch.any(torch.isnan(x))
 
def safe_log(x,eps):
    return (x+eps).log()

def clip(prob,clip_min):
    return prob.clamp(min=clip_min)

def round3(x):
    return round(x,3)

class Meter:
    def __init__(self):
        self.N = 0
        self.total = 0
    def update(self,val,N):
        self.total += val
        self.N += N
    def avg(self):
        return round(self.total / self.N,4)

def X_to_dist(X,model,args,k=None):
    pred_params = model(X)
    return catdist.CatDist(logits=pred_params, args=args, probs=None,k=k)

def X_to_FG_dists(X,Fmodel,Gmodel,args,k=None):
    Fdist = X_to_dist(X,Fmodel,args,k=k)
    Gdist = X_to_dist(X,Gmodel,args,k=k)
    return Fdist,Gdist


############################################
############ IPCW BS and BLL ###############
############################################


def IPCW_batch(fn,k,tgt,Fdist,Gdist,args,is_g=False,detach=True):
   
    if is_g:
        numer_dist=Gdist
        denom_dist=Fdist
    else:
        numer_dist=Fdist
        denom_dist=Gdist
    
    U,Delta=tgt
    kbatch = torch.ones_like(U) * k

    ncdf_k = numer_dist.leq(kbatch)
    observed = ~Delta if is_g else Delta

    if fn == 'bll_game':
        left_loss = -1.0 * safe_log(ncdf_k,args.logeps)
        right_loss = -1.0 * safe_log(1. - ncdf_k,args.logeps)
    elif fn == 'bs_game':
        left_loss = (1. - ncdf_k).pow(2)
        right_loss = ncdf_k.pow(2)
    else:
        assert False

    left_numer = left_loss * observed * (U <= kbatch)
    
    if is_g:
        left_denom = denom_dist.gt(U)
    else:
        left_denom = denom_dist.geq(U)
    left_denom = clip(left_denom, args.clip_min)

    right_numer = right_loss * (U > kbatch)
    right_denom = clip(denom_dist.gt(kbatch),args.clip_min)

    if detach:
        left_denom = left_denom.detach()
        right_denom = right_denom.detach()

    left = left_numer / left_denom
    right = right_numer / right_denom
    ipcw_loss = (left + right).mean(0)
    return ipcw_loss



def uncensored_BS_or_BLL_batch(fn,k,U,Fdist,args):
    kbatch = torch.ones_like(U) * k
    Fk = Fdist.cdf(kbatch)
    if fn=='bs_game':
        # BS(k) = E_T  [  1[T <= k] * (1-F(k))^2 + F(k)^2 1[T>k]  ]
        loss_k = torch.where(U <= kbatch, (1-Fk).pow(2), Fk.pow(2))
    else:
        # BS(k) = E_T  [  1[T <= k] * (1-F(k))^2 + F(k)^2 1[T>k]  ]
        loss_k = -1.0 * torch.where(U <= kbatch, safe_log(Fk,args.logeps), safe_log(1-Fk,args.logeps))

    assert loss_k.shape[0]==U.shape[0]
    loss_k = loss_k.mean(0)
    return loss_k

def game(fn, phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None, mode='normal'):
    return cond_bs_game(fn, phase, loader, Fmodel, Gmodel, args, Foptimizer=Foptimizer, Goptimizer=Goptimizer, mode=mode)


def cond_bs_game(fn, phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None, mode='normal'):
    Fsumm = 0.0
    Gsumm = 0.0
    for k in range(args.K-1):
        floss_meter_k = Meter()
        gloss_meter_k = Meter()
        for batch_idx, batch in enumerate(loader):     
            (U,Delta,X) = batch
            U=U.to(args.device)
            Delta=Delta.to(args.device)
            X=X.to(args.device)  
            bsz = U.shape[0]
            if phase=='train':
                Foptimizer.zero_grad()
                Goptimizer.zero_grad()
                Fdist,Gdist = X_to_FG_dists(X, Fmodel, Gmodel, args, k=None)
            else:
                Fdist,Gdist = X_to_FG_dists(X, Fmodel, Gmodel, args, k=None)
            if mode=='normal':
                floss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, args, is_g=False, detach=True)
                gloss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, args, is_g=True, detach=True)
            elif mode=='uncensored':
                Fdist = X_to_dist(X, Fmodel, args, k=None)
                floss_k = uncensored_BS_or_BLL_batch(fn, k, U, Fdist, args)
                gloss_k = torch.tensor([-1.0])
            elif mode=='kmG':
                assert phase=='test'
                Fdist = X_to_dist(X, Fmodel, args, k=None)
                G_cdfvals = _km.get_KM_cdfvals(loader, args)
                Gdist = _km.cdfvals_to_dist(G_cdfvals, bsz, args)
                floss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, args=args, is_g=False, detach=True)
                gloss_k = torch.tensor([-1.0]).to(args.device)
            else:
                assert False
            if phase=='train':
                floss_k.backward()
                Foptimizer.step()
                gloss_k.backward()
                Goptimizer.step()
            floss_meter_k.update(val = floss_k.item() *  bsz, N = bsz)
            gloss_meter_k.update(val = gloss_k.item() *  bsz, N = bsz)
        Fsumm += floss_meter_k.avg()
        Gsumm += gloss_meter_k.avg()
    Fsumm = Fsumm / (args.K-1)
    Gsumm = Gsumm / (args.K-1)
    return Fsumm,Gsumm

def train_or_val(phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None):
   
    if phase=='train':
        assert Foptimizer is not None
        assert Goptimizer is not None

    if args.loss_fn == 'nll':
        floss,gloss = _nll.nll_FG(phase, loader, Fmodel, Gmodel, args, Foptimizer, Goptimizer)
    elif args.loss_fn in ['bs_game', 'bll_game']:
        floss,gloss = game(args.loss_fn, phase, loader, Fmodel, Gmodel, args, Foptimizer, Goptimizer)
    else:
        assert False

    return floss,gloss


# for game eval
def get_model_from_file(fname,args):
    ckpt = torch.load(fname,map_location=args.device)
    model = args.model_fn(args)
    model.load_state_dict(ckpt['model_state'])
    model.to(args.device)
    model.eval()
    return model

def eval_next_dist(cur_F_file, cur_G_file,args, get_F,loader,fn):
    
    # all models
    dirr = os.path.join(args.save_dir,args.ckpt_basename)
    F_filenames = [os.path.join(dirr,args.ckpt_basename+'_F_epoch{}.pth.tar'.format(epoch)) for epoch in range(0,args.epochs)]
    G_filenames = [os.path.join(dirr,args.ckpt_basename+'_G_epoch{}.pth.tar'.format(epoch)) for epoch in range(0,args.epochs)]

    # cur models
    cur_F = get_model_from_file(cur_F_file,args)
    cur_G = get_model_from_file(cur_G_file,args)

    best_metric = 1000000.0
    best_filename = None
    
    # given cur G, find an F
    if get_F:
        print("Searching for a new F")
        for candidate_model_filename in F_filenames:
            print("--- Trying: {}".format(candidate_model_filename))
            candidate_model = get_model_from_file(candidate_model_filename,args)
            floss,_ = game(fn,'valid',loader,candidate_model,cur_G,args)
            if floss < best_metric:
                best_metric = floss
                best_filename = candidate_model_filename
        if best_filename==cur_F_file:
            changed=False
        else:
            changed=True


    # given cur F, find a G
    else:
        print("Searching for a new G")
        for candidate_model_filename in G_filenames:
            print("--- Trying: {}".format(candidate_model_filename))
            candidate_model = get_model_from_file(candidate_model_filename,args)
            _,gloss = game(fn,'valid',loader,cur_F,candidate_model,args)
            if gloss < best_metric:
                best_metric = gloss
                best_filename = candidate_model_filename
        if best_filename==cur_G_file:
            changed=False
        else:
            changed=True

    return best_filename, changed






def f_metrics(loaders,Fmodel,Gmodel,args):

    if args.dataset in ['gamma','mnist']:
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders

    if args.dataset in ['gamma','mnist']:
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders
    elif args.dataset in args.realsets:
        trainloader,valloader,testloader = loaders
    else:              
        assert False 

    fbs_km,_ = game('bs_game','test',testloader,Fmodel,Gmodel,args,mode='kmG')
    fbs_ipcw,_= game('bs_game','test',testloader,Fmodel,Gmodel,args)
    fbll_km,_ = game('bll_game','test',testloader,Fmodel,Gmodel,args,mode='kmG')
    fbll_ipcw,_ = game('bll_game','test',testloader,Fmodel,Gmodel,args)
    fnll,_ = _nll.nll_FG('test',testloader,Fmodel,Gmodel,args)         
    fconc,_ = _concordance.test_concordance_FG(testloader,testloader,Fmodel,Gmodel,args)
    
    if args.dataset in ['gamma','mnist']:
        assert torch.all(torch.eq(Ftestloader.dataset.Delta,1))
        fbs_uncensored,_ = game('bs_game','test',Ftestloader,Fmodel,Gmodel,args,mode='uncensored')
        print("test fbs uncensored",fbs_uncensored)
        fbll_uncensored,_ = game('bll_game','test',Ftestloader,Fmodel,Gmodel,args,mode='uncensored')
    else:
        fbs_uncensored = -1.0
        fbll_uncensored = -1.0

    metrics = {}     
    metrics['fnll']=round3(fnll)             
    metrics['fconc']=round3(fconc)             
    metrics['fbs_km']=round3(fbs_km)          
    metrics['fbll_km']=round3(fbll_km)      
    metrics['fbs_ipcw']=round3(fbs_ipcw)       
    metrics['fbll_ipcw']=round3(fbll_ipcw)  
    metrics['fbs_uncensored']=round3(fbs_uncensored)
    metrics['fbll_uncensored']=round3(fbll_uncensored)
    return metrics


def eval_nll(loaders,Fmodel,Gmodel,args):
    return f_metrics(loaders,Fmodel,Gmodel,args)

def eval_game(loaders,args):

    if args.dataset in ['gamma','mnist']:
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders
    elif args.dataset in args.realsets:
        trainloader,valloader,testloader = loaders
    else:
        assert False

    dirr = os.path.join(args.save_dir,args.ckpt_basename)
    cur_F_file = os.path.join(dirr,args.ckpt_basename+'_F_epoch0.pth.tar')
    cur_G_file = os.path.join(dirr,args.ckpt_basename+'_G_epoch0.pth.tar')
    while True:
        print("Trying")
        cur_F_file, F_changed = eval_next_dist(cur_F_file,cur_G_file,args,get_F=True,loader=valloader,fn='bs_game')
        cur_G_file, G_changed = eval_next_dist(cur_F_file,cur_G_file,args,get_F=False,loader=valloader,fn='bs_game')
        if not F_changed and not G_changed:
            break
   
    best_F = get_model_from_file(cur_F_file,args)
    best_G = get_model_from_file(cur_G_file,args)
    return f_metrics(loaders,best_F,best_G,args), cur_F_file,cur_G_file


