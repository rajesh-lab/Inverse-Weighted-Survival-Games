import models
import os
import copy
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter as KMFitter
import pycox
import numpy as np


# local
import util

############################################
############ NLL ###########################
############################################

# works for marginal or conditional
def nll_batch(tgt, dist, args, is_g):
    U,Delta=tgt
    U_bin = U # data already discrete
    MAX_BIN_IDX = args.K-1
    U_max = torch.eq(U_bin,MAX_BIN_IDX)
    pmf_term = -1.0 * dist.log_pmf(U)
    
    if not is_g:
        survivor_term = -1.0 * util.safe_log(1. - dist.cdf(U,_1m=False),args.logeps)
        nll = torch.where(Delta | U_max ,pmf_term,survivor_term)
    else:
        survivor_term = -1.0 * util.safe_log(1. - dist.cdf(U,_1m=True),args.logeps)
        nll = torch.where(~Delta, pmf_term,survivor_term)
    
    if util.isnan(nll):
        assert False,"nll nan"

    nll = nll.mean(0)
    return nll



# conditional
def nll(phase, loader, model, optimizer=None, args=None, is_g=False):
    loss_meter = util.Meter()                        
    for batch_idx, batch in enumerate(loader):
        (U,Delta,X) = batch
        U = U.to(args.device)
        Delta = Delta.to(args.device)
        X = X.to(args.device)
        tgt = (U,Delta)
        bsz = U.size()[0]
        dist = util.X_to_dist(X, model, args)
        loss = nll_batch(tgt, dist, args, is_g=is_g)
        if phase=='train':
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()                        
        loss_meter.update(val = loss.item() *  bsz, N = bsz)
    return loss_meter.avg()


def nll_FG(phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None):
    fnll = nll(phase,loader,Fmodel,Foptimizer,args,is_g=False)
    gnll = nll(phase,loader,Gmodel,Goptimizer,args,is_g=True)
    return fnll, gnll


