import models
import os
import copy
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter as KMFitter
import pycox
import numpy as np

# local
import catdist
import data_utils
import _concordance
import _nll
import _saver

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



############################################
############ KM G IPCW F BS and BLL ########
############################################



def cdfvals_to_probs(cdfvals,args):
    K=cdfvals.shape[1]
    Gprobs = torch.zeros_like(cdfvals).to(args.device)
    Gprobs[:,0] = cdfvals[:,0]
    for k in range(1,K-1):
        Gprobs[:,k] = cdfvals[:,k] - cdfvals[:,k-1]
    Gprobs[:,K-1] = 1 - (Gprobs[:,:K-1]).sum(-1)
    return Gprobs

def cdfvals_to_dist(cdfvals,bsz,args):
    cdfvals = cdfvals.unsqueeze(0).repeat(bsz,1)
    Gprobs = cdfvals_to_probs(cdfvals,args) 
    assert torch.all( (Gprobs.sum(-1) - 1.0).abs() < 1e-4)
    Gdist = catdist.CatDist(logits=None, args=args, probs=Gprobs, k=None)
    return Gdist

def get_KM_cdfvals(loader,args):
    u=loader.dataset.U
    delta=loader.dataset.Delta
    durations = u.cpu().numpy()
    is_censored = ~delta.cpu().numpy()
    km = pycox.utils.kaplan_meier
    surv_func = km(durations,is_censored).to_numpy()
    cdf_func = 1. - surv_func
    km_support = np.sort(np.unique(durations))    
    cdfvals = torch.zeros(args.K).to(args.device)
    for i,val in enumerate(km_support):
        cdfvals[val] = cdf_func[i]
    for i,val in enumerate(cdfvals):
        if i > 0:
            if val==0.0:
                cdfvals[i]=cdfvals[i-1]
    return cdfvals
