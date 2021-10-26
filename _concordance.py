import models
import os
import copy
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter as KMFitter
import pycox
import numpy as np

# local
import catdist

############################################
############ CONCORDANCE ###################
############################################


def concordance(loader,model,args,is_g=False):                                                  
    all_U = []                                                                            
    all_Delta = []                                                                        
    all_pred_time = []                                                                    
    for idx, (U,Delta,X) in enumerate(loader):
        U=U.to(args.device)
        Delta=Delta.to(args.device)
        X=X.to(args.device)    
        all_U.append(U)
        Delta=Delta.to(args.device)
        all_Delta.append(Delta)
        pred_params = model(X)
        dist = catdist.CatDist(logits=pred_params,args=args,probs=None,k=None)
        pred_time = dist.predict_time()
        if torch.any(torch.isnan(pred_time)):
            assert False, "bad pred time in conc"

        all_pred_time.append(pred_time)
    all_U = torch.cat(all_U).detach().cpu().numpy()
    all_Delta = torch.cat(all_Delta).long().detach().cpu().numpy()
    all_pred_time = torch.cat(all_pred_time).detach().cpu().numpy()
   
    
    if is_g:
        all_is_observed = ~all_Delta
    else:
        all_is_observed = all_Delta

    concordance = concordance_index(all_U, all_pred_time, all_is_observed)
    return concordance

def test_concordance_FG(Floader,Gloader,Fmodel,Gmodel,args):
    F_conc_te=concordance(loader=Floader,model=Fmodel,args=args,is_g=False)
    G_conc_te=concordance(loader=Gloader,model=Gmodel,args=args,is_g=True)
    return F_conc_te,G_conc_te


