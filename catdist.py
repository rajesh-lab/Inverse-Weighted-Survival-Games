import torch
import torch.nn as nn
import numpy as np

# local
import data_utils
import util

class CatDist():
    def __init__(self, logits, args, probs=None, k=None):
        self.args = args
        self.K = self.args.K

        if probs is not None:
            self.probs = probs
        else:
            
            bsz = logits.shape[0]                                                                                                                                                                           
            zeros = torch.zeros(bsz,1).to(logits.device)
             
            if args.kminusone==1:
                new_logits = torch.cat([logits,zeros],dim=-1)
            elif args.kminusone==0:
                new_logits = logits
            else:
                assert False
             
            assert new_logits.shape==(bsz,self.args.K)

            self.probs = self.softmax_transform(new_logits)


        if k is not None:
            # works marginally
            bsz=self.probs.shape[0]
            probs_const = self.probs.detach()
            probs_k_with_grad = self.probs[:,k].unsqueeze(1).repeat(1,self.K)
            assert probs_k_with_grad.shape==(bsz,self.K)
            idx = torch.ones_like(probs_k_with_grad).to(probs_k_with_grad.device).long() * k
            self.probs = probs_const.scatter(1,idx,probs_k_with_grad)


        if torch.any(torch.isnan(self.probs)):
            assert False, "catdist probs are bad"
        assert torch.all(torch.abs(self.probs.sum(-1) - 1.0) < 1e-4)

    def sample(self,sample_shape):
        torchdist=torch.distributions.Categorical(probs=self.probs)
        return torchdist.sample(sample_shape=sample_shape)

    def predict_time(self):
        ar = torch.arange(self.K).to(self.args.device)
        ar = ar.unsqueeze(0).repeat(self.probs.shape[0],1)
        return (ar * self.probs).sum(-1)

    def log_pmf(self,k):
        bin_times = k # data already discrete
        BSZ=bin_times.shape[0]
        pk = self.probs[torch.arange(BSZ),bin_times]
        logpk = util.safe_log(pk,self.args.logeps)
        assert bin_times.shape==logpk.shape
        return logpk

    def pmf(self,k):
        bin_times = k # data already discrete
        BSZ=bin_times.shape[0]
        pk = self.probs[torch.arange(BSZ),bin_times]
        assert bin_times.shape==pk.shape
        return pk
        #return self.log_pmf(k).exp()
        
    def cdf(self,k,_1m=False):
        #bin_times, _ = self.cat_bin_target(k)
        bin_times = k # data already discrete
        # IMPORTANT
        if _1m:
            bin_times = bin_times-1.0

        BSZ = bin_times.shape[0]
        bin_times_wide = bin_times.unsqueeze(-1)
        probs_batch = self.probs

        indices=torch.arange(self.K).view(1,-1).to(self.args.device)
        # compute some masks
        # 1's up to but not including correct bin, then zeros
        mask1 = (bin_times_wide > indices).float()
        # 1 up to and including correct bin, then zeros
        mask2 = (bin_times_wide >= indices).float()
        cdf_km1 = (probs_batch * mask1).sum(dim=-1)
        prob_k = probs_batch[torch.arange(BSZ), bin_times_wide.long().squeeze()]
        cdf_k = (probs_batch * mask2).sum(dim=-1)

        # IMPORTANT
        if _1m:
            cdf_k = torch.where(bin_times >= 0.0, 
                                cdf_k, 
                                torch.zeros_like(cdf_k))
        return cdf_k

    def eq(self,k): 
        return self.pmf(k)
    def leq(self,k):
        return self.cdf(k)
    def lt(self,k): 
        return self.cdf(k) - self.pmf(k)
    def geq(self,k):
        return 1. - self.lt(k)
    def gt(self,k): 
        return 1. - self.leq(k)

    def softmax_transform(self,logits):
        probs = nn.Softmax(dim=-1)(logits)
        return probs

