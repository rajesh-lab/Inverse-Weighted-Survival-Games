import torch
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pdb
import os
import random
import warnings
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets, transforms

def mean_var_to_alpha_beta(mean,var):
    alpha = mean.pow(2) / var
    beta = mean / var
    return alpha,beta

def synthetic_samples_to_loaders(args,X,T,C,train_N,valid_N):

    print("min T", T.min())
    print("max T", T.max())
    print("min C", C.min())
    print("max C", C.max())

    #############################################
    ############### Discrete  ###################
    #############################################

    def get_bin_boundaries(times, K):
        percents = np.arange(K+1) * 100./ K
        bin_boundaries = np.percentile(times, percents)
        return torch.tensor(bin_boundaries)

    boundaries = get_bin_boundaries(T,args.K)

    def get_bin(time, boundaries):
        original_shape = time.shape
        time = time.unsqueeze(-1)
        boundaries_to_consider = boundaries[1:-1]
        time_cat = (time > boundaries_to_consider)
        time_cat = time_cat.sum(-1)
        assert time_cat.shape == original_shape
        return time_cat

    # use T boundaries for both
    T_bin = get_bin(T, boundaries)
    C_bin = get_bin(C, boundaries)
    Delta = T_bin <= C_bin
    print(" {}  / {} observed ".format(Delta.sum(),Delta.shape[0]))
    U_bin = torch.where(Delta, T_bin, C_bin)


    print("X SHAPE",X.shape)
    print("T SHAPE",T.shape)
    print("C SHAPE",C.shape)
    print("T BIN SHAPE",T_bin.shape)
    print("C BIN SHAPE",C_bin.shape)
    print("U BIN SHAPE",U_bin.shape)
    print("DELTA SHAPE",Delta.shape)


    # TRAIN VALID TEST SPLIT

    VAL_SPLIT = train_N
    TEST_SPLIT = train_N + valid_N

    X_tr = X[: VAL_SPLIT]
    X_va = X[VAL_SPLIT : TEST_SPLIT]
    X_te = X[TEST_SPLIT : ]

    T_tr = T_bin[: VAL_SPLIT]
    T_va = T_bin[VAL_SPLIT : TEST_SPLIT]
    T_te = T_bin[TEST_SPLIT : ]

    C_tr = C_bin[: VAL_SPLIT]
    C_va = C_bin[VAL_SPLIT : TEST_SPLIT]
    C_te = C_bin[TEST_SPLIT : ]

    U_tr = U_bin[: VAL_SPLIT]
    U_va = U_bin[VAL_SPLIT : TEST_SPLIT]
    U_te = U_bin[TEST_SPLIT : ]

    Delta_tr = Delta[: VAL_SPLIT]
    Delta_va = Delta[VAL_SPLIT : TEST_SPLIT]
    Delta_te = Delta[TEST_SPLIT : ]

    trainset = SyntheticDataset(U=U_tr[:args.N_train],Delta=Delta_tr[:args.N_train],X=X_tr[:args.N_train])
    valset   = SyntheticDataset(U=U_va[:args.N_val],Delta=Delta_va[:args.N_val],X=X_va[:args.N_val])
    testset  = SyntheticDataset(U=U_te[:args.N_test],Delta=Delta_te[:args.N_test],X=X_te[:args.N_test])
    Ftestset = SyntheticDataset(U=T_te[:args.N_test],Delta=torch.ones_like(Delta_te)[:args.N_test],X=X_te[:args.N_test])
    Gtestset = SyntheticDataset(U=C_te[:args.N_test],Delta=torch.zeros_like(Delta_te)[:args.N_test],X=X_te[:args.N_test])
    
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.train_batch_sz,num_workers=args.num_workers,shuffle=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=args.test_batch_sz,num_workers=args.num_workers,shuffle=False)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args.test_batch_sz,num_workers=args.num_workers,shuffle=False)
    Ftestloader = torch.utils.data.DataLoader(Ftestset,batch_size=args.test_batch_sz,num_workers=args.num_workers,shuffle=False)
    Gtestloader = torch.utils.data.DataLoader(Gtestset,batch_size=args.test_batch_sz,num_workers=args.num_workers,shuffle=False)

    assert torch.all(torch.eq(Ftestloader.dataset.Delta,1))
    assert torch.all(torch.eq(Gtestloader.dataset.Delta,0))
    
    return trainloader, valloader, testloader,Ftestloader, Gtestloader





def make_mnist_loaders(args):


    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('./data',train=True, download=True,transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)

    def dset_to_xy(mnist):
        xlist = []
        ylist = []
        for x,y in mnist:
            xlist.append(x.clone().numpy()) 
            ylist.append(np.array(y)) 
        return np.array(xlist),np.array(ylist)
    
    X_tr,Y_tr = dset_to_xy(mnist_train)
    X_te,Y_te = dset_to_xy(mnist_test)

    train_N=Y_tr.shape[0]
    valtest_N = Y_te.shape[0]
    val_N = int(0.5 * valtest_N)
    test_N = valtest_N - val_N

    X = np.concatenate((X_tr,X_te),axis=0)
    Y = np.concatenate((Y_tr,Y_te),axis=0)


    X = torch.tensor(X)
    Y = torch.tensor(Y)

    def sim_times(y):

        coef_T  = 10.0
        coef_C = 9.9
        mean_T = (coef_T * (y+1))
        mean_C = (coef_C * (y+1))
        args.conditional_var = 0.05
        a_T,b_T = mean_var_to_alpha_beta(mean_T,args.conditional_var)
        a_C,b_C = mean_var_to_alpha_beta(mean_C,args.conditional_var)
        T = p_T = torch.distributions.Gamma(a_T,b_T).sample()
        C = p_C = torch.distributions.Gamma(a_C,b_C).sample()
        return T,C
       
    T,C = sim_times(Y)
    print("min T",T.min())
    print("max T",T.max())
    print("min C",C.min())
    print("max C",C.max())
    #X = torch.tensor(X)
    return synthetic_samples_to_loaders(args,X,T,C,train_N,val_N)


def make_gamma_loaders(args):

    args.D = 32
    args.N = 10000
    args.conditional_var=0.05
    args.x_var=10.0

    MVN = torch.distributions.MultivariateNormal

    def bad(a):
        return torch.any(torch.isnan(a))


    def x_to_gamma_dist(x):
        weight_scale = 0.1
        w_T = torch.rand(args.D) * weight_scale
        w0_T = torch.rand(1) * weight_scale
        w_C = torch.rand(args.D) * weight_scale
        w0_C = torch.rand(1) * weight_scale
        print("X shape",x.shape)
        print("w_T shape",w_T.shape)
        mean_T = ((x*w_T).sum(-1) + w0_T).exp()
        #mean_C = ((x*w_C).sum(-1) + w0_C).exp() * 0.05
        mean_C = mean_T * 0.9
        print("mean T shape",mean_T.shape)
        a_T,b_T = mean_var_to_alpha_beta(mean_T,args.conditional_var)
        a_C,b_C = mean_var_to_alpha_beta(mean_C,args.conditional_var)
        p_T = torch.distributions.Gamma(a_T,b_T)
        p_C = torch.distributions.Gamma(a_C,b_C)
        return p_T,p_C

    ############################################
    ############### SET SIZES  #################
    #############################################

    train_N = args.N//2
    valid_N = train_N//2
    test_N = valid_N
    N = train_N + valid_N + test_N

    ############################################
    ############### MAKE X  #####################
    #############################################
    x_scale=5.0
    x_scale = torch.sqrt(torch.tensor([args.x_var]))
    x_dist = MVN(loc=torch.zeros(args.D),covariance_matrix=x_scale*torch.eye(args.D))
    X = x_dist.sample(sample_shape=(N,))
    #############################################
    ############### MAKE P(T,C)  ################
    #############################################
    p_T,p_C = x_to_gamma_dist(X)
    T = p_T.sample()
    C = p_C.sample()
    return synthetic_samples_to_loaders(args,X,T,C,train_N,valid_N)








class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, U, Delta, X):
        self.X=X
        self.U=U
        self.Delta=Delta

    def __getitem__(self, index):
        u=self.U[index]
        delta=self.Delta[index]
        x=self.X[index]
        return u,delta,x

    def __len__(self):
        return len(self.U)


def make_real_loaders(args, mode='normal'):
    assert args.dataset in args.realsets
    fname = 'data/' + args.dataset + '.csv'
    trainset, valset, testset = file_to_dataset(fname, args)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.train_batch_sz,num_workers=args.num_workers,shuffle=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=args.test_batch_sz,num_workers=args.num_workers,shuffle=False)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args.test_batch_sz,num_workers=args.num_workers,shuffle=False)
    return trainloader,valloader,testloader


def tensors_to_dataset(X, U, Delta, phase, args):
    N_full = X.shape[0]
    if phase == 'train':
        N = args.N_train
    elif phase in ['val', 'test']:
        N = N_full
    X = X.float()
    U = U.long()
    Delta = Delta.bool()
    dataset = SyntheticDataset(U=U[:N], Delta=Delta[:N], X=X[:N])
    return dataset

def file_to_dataset(fname, args):
    K = args.K
    print("fname", fname)
    df = pd.read_csv(fname)

    cols_to_drop = [c for c in df.columns if 'Unnamed' in c]
    df = df.drop(columns=cols_to_drop)

    df_u = df['duration']
    df_delta = df['event']
    df_x = df.drop(columns=['duration', 'event'])
    print("x columns", df_x.columns)
    print(df_u.shape)
    print(df_delta.shape)
    print(df_x.shape)
    u = torch.tensor(df_u.to_numpy())
    delta = torch.tensor(df_delta.to_numpy()).bool()
    x = torch.tensor(df_x.to_numpy()).float()
    assert not torch.any(torch.isnan(u))
    assert not torch.any(torch.isnan(delta))
    assert not torch.any(torch.isnan(x))

    def get_bin_boundaries(times, K):
        percents = np.arange(K + 1) * 100. / K
        bin_boundaries = np.percentile(times, percents)
        return torch.tensor(bin_boundaries)

    t = u[delta]
    c = u[~delta]
    print(get_bin_boundaries(t, K))
    print(get_bin_boundaries(c, K))
    print("t min", t.min())
    print("t max", t.max())
    print("c min", c.min())
    print("c max", c.max())
    boundaries = get_bin_boundaries(t, K)

    def get_bin(time, boundaries):
        original_shape = time.shape
        time = time.unsqueeze(-1)
        boundaries_to_consider = boundaries[1:-1]
        time_cat = (time > boundaries_to_consider)
        time_cat = time_cat.sum(-1)
        assert time_cat.shape == original_shape
        return time_cat

    u = get_bin(u, boundaries)
    print("delta=1: {}/{}".format(delta.sum(), delta.shape[0]))

    N = u.shape[0]
    p = np.random.permutation(N)
    u = u[p]
    delta = delta[p]
    x = x[p]

    N_train = int(0.6 * N)
    N_val = int(0.2 * N)

    TRAIN_SPLIT = N_train
    VAL_SPLIT = N_train + N_val

    utr = u[:TRAIN_SPLIT]
    uva = u[TRAIN_SPLIT:VAL_SPLIT]
    ute = u[VAL_SPLIT:]

    deltatr = delta[:TRAIN_SPLIT]
    deltava = delta[TRAIN_SPLIT:VAL_SPLIT]
    deltate = delta[VAL_SPLIT:]

    xtr = x[:TRAIN_SPLIT]
    xva = x[TRAIN_SPLIT:VAL_SPLIT]
    xte = x[VAL_SPLIT:]

    def standardize(x, mu_train, std_train, eps=1e-4):
        return (x - mu_train) / (std_train + eps)

    mu_tr = xtr.mean(dim=0, keepdim=True)
    std_tr = xtr.std(dim=0, keepdim=True)

    xtr = standardize(xtr, mu_tr, std_tr)
    xva = standardize(xva, mu_tr, std_tr)
    xte = standardize(xte, mu_tr, std_tr)
    trainset = tensors_to_dataset(xtr, utr, deltatr, 'train', args)
    valset = tensors_to_dataset(xva, uva, deltava, 'val', args)
    testset = tensors_to_dataset(xte, ute, deltate, 'test', args)
    return trainset, valset, testset

