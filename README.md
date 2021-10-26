# Inverse-Weighted-Survival-Games

This repo contains the code for paper Inverse Weighted Survival Games

# instructions


## general
- loss function (--lfn) can be in ['nll','bs_game','bll_game']
- model checkpoints will be saved in 'ckpts' dir by default
- pkls with metrics will be saved in 'results' dir by default
- ckpt_basename will be used in filenames in these two directories
- dset in 'gamma', 'mnist', 'support', 'gbsg', 'metabric'
- For your own data, you can follow the data format in 'data/support.csv' 

## gamma

python main.py --N_train 500 --N_val 1024 --N_test 2048 --dropout_rate 0.00 --epochs 300 --train_batch_sz 1000 --test_batch_sz 1000 --seed 1 --dataset gamma --loss_fn nll --save_dir ckpts --ckpt_basename tmp

## mnist

python main.py --N_train 1024 --N_val 1024 --N_test 2048 --dropout_rate 0.00 --epochs 300 --train_batch_sz 512 --test_batch_sz 512 --seed 1 --dataset mnist --loss_fn nll --save_dir ckpts --ckpt_basename tmp

## real data 

python main.py --N_train 100 --N_val 300 --N_test 8000 --dropout_rate 0.00 --epochs 200 --train_batch_sz 150 --test_batch_sz 150 --seed 1 --dataset metabric --loss_fn nll  --save_dir ckpts --ckpt_basename tmp
