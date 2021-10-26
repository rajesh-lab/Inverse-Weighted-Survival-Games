import models
import os
import copy
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter as KMFitter
import pycox
import numpy as np


class ModelSaver(object):
    
    def __init__(self, args,is_g=False):
        super(ModelSaver, self).__init__()
        self.save_dir = args.save_dir
        self.ckpt_basename = args.ckpt_basename
        self.is_g = is_g
        self.best_metric_val = None
        self.minimize=True
        self.best_epoch = 0
        self.args = args
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        dirr = os.path.join(self.save_dir,self.ckpt_basename)
        from pathlib import Path
        Path(dirr).mkdir(parents=True, exist_ok=True)
        

        suffix = '_G' if is_g else '_F'
        
        self.best_path = os.path.join(dirr,
                                self.ckpt_basename+suffix+'_best.pth.tar')
        
        self.epoch_path = os.path.join(dirr,
                                self.ckpt_basename+suffix+'_epoch{}.pth.tar')

        print("Saver")
        print("Is g {}".format(is_g))
        print("Best path",self.best_path)
        print("Epoch path",self.epoch_path)


    def _is_best(self, metric_val):
        if metric_val is None:
            return False
        return self.best_metric_val is None or metric_val < self.best_metric_val
   

    def always_save(self, epoch, model): 
        pth = self.epoch_path.format(epoch)
        #print("Saving {}".format(pth))
        model_copy = copy.deepcopy(model)
        ckpt_dict = {'model_state':model_copy.to('cpu').state_dict()}
        torch.save(ckpt_dict, pth)
        return True


    def maybe_save(self, epoch, model, metric_val):
        if self._is_best(metric_val):
            #print("Saving {}".format(self.best_path))
            self.best_metric_val = metric_val
            self.best_epoch = epoch
            model_copy = copy.deepcopy(model)
            ckpt_dict = {'model_state':model_copy.to('cpu').state_dict()}
            torch.save(ckpt_dict, self.best_path)
            return True
        else:
            return False


    def load_best(self):
        print("Loading Model From Epoch {} From File {}".format(self.best_epoch,self.best_path))
        ckpt_dict = torch.load(self.best_path, map_location=self.args.device)
        model = self.args.model_fn(self.args)
        model.load_state_dict(ckpt_dict['model_state'])
        return model

