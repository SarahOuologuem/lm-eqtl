import pickle
import time
import builtins
import sys
import torch
import os
import numpy as np

bases = 'RA'

class EMA():
    '''
    Exponential moving average
    '''
    def __init__(self, beta = 0.98):
        
        self.beta = beta
        self.itr_idx = 0
        self.average_value = 0
        
    def update(self, value):
        self.itr_idx += 1
        self.average_value = self.beta * self.average_value + (1-self.beta)*value
        smoothed_average = self.average_value / (1 - self.beta**(self.itr_idx))
        return smoothed_average
    
class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def list2range(v):
    r = []
    for num in v:
        if not ':' in num:
            r.append(int(num))
        else:
            k = [int(x) for x in num.split(':')]
            if len(k)==2:
                r.extend(list(range(k[0],k[1]+1)))
            else:
                r.extend(list(range(k[0],k[1]+1,k[2])))
    return r

def print_class_recall(class_recall, title):
    s = [f'{k}={v:.3}' for k,v in zip(list(bases),class_recall)]
    s.append(f'AVG={np.mean(class_recall):.3}')
    return title + ';'.join(s)

def print(*args, **kwargs):
    '''
    Redefine print function for logging
    '''
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #current date and time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()

def save_model_weights(model, optimizer, output_dir, epoch):
    '''
    Save model and optimizer weights
    '''

    config_save_base = os.path.join(output_dir, f'epoch_{epoch}_weights')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f'EPOCH:{epoch}: SAVING MODEL, CONFIG_BASE: {config_save_base}\n')

    torch.save(model.state_dict(), config_save_base+'_model.pt') #save model weights

    torch.save(optimizer.state_dict(), config_save_base+'_optimizer.pt') #save optimizer weights
    
