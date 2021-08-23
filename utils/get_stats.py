from torch.utils import data 
from collections import namedtuple
import numpy as np
import os

from utils.transforms import SnomedToOneHot


def get_stats(filenames, config):

    snomedToOneHot = SnomedToOneHot()
    
    names_onehot_lens = []
    
    Data = namedtuple('Data','name onehot len')
    
    for name in filenames:
        
        
        head, tail = os.path.split(name.replace('.npy',''))
        _,abbs,len_ = tail.split('--')
        
        abbs = abbs.split('_')
        
        if config.pato_use:
            if not len(set.intersection(set(abbs),set(config.pato_use)))>0:
                continue
                
                
                
        
        onehot = snomedToOneHot(abbs,config.ABB2IDX_MAP).astype(np.int)
        
        onehot = onehot[[config.pato_all.index(pato) for pato in config.pato_use_for_prediction]]
        
        len_ = int(len_)
        
        names_onehot_lens.append(Data(name,onehot,len_))
    
    return names_onehot_lens




    
    
    
    