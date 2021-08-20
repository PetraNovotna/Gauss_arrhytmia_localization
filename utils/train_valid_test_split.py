import numpy as np 

def train_valid_test_split(names_onehot_lens,seed,split_ratio):
    
    
    
    num_files = len(names_onehot_lens)
    np.random.seed(seed)


    permuted_idx = np.random.permutation(num_files)
    
    split_ind = np.array(split_ratio)
    split_ind = np.floor(np.cumsum(split_ind/np.sum(split_ind)*num_files)).astype(np.int)
    
    
    train_ind = permuted_idx[:split_ind[0]]
    valid_ind = permuted_idx[split_ind[0]:split_ind[1]]         
    test_ind = permuted_idx[split_ind[1]:] 
    
    
    names_onehot_lens_train = [names_onehot_lens[i] for i in train_ind]
    names_onehot_lens_valid = [names_onehot_lens[i] for i in valid_ind]
    names_onehot_lens_test = [names_onehot_lens[i] for i in test_ind]
    
    
    
    
    
    return names_onehot_lens_train,names_onehot_lens_valid,names_onehot_lens_test