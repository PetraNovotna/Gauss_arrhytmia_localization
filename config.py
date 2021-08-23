import os
import numpy as np

class Config:
    

    model_save_dir = "../tmp"



    if os.path.isdir('../data'):
        DATA_PATH = "../data/Training_WFDB"
        lbls_path='../data/output_labeled'
    elif os.path.isdir('../../cinc2021_petka_data'):
        DATA_PATH = "../../cinc2021_petka_data/Training_WFDB"
        lbls_path='../../cinc2021_petka_data/output_labeled'
    else:
        raise Exception('no data')
        
    

    mil_solution = 'max'
    
    gaussian_sigma = 30   
    # gaussian_sigma = 'mil'
    
    
    
    pato_use = ['Normal', 'PVC', 'PAC'] 
    # pato_use = ['Normal', 'PVC']
    # pato_use = ['Normal', 'PAC'] 
    # pato_use = None
    
    
    
    pato_use_for_prediction_real = ['PAC','PVC']
    # pato_use_for_prediction_real = ['PVC']
    # pato_use_for_prediction_real = ['PAC']
    
    
    
    DATA_TMP_PATH = "../data_resave"
    
    Fs = 150
    
    MODELS_SEED = 42
    SPLIT_RATIO = [7,1,2]
    
    
    # res_dir='../res_detection_gausian' + str(gaussian_sigma)
        

    pato_all = ['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    pato_use_for_prediction = ['PAC','PVC']
    
 
    ABB2IDX_MAP = {key: idx for idx, key in enumerate(pato_all)}
  

    # train_batch_size = 8
    # valid_batch_size = 8

    train_batch_size = 32
    valid_batch_size = 32
    
    valid_num_workers = 6
    train_num_workers = 6

    # valid_num_workers = 0
    # train_num_workers = 0

    
    LR_LIST=np.array([0.001,0.0001,0.00001])
    LR_CHANGES_LIST=[60,30,15]
    # LR_CHANGES_LIST=[10,5,2]
        
        
    max_epochs=np.sum(LR_CHANGES_LIST)


    model_note = 'xxx'



    ## network setting
    levels = 4
    # lvl1_size = 8
    lvl1_size = 32
    input_size = 12
    output_size = 2
    convs_in_layer = 5
    # convs_in_layer = 2
    init_conv = lvl1_size
    filter_size = 5
    # filter_size = 3


