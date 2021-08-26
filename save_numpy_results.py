import shutil
import os
import logging
import sys
import torch
import json

from config import Config
from resave_data import resave_data
from train import train
from evaluate import evaluate

if __name__ == "__main__":
    
    # logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
        
        
    if True:
        
        config = Config()
        
        if len(sys.argv)>1:
            config.model_save_dir = sys.argv[1] + '/tmp'
            config.results_dir = sys.argv[1]
        
        
        # if os.path.isdir(config.DATA_TMP_PATH):
        #     shutil.rmtree(config.DATA_TMP_PATH)
            
        # resave_data(config)
        
        
        # for k in range(0,8):
        k = 0
            
            
     
        if k == 0:
            config.pato_use = ['Normal', 'PVC', 'PAC'] 
            config.pato_use_for_prediction_real = ['PAC','PVC']
        
        elif k == 1:
            config.pato_use = ['Normal', 'PVC', 'PAC']
            config.pato_use_for_prediction_real = ['PAC']
        
        elif k == 2:
            config.pato_use = ['Normal', 'PVC', 'PAC']
            config.pato_use_for_prediction_real = ['PVC']
        
        elif k == 3:
            config.pato_use = ['Normal', 'PVC']
            config.pato_use_for_prediction_real = ['PVC']
            
        elif k == 4:
            config.pato_use = ['Normal', 'PAC']
            config.pato_use_for_prediction_real = ['PAC']    
            
        elif k == 5:
            config.pato_use = ['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
            config.pato_use_for_prediction_real = ['PAC','PVC']  
            
            
        elif k == 6:
            config.pato_use = ['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
            config.pato_use_for_prediction_real = ['PVC']     
            
            
        elif k == 7:
            config.pato_use = ['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
            config.pato_use_for_prediction_real = ['PAC']
            
        else:
            
            raise Exception('no settings')
        
        
        if not os.path.isdir(config.DATA_TMP_PATH):
            os.makedirs(config.DATA_TMP_PATH)
            
        if not os.path.isdir(config.model_save_dir):
            os.makedirs(config.model_save_dir)
            
        if not os.path.isdir(config.DATA_TMP_PATH):
            os.makedirs(config.DATA_TMP_PATH)
        
        
        
        for gaussian_sigma in [20,30,40,'max','att1-nolenmul','att2-nolenmul']:
        # for gaussian_sigma in ['att2-lenmul']:
            
            config.gaussian_sigma = gaussian_sigma   
            config.mil_solution = 'gauss'
            
            if config.gaussian_sigma == 'max':
                config.mil_solution = 'max'
                config.gaussian_sigma = 'mil'
            if config.gaussian_sigma == 'att1-nolenmul':
                config.mil_solution = 'att1-nolenmul'
                config.gaussian_sigma = 'mil' 
            if config.gaussian_sigma == 'att2-nolenmul':
                config.mil_solution = 'att2-nolenmul'
                config.gaussian_sigma = 'mil'
            if config.gaussian_sigma == 'att1-lenmul':
                config.mil_solution = 'att1-lenmul'
                config.gaussian_sigma = 'mil' 
            if config.gaussian_sigma == 'att2-lenmul':
                config.mil_solution = 'att2-lenmul'
                config.gaussian_sigma = 'mil'
            
            # train(config)
            
            
            tmp_name = '-'.join(config.pato_use) + '_' + '-'.join(config.pato_use_for_prediction_real) + '_' + str(config.gaussian_sigma) + '_' + config.mil_solution
 
            
            device = torch.device("cuda:0")    
            final_model_name = config.results_dir + '/finalmodel_' + tmp_name +  '.pt'
            
            model = torch.load(final_model_name,map_location=device)
    
 
            res_dir =  config.results_dir + '/results_np_' + tmp_name
            recall, precision, dice, acc, TP, FP, FN, params = evaluate(model,res_dir)
            
            tmp = {'recall_' +  '-'.join(config.pato_use_for_prediction_real) :recall, 'precision':precision, 'dice':dice,
           'acc':acc, 'TP':TP, 'FP':FP, 'FN':FN, 'final_model_name':final_model_name, 'params':params}

            with open(config.results_dir + '/results_' + tmp_name +  '.json', 'w') as outfile:
                json.dump(tmp, outfile)
            
            

    
    
    # except Exception as e:
    #     logging.critical(e, exc_info=True)






