import shutil
import os
import logging

from config import Config
from resave_data import resave_data
from train import train

if __name__ == "__main__":
    
    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
        
        
    # if True:
        
        config = Config()
        
        
        if os.path.isdir(config.DATA_TMP_PATH):
            shutil.rmtree(config.DATA_TMP_PATH)
            
        resave_data(config)
        
        
        for k in range(0,8):
     
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
            
            
            
            for gaussian_sigma in [10,20,30,40,50,60,'mil']:
                
            # for gaussian_sigma in ['mil']:   
            
                config.gaussian_sigma = gaussian_sigma   
                
                
                train(config)

    
    
    except Exception as e:
        logging.critical(e, exc_info=True)






