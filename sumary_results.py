from config import Config
import json
import numpy as np


config = Config()




results_all = [] 

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
        
        
        
        
    config.results_dir = '../results_meta'
    
    
    results = []
    for gaussian_sigma in [20,30,40,'max','att1-nolenmul','att2-nolenmul','att1-lenmul','att2-lenmul']:
    # for gaussian_sigma in [20,30,40,'max','att1-nolenmul']:

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
    
        tmp_name = '-'.join(config.pato_use) + '_' + '-'.join(config.pato_use_for_prediction_real) + '_' + str(config.gaussian_sigma) + '_' + config.mil_solution
    
        with open(config.results_dir + '/results_' + tmp_name +  '.json', 'r') as outfile:
            data = json.load(outfile)

        # tmp = data['dice']
        # tmp = data['precision']
        tmp = data['recall_' +  '-'.join(config.pato_use_for_prediction_real) ]
        if config.pato_use_for_prediction_real == ['PVC']:
            tmp = ['-',tmp[0]]
        elif config.pato_use_for_prediction_real == ['PAC']:
            tmp = [tmp[0],'-']
        
        
        results.extend(tmp)
        
        
        
        

    results_all.append(results)
    
    


results_all = np.array(results_all)











