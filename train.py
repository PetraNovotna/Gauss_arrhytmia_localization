import os
import scipy.io as io
import torch
from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch.nn.functional as F
from shutil import copyfile

from config import Config
from dataloader import Dataset
from utils.losses import get_lr
import net
from utils.get_stats import get_stats
from utils.adjustLearningRateAndLoss import AdjustLearningRateAndLoss
from utils.log import Log
from utils.losses import wce,mse
from utils.train_valid_test_split import train_valid_test_split
from evaluate import evaluate



def train(config):
    
    
    
    device = torch.device("cuda:0")

    
    
    file_list = glob.glob(config.DATA_TMP_PATH + "/*.npy")

    
    names_onehot_lens = get_stats(file_list,config)
    
    

    names_onehot_lens_train,names_onehot_lens_valid,names_onehot_lens_test = train_valid_test_split(names_onehot_lens,config.MODELS_SEED,config.SPLIT_RATIO)
    
    
    
    num_of_sigs = len(names_onehot_lens_train)
    lbl_counts = sum(map(lambda x: x.onehot,names_onehot_lens_train))
    w_positive = num_of_sigs / lbl_counts
    w_negative = num_of_sigs / (num_of_sigs - lbl_counts)
    w_positive_tensor = torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor = torch.from_numpy(w_negative.astype(np.float32)).to(device)
    
    
    


    training_generator = Dataset(names_onehot_lens_train, 'train', config)
    training_generator = data.DataLoader(training_generator, batch_size=Config.train_batch_size,
                                         num_workers=Config.train_num_workers, shuffle=True, drop_last=True,
                                         collate_fn=Dataset.collate_fn)

    validation_generator = Dataset(names_onehot_lens_valid, 'valid', config)
    validation_generator = data.DataLoader(validation_generator, batch_size=Config.valid_batch_size,
                                           num_workers=Config.valid_num_workers, shuffle=True, drop_last=False,
                                           collate_fn=Dataset.collate_fn)


    model = net.Net_addition_grow(levels=config.levels, lvl1_size=config.lvl1_size, input_size=config.input_size,
                              output_size=len(config.pato_use_for_prediction_real),
                              convs_in_layer=config.convs_in_layer, init_conv=config.init_conv,
                              filter_size=config.filter_size,mil_solution=config.mil_solution)


    model = model.to(device)
    
    train_names = [item.name for item in names_onehot_lens_train]
    valid_names = [item.name for item in names_onehot_lens_valid]
    test_names = [item.name for item in names_onehot_lens_test]
    model.train_names = train_names
    model.valid_names = valid_names
    model.test_names = test_names
    
    

    ## create optimizer and learning rate scheduler to change learnng rate after
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR_LIST[0], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    scheduler=AdjustLearningRateAndLoss(optimizer,Config.LR_LIST,Config.LR_CHANGES_LIST,[None,None,None,None,None])


    ## create empty log - object to save training results
    log = Log(['loss'])

    for epoch in range(Config.max_epochs):

        N=len(training_generator)
        # change model to training mode
        model.train()
        for it,(pad_seqs, lens, lbls,file_names,detection)  in enumerate(training_generator):
            
            if it%20==0:
                print(str(it) + '/' + str(N))
                
                
            ind_pato = [config.pato_use_for_prediction.index(x) for x in config.pato_use_for_prediction_real]
            lbls = lbls[:,ind_pato] 
            detection = detection[:,ind_pato,:] 

                
            ## send data to graphic card
            pad_seqs, lens, lbls,detection  = pad_seqs.to(device), lens.to(device), lbls.to(device), detection.to(device)

            ## aply model
            res, heatmap, score,detection_subsampled = model(pad_seqs, lens,detection)

            
            ## calculate loss
            if config.gaussian_sigma == 'mil':
                
                loss=wce(res, lbls, w_positive_tensor, w_negative_tensor)
            else:
                loss=mse(heatmap, detection_subsampled)

            ## update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            ## save results
            log.append_train([loss])



        N=len(validation_generator)
        ## validation mode - "disable" batch norm
        model.eval()
        for it,(pad_seqs, lens, lbls,file_names,detection)  in enumerate(validation_generator):
            
            if it%20==0:
                print(str(it) + '/' + str(N))
            
            ind_pato = [config.pato_use_for_prediction.index(x) for x in config.pato_use_for_prediction_real]
            lbls = lbls[:,ind_pato] 
            detection = detection[:,ind_pato,:] 
            
            
            pad_seqs, lens, lbls,detection  = pad_seqs.to(device), lens.to(device), lbls.to(device), detection.to(device)

            res, heatmap, score,detection_subsampled = model(pad_seqs, lens,detection)

            
            ## calculate loss
            if config.gaussian_sigma == 'mil':
                
                loss=wce(res, lbls, w_positive_tensor, w_negative_tensor)
            else:
                loss=mse(heatmap, detection_subsampled)

            
            loss = loss.detach().cpu().numpy()
            res = res.detach().cpu().numpy()
            lbls = lbls.detach().cpu().numpy()
            detection_subsampled = detection_subsampled.detach().cpu().numpy()
            heatmap = heatmap.detach().cpu().numpy()
            pad_seqs = pad_seqs.detach().cpu().numpy()

            ## save results
            log.append_valid([loss])
            



        plt.plot(pad_seqs[0,0,:int(np.floor(lens.detach().cpu().numpy()[0]))])
        plt.title('signal')
        plt.show()


        plt.plot(heatmap[0,0,:int(np.floor(lens.detach().cpu().numpy()[0]/(2**config.levels)))])
        if not config.gaussian_sigma == 'mil':
            plt.ylim(-0.05,1)
        plt.title('result')
        plt.show()
        
        plt.plot(detection_subsampled[0,0,:int(np.floor(lens.detach().cpu().numpy()[0]/(2**config.levels)))])
        plt.ylim(-0.05,1)
        if not config.gaussian_sigma == 'mil':
            plt.title('gt')
        plt.show()
        
        
        log.save_and_reset()

        lr = get_lr(optimizer)

        info = str(epoch) + '_' + str(lr) + '_train_' + str(log.train_log['loss'][-1]) + '_valid_' + str(log.valid_log['loss'][-1])
        print(info)

        model_name = Config.model_save_dir + os.sep + Config.model_note + info + '.pt'
        log.save_log_model_name(model_name)
        model.save_log(log)
        model.save_config(Config)
        torch.save(model, model_name)

        if not config.gaussian_sigma == 'mil':
            log.plot(model_name,ylim=[0,0.01])
        else:
            log.plot(model_name)
        
        scheduler.step()
        
        
    best_model_name=log.model_names[np.argmax(log.valid_log['loss'])]
        
    copyfile(best_model_name, '../final_model.pt')
    
    evaluate(model)
    
    
    
    
    
    



if __name__ == "__main__":
    
    config = Config()
        
    train(config)
