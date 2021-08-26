import glob
import numpy as np
from bayes_opt import BayesianOptimization
from scipy.signal import find_peaks
from torch.utils import data
import torch
import matplotlib.pyplot as plt
import os

from dataloader import Dataset
from utils.get_results import get_results
from utils.get_stats import get_stats



def evaluate(model,res_dir):
    


    device = torch.device("cuda:0")
    
    config = model.config
    
    for ind_pato in range(len(config.pato_use_for_prediction_real)):
        name = res_dir + '/'+ config.pato_use_for_prediction_real[ind_pato]
        if not os.path.isdir(name):
            os.makedirs(name)

    
    names_test = model.names_test
    
    names_onehot_lens_test = get_stats(names_test,config)
    
    
    test_generator = Dataset(names_onehot_lens_test, 'valid', config, get_positions=True)
    test_generator = data.DataLoader(test_generator, batch_size=config.valid_batch_size,
                                           num_workers=0, shuffle=False, drop_last=False,
                                           collate_fn=Dataset.collate_fn)
    
    
    lblss, ress, detection_gts_paralel, heatmaps_paralel, signals, names, positions_gt = [], [], [], [], [], [], []
    
    with torch.no_grad():
        for it,(pad_seqs, lens, lbls,file_names,detection,positions)  in enumerate(test_generator):
                

            ind_pato = [config.pato_use_for_prediction.index(x) for x in config.pato_use_for_prediction_real]
            lbls = lbls[:,ind_pato] 
            detection = detection[:,ind_pato,:] 
            
            
            
            pad_seqs, lens, lbls,detection  = pad_seqs.to(device), lens.to(device), lbls.to(device), detection.to(device)

            res, heatmap, score,detection_subsampled = model(pad_seqs, lens,detection)


            res = res.detach().cpu().numpy()
            lbls = lbls.detach().cpu().numpy()
            detection_subsampled = detection_subsampled.detach().cpu().numpy()
            heatmap = heatmap.detach().cpu().numpy()
            pad_seqs = pad_seqs.detach().cpu().numpy()
            lens = lens.detach().cpu().numpy()
            detection = detection.detach().cpu().numpy()



            for hm_index in range(heatmap.shape[0]):
                heatmap0 = heatmap[hm_index, :, :]
                pad_seqs0 = pad_seqs[hm_index,:,:]
                detection0 = detection[hm_index,:,:]
                positions0 = positions[hm_index]

                len_short = int(np.floor(lens[hm_index]/(2**config.levels))*(2**config.levels))

                heatmap0 = heatmap0[:,:int(np.floor(lens[hm_index]/(2**config.levels)))]
                
                detection_gt = detection0[:,:int(lens[hm_index])]
                
                signal0 = pad_seqs0[:,:int(lens[hm_index])]
                
                N=heatmap0.shape[1]
                heatmap0_res=[]
                for k in range(heatmap0.shape[0]):
                    tmp=np.zeros(int(lens[hm_index]))
                    tmp[:len_short]=np.interp(np.linspace(0, N - 1, len_short),np.linspace(0, N - 1, N), heatmap0[k,:])
                    heatmap0_res.append(tmp)
                heatmap0_res=np.stack(heatmap0_res,0)
                
                res0 = res[hm_index,:]
                lbls0 = lbls[hm_index,:]

                lblss.append(lbls0)
                ress.append(res0)
                detection_gts_paralel.append(detection_gt)
                heatmaps_paralel.append(heatmap0_res)
                signals.append(signal0)
                names.append(file_names[hm_index])
                positions_gt.append(positions0)
                
                
            
                
                
                
    height_max = np.max([np.max(heatmap) for heatmap in heatmaps_paralel])
    height_min = np.min([np.min(heatmap) for heatmap in heatmaps_paralel])
    
    
    def get_peaks(heatmaps,height,distance,prominence):
        detections=[]
        for k,heatmap in enumerate(heatmaps):
            
            peaks,properties=find_peaks(heatmap,height=height,distance=distance,prominence=prominence) 
            detections.append(peaks)
        
        return detections
    
    
    detection_gts_tmp = []
    heatmaps = []
    for heatmap_ind in range(heatmaps_paralel[0].shape[0]):
        
        detection_gts_tmp.append([x[heatmap_ind,:] for x in detection_gts_paralel])
        heatmaps.append([x[heatmap_ind,:] for x in heatmaps_paralel])
    
    gt_detections = []
    gt_detections2 = []
    for heatmap_ind in range(len(heatmaps)):
        gt_detections2.append(get_peaks(detection_gts_tmp[heatmap_ind],0.8,20,0.1))
        
        pato = config.pato_use_for_prediction_real[heatmap_ind]
        
        gt_detections.append([np.array(x[pato]).astype(np.int) for x in positions_gt])
        
    
    
    
    recall, precision, dice, acc, TP, FP, FN, params = [], [], [] ,[], [], [] ,[], []
    
    for heatmap_ind in range(len(heatmaps)):
    
        
        def func(all_results=False,**params):
            
            detections_tmp = get_peaks(heatmaps[heatmap_ind],params['height'],params['distance'],params['prominence'])
            
            recall, precision, dice, acc, TP, FP, FN = get_results(gt_detections[heatmap_ind],detections_tmp)
                
                
            
            if all_results:
                return recall, precision, dice, acc, TP, FP, FN
            else :
                return dice
        
        
        
        
        
        pbounds = {'height':[height_min,height_max],'distance':[1,4*config.Fs],'prominence':[0,height_max-height_min]}
    
        optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
        
        
        # optimizer.maximize(init_points=200,n_iter=200)
        optimizer.maximize(init_points=200,n_iter=100)
        
        print(optimizer.max)
        
        params_tmp=optimizer.max['params']
        
        
        recall_tmp, precision_tmp, dice_tmp, acc_tmp, TP_tmp, FP_tmp, FN_tmp = func(all_results=True,**params_tmp)
        
        recall.append(recall_tmp)
        precision.append(precision_tmp)
        dice.append(dice_tmp) 
        acc.append(acc_tmp)
        TP.append(TP_tmp)
        FP.append(FP_tmp)
        FN.append(FN_tmp)
        params.append(params_tmp)
        
        
    
    
    for file_num in range(len(heatmaps[0])):
        
        for ind_pato in range(heatmap.shape[1]):
        
            params_tmp = params[ind_pato]
        
            detections_tmp = get_peaks([heatmaps[ind_pato][file_num]],params_tmp['height'],params_tmp['distance'],params_tmp['prominence'])
            
            detections_tmp_gt = gt_detections[ind_pato][file_num]
            
        
            name = res_dir + '/'+ config.pato_use_for_prediction_real[ind_pato] + '/' + os.path.split(names[file_num])[1].replace('.npy','') 
            
            plt.plot(signals[file_num][0,:])
            plt.title('signal')
            plt.savefig(name + '_signal.png') 
            # plt.show()
            plt.close()
            tmp = signals[file_num][0,:]
            np.save(name + '_signal.npy',tmp)
            
            
            
    
            plt.plot(heatmaps[ind_pato][file_num])
            if len(detections_tmp)>0:
                plt.plot(detections_tmp[0],np.zeros(detections_tmp[0].shape),'r*')
            if not config.gaussian_sigma == 'mil':
                plt.ylim(-0.05,1)
            plt.title('result')
            plt.savefig(name + '_result.png') 
            # plt.show()
            plt.close()
            tmp = heatmaps[ind_pato][file_num]
            np.save(name + '_result.npy',tmp)
            
            
            plt.plot(detection_gts_tmp[ind_pato][file_num])
            if len(detections_tmp)>0:
                plt.plot(detections_tmp_gt,np.zeros(detections_tmp_gt.shape),'r*')
            plt.title('gt')
            plt.savefig(name + '_gt_detection.png') 
            # plt.show()
            plt.close()
            tmp = detection_gts_tmp[ind_pato][file_num]
            np.save(name + '_gt_detection.npy',tmp)
    
    
    print('recall ' + str(recall))
    print('precision ' + str(precision))
    print('dice ' + str(dice))
    print('acc ' + str(acc))
    print('TP ' + str(TP))
    print('FP ' + str(FP))
    print('FN ' + str(FN))
    print('params ' + str(params))
    
    
    return recall, precision, dice, acc, TP, FP, FN, params



if __name__ == "__main__":

    device = torch.device("cuda:0")    

    model = torch.load('../finalmodel_Normal-PVC-PAC_PAC-PVC_mil_max.pt',map_location=device)

    recall, precision, dice, acc, TP, FP, FN, params = evaluate(model,'../results_Normal-PVC-PAC_PAC-PVC_mil')
    
    
    
    
    
    
    


