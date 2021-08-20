from glob import glob
import wfdb
import numpy as np
import os
import scipy.io as io
import json

from utils import transforms



def read_lbl_pos(file_name):
    
    data_dict = io.loadmat(file_name)
    
    lbl_PAC,lbl_PVC = data_dict["result_positions_PAC"].flatten(),data_dict["result_positions_PVC"].flatten()
    
    return lbl_PAC,lbl_PVC


def resave_one(filename,src_path,dst_path,config):
    
     
    resampler = transforms.Resample(output_sampling=config.Fs)
    remover_50_100_150_60_120_180Hz = transforms.Remover_50_100_150_60_120_180Hz()
    baseLineFilter = transforms.BaseLineFilter()
    
    
    signal,fields = wfdb.io.rdsamp(filename)
    
    
    signal = signal.T.astype(np.float32)
    
    
    head,tail = os.path.split(filename)
    lbl_file_name = config.lbls_path + os.sep + tail + '_position_labels.mat'
    lbl_PAC,lbl_PVC = read_lbl_pos(lbl_file_name)
    
    
    positions_resampled = dict()
    positions_resampled['PAC'] = np.round(lbl_PAC * (config.Fs / fields['fs'])).tolist()
    positions_resampled['PVC'] = np.round(lbl_PVC * (config.Fs / fields['fs'])).tolist()
    
    

    
    
    signal = remover_50_100_150_60_120_180Hz(signal,input_sampling=fields['fs'])
    signal = resampler(signal,input_sampling=fields['fs'])
    signal = baseLineFilter(signal)
    
    
    Dxs = [sub for sub in fields['comments'] if 'Dx: ' in sub][0].replace('Dx: ','').split(',')
    
    
    for patology in ['PAC','PVC']:
        if positions_resampled[patology]:
            if patology not in Dxs:
                Dxs.append(patology)
        else:
            if patology in Dxs:
                Dxs.remove(patology)
        
        
        
    
    Dxs_string = '_'.join(Dxs)
    
    filename_save = filename.replace(src_path,dst_path) + '--' + Dxs_string + '--' + str(signal.shape[1]) + '.npy'
    
    np.save(filename_save,signal)
    
    
    filename_save_lbl = filename.replace(src_path,dst_path) + '--' + Dxs_string + '--' + str(signal.shape[1]) + '.json'
    
    with open(filename_save_lbl, 'w') as outfile:
        json.dump(positions_resampled, outfile)
    



def resave_data(config):

    src_path = config.DATA_PATH
    dst_path = config.DATA_TMP_PATH

    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)


    filenames  = [name.replace('.mat','') for name in glob(src_path + r"/**/*.mat", recursive=True)]




    for filename in (filenames):
        resave_one(filename,src_path,dst_path,config)
        
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    
    from config import Config
    import shutil
    
    config = Config()
        
        
    if os.path.isdir(config.DATA_TMP_PATH):
        shutil.rmtree(config.DATA_TMP_PATH)
        
    resave_data(config)