from torch.utils import data
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter


class Dataset(data.Dataset):

    def __init__(self, names_onehot_lens, split, config):
        """Initialization"""
        self.names_onehot_lens = names_onehot_lens
        self.split = split
        self.config = config

    def __len__(self):
        """Return total number of data samples"""
        return len(self.names_onehot_lens)

    def __getitem__(self, idx):
        """Generate data sample"""
        # Select sample
        name_onehot_len = self.names_onehot_lens[idx]

        # Read data and get label
        X = np.load(name_onehot_len.name)
            
        sig_len = X.shape[1]
        signal_num = X.shape[0]
        

        y  = name_onehot_len.onehot
        
        
        file_name = name_onehot_len.name
    

        with open(name_onehot_len.name.replace('.npy','.json'), 'r') as file:
            positions_resampled = json.load( file)

        lbl_PAC = np.array(positions_resampled['PAC']).astype(np.int)
        lbl_PVC = np.array(positions_resampled['PVC']).astype(np.int)

        Y_PAC=np.zeros((sig_len))
        
        Y_PAC[lbl_PAC]=1
        
        Y_PAC = gaussian_filter(Y_PAC,self.config.gaussian_sigma,mode='constant')/(1/(self.config.gaussian_sigma*np.sqrt(2*np.pi)))
        Y_PAC = Y_PAC.reshape((1,len(Y_PAC))).astype(np.float32)
        
        
        
        Y_PVC=np.zeros((sig_len))
        
        Y_PVC[lbl_PVC]=1
        
        Y_PVC = gaussian_filter(Y_PVC,self.config.gaussian_sigma,mode='constant')/(1/(self.config.gaussian_sigma*np.sqrt(2*np.pi)))
        Y_PVC = Y_PVC.reshape((1,len(Y_PVC))).astype(np.float32)
        
        
        Y = np.concatenate((Y_PAC,Y_PVC),axis = 0)
        lbl_num = Y.shape[0]
        



        ##augmentation
        if self.split == 'train':
            ##random circshift
            if torch.rand(1).numpy()[0] > 0.3:
                shift = torch.randint(sig_len, (1, 1)).view(-1).numpy()

                X = np.roll(X, shift, axis=1)
                Y = np.roll(Y, shift, axis=1)


            ## random stretch -
            if torch.rand(1).numpy()[0] > 0.3:

                max_resize_change = 0.2
                relative_change = 1 + torch.rand(1).numpy()[0] * 2 * max_resize_change - max_resize_change
                ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                if relative_change<1:
                    relative_change=1/(1-relative_change+1)
                
                new_len = int(relative_change * sig_len)

                XX = np.zeros((signal_num, new_len))
                for k in range(signal_num):
                    XX[k, :] = np.interp(np.linspace(0, sig_len - 1, new_len), np.linspace(0, sig_len - 1, sig_len),
                                        X[k, :])
                X = XX
                
                
                YY= np.zeros((lbl_num, new_len))
                for k in range(lbl_num):
                    YY[k, :] = np.interp(np.linspace(0, sig_len - 1, new_len), np.linspace(0, sig_len - 1, sig_len),
                                        Y[k, :])
                    
                Y=YY
                

            ## random multiplication of each lead by a number
            if torch.rand(1).numpy()[0] > 0.3:

                max_mult_change = 0.4

                for k in range(signal_num):
                    mult_change = 1 + torch.rand(1).numpy()[0] * 2 * max_mult_change - max_mult_change
                    ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                    if mult_change<1:
                        mult_change=1/(1-mult_change+1)
                        
                    X[k, :] = X[k, :] * mult_change



        return X, y,file_name,Y

    def collate_fn(data):
        ## this take list of samples and put them into batch

        ##pad with zeros
        pad_val = 0

        ## get list of singals and its lengths
        seqs, lbls, file_names,lbls_seqs = zip(*data)

        lens = [seq.shape[1] for seq in seqs]

        ## pad shorter signals with zeros to make them same length
        padded_seqs = pad_val * np.ones((len(seqs), seqs[0].shape[0], np.max(lens))).astype(np.float32)
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :, :end] = seq
            
        padded_lbls_seqs = pad_val * np.ones((len(lbls_seqs), lbls_seqs[0].shape[0], np.max(lens))).astype(np.float32)
        for i, seq in enumerate(lbls_seqs):
            end = lens[i]
            padded_lbls_seqs[i, :, :end] = seq    

        ## stack and reahape signal lengts to 10 vector
        lbls = np.stack(lbls, axis=0)
        lbls = lbls.reshape(lbls.shape[0:2])
        lens = np.array(lens).astype(np.float32)

        ## numpy -> torch tensor
        padded_seqs = torch.from_numpy(padded_seqs)
        lbls = torch.from_numpy(lbls)
        lens = torch.from_numpy(lens)
        padded_lbls_seqs = torch.from_numpy(padded_lbls_seqs)

        return padded_seqs, lens, lbls,file_names,padded_lbls_seqs


def main():
    return Dataset


if __name__ == "__main__":
    main()