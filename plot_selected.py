
import numpy as np
import matplotlib.pyplot as plt




font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


save_folder = r'C:\Users\vicar\OneDrive - Vysoké učení technické v Brně\cinc_imgs\plots_export'





name = '../na_vyber_ukazky2/results_np_Normal-PVC-PAC_PAC-PVC_30_gauss/PAC/A0097--PAC--2101_gt_detection.npy'
crop = np.arange(0,1350)
save_name = 'pac'


# name = '../na_vyber_ukazky2/results_np_Normal-PVC-PAC_PAC-PVC_30_gauss/PVC/A0052--PVC--1500_gt_detection.npy'
# crop = np.arange(0,1350)
# save_name = 'pvc'



name = '../na_vyber_ukazky2/results_np_Normal-PVC-PAC_PAC-PVC_30_gauss/PVC/A0454--PVC--3029_gt_detection.npy'
crop = np.arange(0,1350)
save_name = 'pvc2'





gt = np.load(name)
gt = gt[crop]


res = np.load(name.replace('_gt_detection.npy','_result.npy'))
res = res[crop]


sig = np.load(name.replace('_gt_detection.npy','_signal.npy'))
sig = sig[crop]



res_max = np.load(name.replace('_gt_detection.npy','_result.npy').replace('results_np_Normal-PVC-PAC_PAC-PVC_30_gauss','results_np_Normal-PVC-PAC_PAC-PVC_mil_max'))
res_max = res_max[crop]




Fs = 150
x = np.arange(0,len(sig)/Fs,1/Fs)


plt.figure(figsize=(15,4))  
plt.plot(x,sig)
plt.ylabel('Lead I (mV)')
plt.xlabel('time (s)')
plt.savefig(save_folder + '/' + save_name + '_signal.svg', transparent=True)
plt.show()    
plt.close()


plt.figure(figsize=(15,4))  
plt.plot(x,gt)
plt.ylim(-0.1,1.1)
plt.xlabel('time (s)')
plt.ylabel('Ground truth')
plt.savefig(save_folder + '/' + save_name + '_gt.svg', transparent=True)
plt.show()    
plt.close()


plt.figure(figsize=(15,4))  
plt.plot(x,res)
plt.ylim(-0.1,1.1)
plt.xlabel('time (s)')
plt.ylabel('Prediction Gaussian')
plt.savefig(save_folder + '/' + save_name + '_res.svg', transparent=True)
plt.show()    
plt.close()


plt.figure(figsize=(15,4))  
plt.plot(x,res_max)
plt.xlabel('time (s)')
plt.ylabel('Prediction max-pooling')
plt.savefig(save_folder + '/' + save_name + '_res_max.svg', transparent=True)
plt.show()    
plt.close()

