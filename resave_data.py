# resave data
from shutil import copyfile, rmtree
import os
from config import Config
import utils.load_fncs as lf




try:
    rmtree(Config.DATA_TMP_PATH)
except:
    pass

try:
    os.mkdir(Config.DATA_TMP_PATH)
except:
    pass


##get all file names
names=[]
for root, dirs, files in os.walk(Config.DATA_PATH):
    for name in files:
        if name.endswith(".mat"):
            name=name.replace('.mat','')
            names.append(name)



for k,file_name in enumerate(names):

    file_name_full = Config.DATA_PATH + os.sep + file_name + '.mat' 
    head,tail = os.path.split(file_name_full)
    lbl_PAC,lbl_PVC = lf.read_lbl_pos(Config.lbls_path + os.sep + tail.replace('.mat','_position_labels.mat'))
    
    lbl = []
    if len(lbl_PAC)>0:
        lbl.append('PAC')
    if len(lbl_PVC)>0:
        lbl.append('PVC')
        

    print(file_name)
    print(lbl)
    print(k)

    
    if False:

        for pato_name in Config.pato_names:
    
          if pato_name==lbl:
            copyfile(Config.DATA_PATH + os.sep +file_name +'.mat',Config.DATA_TMP_PATH + os.sep +file_name +'.mat')
            copyfile(Config.DATA_PATH + os.sep +file_name + '.hea' ,Config.DATA_TMP_PATH + os.sep +file_name + '.hea')
            print("saved")
            
            
    else:
        copyfile(Config.DATA_PATH + os.sep +file_name +'.mat',Config.DATA_TMP_PATH + os.sep +file_name +'.mat')
        copyfile(Config.DATA_PATH + os.sep +file_name + '.hea' ,Config.DATA_TMP_PATH + os.sep +file_name + '.hea')

