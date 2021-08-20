import shutil
import os

from config import Config
from resave_data import resave_data
from train import train

if __name__ == "__main__":
    
    # logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
        
        
    if True:
        
        config = Config()
        
        
        if os.path.isdir(config.DATA_TMP_PATH):
            shutil.rmtree(config.DATA_TMP_PATH)
            
        resave_data(config)
        
        train(config)

    
    
    # except Exception as e:
    #     logging.critical(e, exc_info=True)






