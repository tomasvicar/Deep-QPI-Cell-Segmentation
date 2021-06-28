import logging
import sys
import numpy as np
from glob import glob
import os
import json
from shutil import rmtree

from config import Config
from train import train
from optimize_segmentation import optimize_segmentation
from test_fcn import test_fcn
from split_train_test import split_train_test


if __name__ == "__main__":
    
    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
    
    # if True:
        
        kk = -1
        results = {'mean_jacard':[],'mean_binary_jacard':[],'aps':[],'segmenter_name':[],'method':[],'border_width':[],
                   'pretraind_model_name':[],'cell_type_train':[],'cell_type_opt':[],'cell_type_res':[]}
        
        for it in range(5):
        
            config = Config()
            
 
            if os.path.isdir(config.model_save_dir):
                rmtree(config.model_save_dir) 
                

            split_train_test(seed=it*100)
                
            if not os.path.isdir(config.best_models_dir):
                os.mkdir(config.best_models_dir)
                
            if not os.path.isdir(config.model_save_dir):
                os.mkdir(config.model_save_dir)
    
            with open('../result_crossval_pret_dt' + str(it) + '.json', 'r') as file:
                data = json.load(file)
            

            # for pretraind_model_name,method,border in [['imagenet','boundary_line',4],['1','boundary_line',4],['2','boundary_line',4] ]:
            # for pretraind_model_name,method,border in [['1','boundary_line',4] ]:
                
            for pretraind_model_name,method,border in [['imagenet','dt',4],['1','dt',4],['2','dt',4] ]:

                kk = kk+1
                
                config.method = 'xxx'
                config.border_width = 999
                
                
                
                segmenter_name = data['segmenter_name'][kk]
                
                cell_type_res = '*'
                
                names_test =  glob(config.data_train_valid_test_path + os.sep + 'test' +'/**/*' + cell_type_res + '_img.tif', recursive=True)
                
                mean_jacard,mean_binary_jacard,aps = test_fcn(config,segmenter_name,names_test)
                
                print(mean_jacard)
                print( data['mean_jacard'][kk])
                
                
                data['aps'][kk] = aps
                            
                            
                
            with open('../result_crossval_pret_dt_apfix' + str(it) + '.json', 'w') as outfile:
                json.dump(data, outfile)    
        
            
    except Exception as e:
        logging.critical(e, exc_info=True)
        
        
        
        
