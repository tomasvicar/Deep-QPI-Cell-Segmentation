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
    
            
            

            # for pretraind_model_name,method,border in [['imagenet','boundary_line',4],['1','boundary_line',4],['2','boundary_line',4] ]:
            for pretraind_model_name,method,border in [['1','boundary_line',4] ]:

                if pretraind_model_name == '1':
                    config.pretrain_num_blocks = 13
                    config.pretrain_max_block_size = 50
                    config.pretrain_noise_std_fraction = 0.6472
                    config.pretrain_noise_pixel_p = 0.3134
                    config.pretrain_chessboard_num_blocks = 15
                    config.pretrain_chessboard_max_block_size = 53
                    config.pretrain_rot_num_blocks = 10
                    config.pretrain_rot_max_block_size = 47
                    
                    names_train =  glob(config.data_pretrain_train_valid_path+ os.sep+'train' + '/**/*.tif',recursive=True)
                    names_valid =  glob(config.data_pretrain_train_valid_path+ os.sep+'valid'+'/**/*.tif',recursive=True)
                
                    config.model_name_load = 'imagenet'
                    # config.model_name_load = None
                    config.method = 'pretraining'
                    config.border_width = 88
                    
                    pretraind_model_name = train(config,names_train,names_valid)
                    
                    
                if pretraind_model_name == '2':
                    config.pretrain_num_blocks = 20
                    config.pretrain_max_block_size = 34
                    config.pretrain_noise_std_fraction = 0.4045
                    config.pretrain_noise_pixel_p = 0.06879
                    config.pretrain_chessboard_num_blocks = 14
                    config.pretrain_chessboard_max_block_size = 32
                    config.pretrain_rot_num_blocks = 42
                    config.pretrain_rot_max_block_size = 63
                    
                    names_train =  glob(config.data_pretrain_train_valid_path+ os.sep+'train' + '/**/*.tif',recursive=True)
                    names_valid =  glob(config.data_pretrain_train_valid_path+ os.sep+'valid'+'/**/*.tif',recursive=True)
                
                    config.model_name_load = 'imagenet'
                    # config.model_name_load = None
                    config.method = 'pretraining'
                    config.border_width = 88
                    
                    pretraind_model_name = train(config,names_train,names_valid)
                    



                cell_type_train ='*'
    
                config.model_name_load = pretraind_model_name
                config.method = 'semantic'
                config.border_width = 10
        
                names_train =  glob(config.data_train_valid_test_path+ os.sep+'train' +'/**/*' + cell_type_train + '_img.tif', recursive=True)
                names_valid = glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*' + cell_type_train + '_img.tif', recursive=True)
  
                fg_model_name = train(config,names_train,names_valid)
    
                last_pretrain_model_name = pretraind_model_name
                
                config.method = method
                config.border_width = border
                config.model_name_load = pretraind_model_name
                
                names_train =  glob(config.data_train_valid_test_path+ os.sep+'train' +'/**/*' + cell_type_train + '_img.tif', recursive=True)
                names_valid = glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*' + cell_type_train + '_img.tif', recursive=True)
                best_model_name = train(config,names_train,names_valid)
                # best_model_name = '../best_models/boundary_line_78_0.00100_gpu_5.23135_train_0.25452_valid_0.21993.pt'
                
                print(best_model_name)
                
                cell_type_opt = '*'
                
                
                names_valid =  glob(config.data_train_valid_test_path+ os.sep+'valid' +'/**/*' + cell_type_opt + '_img.tif', recursive=True)
                segmenter_name = optimize_segmentation(config,fg_model_name,best_model_name,names_valid)
                # segmenter_name = '../best_models/dt_0.697048542955851.p'
                
                print(segmenter_name)
                
                cell_type_res = '*'
                
                names_test =  glob(config.data_train_valid_test_path + os.sep + 'test' +'/**/*' + cell_type_res + '_img.tif', recursive=True)
                
                mean_jacard,mean_binary_jacard,aps = test_fcn(config,segmenter_name,names_test)
            
        
            
                results['mean_jacard'].append(mean_jacard)
                results['mean_binary_jacard'].append(mean_binary_jacard)
                results['aps'].append(aps)
                results['segmenter_name'].append(segmenter_name)
                results['method'].append(config.method)
                results['border_width'].append(config.border_width)
                results['pretraind_model_name'].append(pretraind_model_name)
                results['cell_type_train'].append(cell_type_train)
                results['cell_type_opt'].append(cell_type_opt)
                results['cell_type_res'].append(cell_type_res)
                            
                            
                
            with open('../result_crossval_pret_mix' + str(it) + '.json', 'w') as outfile:
                json.dump(results, outfile)    
        
            
    except Exception as e:
        logging.critical(e, exc_info=True)
        
        
        
        
