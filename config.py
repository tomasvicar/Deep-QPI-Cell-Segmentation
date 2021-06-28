import numpy as np
import os

class Config:
    
    model_save_dir='../tmp'
    
    best_models_dir='../best_models'
    
    opt_folder = '../opt_res'
    
    data_orig_path ="../data"
    
    data_train_valid_test_path ="../data_train_valid_test"
    
    split_ratio_train_valid_test=[8.5,0.5,1]
    
    
    data_pretrain_orig_path = "../data_pretraining"
    
    data_pretrain_train_valid_path = "../data_pretraining_train_valid"
    
    split_ratio_pretrain_train_valid = [9.5,0.5]
    
    
    train_batch_size = 16
    train_num_workers = 8
    valid_batch_size = 4
    valid_num_workers = 2


    init_lr = 0.01
    lr_changes_list = np.cumsum([40,20,10,5])
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    

    filters=list((np.array([64,128,256,512,1024])/4).astype(np.int))
    in_size=1
    out_size=1
    
    
    device='cuda:0'
    
    
    patch_size=256
    
    
    pretrain_num_blocks = 0
    pretrain_max_block_size = 20
    pretrain_mean = 0.10273255
    pretrain_std = 0.31816605
    pretrain_noise_std_fraction = None
    pretrain_noise_pixel_p = None
    pretrain_noise_pixel_std_fraction = 5
    pretrain_chessboard_num_blocks = 0
    pretrain_chessboard_max_block_size = 20
    pretrain_rot_num_blocks = 0
    pretrain_rot_max_block_size = 20
    
    
    model_name_load = None
    method = None
    border_width = None
    
    
    