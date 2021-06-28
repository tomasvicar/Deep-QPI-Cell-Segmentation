import numpy as np
from shutil import copyfile,rmtree
from config import Config
import os



def split_train_test(seed=666,config=Config(),reduction=1):
    
    
    path_train = config.data_pretrain_train_valid_path + os.sep + 'train'
    path_valid = config.data_pretrain_train_valid_path + os.sep + 'valid'
    

    
    try:
        rmtree(config.data_pretrain_train_valid_path)
    except:
        pass
    
    
    
    try:
        os.makedirs(path_train)
    except:
        pass
    
    try:
        os.makedirs(path_valid)
    except:
        pass
    
    
    
    path_train = config.data_train_valid_test_path + os.sep + 'train'
    path_valid = config.data_train_valid_test_path + os.sep + 'valid'
    path_test = config.data_train_valid_test_path + os.sep + 'test'
    
    
    
    try:
        rmtree(config.data_train_valid_test_path)
    except:
        pass
    
    
    
    try:
        os.makedirs(path_train+os.sep+'PC3')
    except:
        pass
    
    try:
        os.makedirs(path_valid+os.sep+'PC3')
    except:
        pass
    
    try:
        os.makedirs(path_test+os.sep+'PC3')
    except:
        pass
    
    
    
    try:
        os.makedirs(path_train+os.sep+'PNT1A')
    except:
        pass
    
    try:
        os.makedirs(path_valid+os.sep+'PNT1A')
    except:
        pass
    
    try:
        os.makedirs(path_test+os.sep+'PNT1A')
    except:
        pass
    
    
    
    
    
    
    
    
    
    np.random.seed(seed)
    
    
   

    
    names=[]
    for root, dirs, files in os.walk(config.data_orig_path):
        for name in files:
            if name.endswith("_img.tif"):
                name=name.replace('_img.tif','')
                names.append(root + os.sep +name)
                
         
    perm=np.random.permutation(len(names))   
         
    split_ind=np.array(config.split_ratio_train_valid_test)
    split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(names))).astype(np.int)
    
    
    train_ind=perm[:split_ind[0]]
    valid_ind=perm[split_ind[0]:split_ind[1]]         
    test_ind=perm[split_ind[1]:]       
                
    train_ind_tmp = train_ind[:int(reduction*len(train_ind))]
    for k in train_ind_tmp:
        name_in=names[k]
        name_out=name_in.replace(config.data_orig_path,path_train)
        
        copyfile(name_in + '_img.tif',name_out+'_img.tif')
        copyfile(name_in + '_mask.png',name_out+'_mask.png')
        
        
    for k in train_ind:
        name_in=names[k]
        name_out=name_in.replace(config.data_orig_path,path_train)
        
        name_out2=name_in.replace(config.data_orig_path,config.data_pretrain_train_valid_path + os.sep + 'train').replace(os.sep + 'PC3','').replace(os.sep + 'PNT1A','')
        copyfile(name_in + '_img.tif',name_out2+'_img.tif')
        
      
    for k in valid_ind:
        name_in=names[k]
        name_out=name_in.replace(config.data_orig_path,path_valid)
        
        copyfile(name_in + '_img.tif',name_out+'_img.tif')
        copyfile(name_in + '_mask.png',name_out+'_mask.png')
        
    
    for k in test_ind:
        name_in=names[k]
        name_out=name_in.replace(config.data_orig_path,path_test)
        
        copyfile(name_in + '_img.tif',name_out+'_img.tif')
        copyfile(name_in + '_mask.png',name_out+'_mask.png')
    
    
    
    
    
    
    
    
    
    
    
    np.random.seed(666)
    
    
    path_train = config.data_pretrain_train_valid_path + os.sep + 'train'
    path_valid = config.data_pretrain_train_valid_path + os.sep + 'valid'
    

    


    
    names=[]
    for root, dirs, files in os.walk(config.data_pretrain_orig_path):
        for name in files:
            if name.endswith(".tif"):
                names.append(root + os.sep + name)
    
    
    
    perm=np.random.permutation(len(names))   
         
    split_ind=np.array(config.split_ratio_pretrain_train_valid)
    split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(names))).astype(np.int)
    
    
    train_ind=perm[:split_ind[0]]
    valid_ind=perm[split_ind[0]:]
    
    
    for k in train_ind:
        name_in=names[k]
        name_out=name_in.replace(config.data_pretrain_orig_path,path_train)
        
        copyfile(name_in, name_out)
        copyfile(name_in, name_out)
        
      
    for k in valid_ind:
        name_in=names[k]
        name_out=name_in.replace(config.data_pretrain_orig_path,path_valid)
        
        copyfile(name_in, name_out)
        copyfile(name_in, name_out)
    
    
if __name__ == "__main__":
    
    split_train_test()