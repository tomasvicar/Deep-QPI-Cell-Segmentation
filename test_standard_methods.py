import torch
import os
from skimage.io import imread,imsave
import numpy as np
import matplotlib.pyplot as plt
from utils.visboundaries import visboundaries
from utils.colorize_notouchingsamecolor import colorize_notouchingsamecolor
import pickle
from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp
from utils.overlap_for_ap import overlap_for_ap
from skimage.measure import label


def test_standard_methods(segmentation_name,names):
    


                      
    jacards=[]
    
    
    binary_gts=[]
    binary_res=[]
    
    for_aps_jac = []
    for_aps_fp = []
    for_aps_fn = []
    
    
    for name_num,name in enumerate(names):
        
        
        img=imread(name)
        img0=img.astype(np.float32)
        img=img0.copy()
        
        mask=imread(name.replace("_img.tif","_mask.png"))
        mask0=mask.copy()
        
    

        tmp = imread(name.replace('data_train_valid_test',segmentation_name).replace('.tif','.png').replace(r'\test\PC3','').replace(r'\test\PNT1A',''))>0
        

        tmp = label(tmp)
        
        jacards=jacards+get_jacards_cell_with_fp(mask,tmp)
        
        
        binary_gts.append(mask)
        binary_res.append(tmp)
        
        aps_jac,aps_fp,aps_fn = overlap_for_ap(mask,tmp)
            
        for_aps_jac.append(aps_jac)
        for_aps_fp.append(aps_fp)
        for_aps_fn.append(aps_fn)




        
    mean_jacard = np.mean(jacards)
    


    tmp_gt = np.stack(binary_gts,axis=2)>0
    tmp_res = np.stack(binary_res,axis=2)>0
    mean_binary_jacard = np.sum(tmp_gt & tmp_res)/np.sum(tmp_gt | tmp_res)
    
    
    
    jacards_ap = np.concatenate(for_aps_jac,axis=0)
    fp = np.sum(for_aps_fp)
    fn = np.sum(for_aps_fn)
 
    
    aps=[]
    for t in np.linspace(0.5,1,100):
        
        tp_tmp = np.sum(jacards_ap>=t)
        
        fp_tmp = fp + np.sum(jacards_ap<t)
        
        fn_tmp = fn + np.sum(jacards_ap<t)
        
        ap = tp_tmp / (tp_tmp+fp_tmp+fn_tmp)
        
        aps.append(ap)
    
    
    
    
        
    return mean_jacard,mean_binary_jacard,aps
    
        



