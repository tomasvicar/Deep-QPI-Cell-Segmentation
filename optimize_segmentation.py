import os
import numpy as np
import torch
from torch import optim
from torch.utils import data
from dataset import Dataset
from unet import Unet
from config import Config
from utils.log import Log
import matplotlib.pyplot as plt


from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp
from scipy.optimize import dual_annealing
from skimage.measure import label
from bayes_opt import BayesianOptimization

from segmenters import DtSegmenter,NdtSegmenter,CellBorderSegmenter,BoundaryLineSegmenter

import pickle

def optimize_segmentation(config,fg_model_name,best_model_name,names_valid,get_value=False):

    
    dt_model_name = best_model_name
    
    seg_model_name = fg_model_name
    
    
    

    if config.method == 'dt':    

        segmenter=DtSegmenter()
        
        func = lambda min_dist,min_value,min_h,min_size: segmenter.get_mean_jacard(gts, imgss,[min_dist,min_value,min_h,min_size])
    
    elif config.method == 'ndt':   
    
        segmenter=NdtSegmenter()
        
        func = lambda min_dist,er_size,min_h,min_size: segmenter.get_mean_jacard(gts, imgss,[min_dist,er_size,min_h,min_size])
    
    elif config.method == 'cell_border': 
          
        segmenter=CellBorderSegmenter()
        
        func = lambda min_dist,min_h,min_size1,min_size2: segmenter.get_mean_jacard(gts, imgss,[min_dist,min_h,min_size1,min_size2])
     
        
    elif config.method == 'boundary_line': 
        
        segmenter=BoundaryLineSegmenter()

        func = lambda min_dist,er_size,min_size1,min_size2: segmenter.get_mean_jacard(gts, imgss,[min_dist,er_size,min_size1,min_size2])
    

    segmenter.save_models_name(dt_model_name,seg_model_name)
    
    
    valid_generator = Dataset(names_valid,augment=False,crop=False,config=config)
    valid_generator = data.DataLoader(valid_generator,batch_size=1, num_workers=0, shuffle=False,drop_last=False)
    
     
    
    gts=[]
    imgss=[]
    
    for k,(img,mask,gt) in enumerate(valid_generator):
        print(k)
        
        
        gts.append(gt.detach().cpu().numpy()[0,0,:,:]>0)
    
        
        imgs=segmenter.predict_imgs(img)
        
        imgss.append(imgs)
    
    
    pbounds=dict(zip(segmenter.param_names, zip(segmenter.bounds_lw,segmenter.bounds_up)))
    
        
    optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)
    
    optimizer.maximize(init_points=10,n_iter=150)
    
    best_value=optimizer.max['target']
    
    tmp=optimizer.max['params']
    params=[]
    for param_name in segmenter.param_names:
        params.append(tmp[param_name])
    
    
    value=segmenter.get_mean_jacard(gts,imgss,params)
    print(value)
    
    
    segmenter.save_detection_params(params)      
    
    segmenter_name=Config.best_models_dir + os.sep + config.method + '_' + str(value) + '.p'
    pickle.dump( segmenter, open( segmenter_name, "wb" ) )
    
    
    if get_value:
        return segmenter_name,value
    else:
        return segmenter_name




