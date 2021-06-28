import logging
import sys
import numpy as np
from glob import glob
import os
import json
from shutil import rmtree
from torch.utils import data
import pickle
from skimage.io import imread,imsave
from skimage.measure import label

from config import Config
from train import train
from optimize_segmentation import optimize_segmentation
from test_fcn import test_fcn
from split_train_test import split_train_test
from dataset import Dataset
import matplotlib.pyplot as plt
from utils.visboundaries import visboundaries
from utils.colorize_notouchingsamecolor import colorize_notouchingsamecolor
from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp
from utils.mat2gray import mat2gray

it = 0


# split_train_test(seed=it*100)



for method,xxx in [['frst','88'],['loewke','88'],['qpi_dt','88'],['qpi_log2','88']]:

    save_folder0 = '../examples/' + method + xxx
    
    if not os.path.isdir(save_folder0):
        os.makedirs(save_folder0)
    
    config = Config()
    
    
    names_test =  glob(config.data_train_valid_test_path + os.sep + 'test' +'/**/*' + '*' + '_img.tif', recursive=True)
    
    
    names_test = [names_test[k] for k in [0,1,2,3,4,5,6,7,32,33,34,35,36,37,38,39,40,41,42]]

    
    config.method = method
    try:
        config.border_width = int(xxx)
    except:
        config.border_width = xxx
    
    
    
    if method!='pretraining':
        test_generator = Dataset(names_test,augment=False,crop=False,config=config)
    else:
        test_generator = Dataset(names_test,augment=False,crop=True,config=config)
        
    test_generator = data.DataLoader(test_generator,batch_size=1, num_workers=0, shuffle=False,drop_last=False)



    for name_num,(name,(img,mask,gt)) in enumerate(zip(names_test,test_generator)):
        
        nametmp2 = os.path.split(name)[1][:-4]
        
        save_folder = save_folder0 + '/' + nametmp2
        
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        nametmp = ''
        
        
        namex = 'D:/qpi_segmentation_tmp/for_standard_methods/' + method + '_res0/' + nametmp2 + '.png'  
        
        res_method = imread(namex)
        
        foreground = res_method>0
        result = res_method 
        
        img0 = img.detach().cpu().numpy()[0,0,:,:]
        mask0 = gt.detach().cpu().numpy()[0,0,:,:]
        
        
        tmp = label(result)
        jaccard = np.mean(get_jacards_cell_with_fp(mask0.astype(np.int64),tmp))
        
        
        
        imsave(save_folder+'/foreground_predicted_' + nametmp + '.tif', foreground)
        imsave(save_folder+'/output_' + nametmp + '.tif', result>0)
        
        
        
        colormap_cells=[[1,0,0],[0,1,0],[0,0,1],[0.8314,0.8314,0.0588],[1,0,1],[1,0.5,0],[0.00,1.00,1.00],[0.45,0.00,0.08]]

        L=colorize_notouchingsamecolor(result>0)
        
        plt.figure(figsize=(10,10))
        plt.imshow(img0,'gray',vmin=-0.5,vmax=1.7)
        plt.title(str(jaccard))
        for k,c in enumerate(colormap_cells):
            visboundaries(L==(k+1),color=c)
        plt.savefig(save_folder+'/example_results_onimg_'+nametmp + '.png')
        plt.savefig(save_folder+'/example_results_onimg_'+nametmp + '.svg', transparent=True)
        plt.show()    
        plt.close()
        
        
        plt.figure(figsize=(10,10))
        plt.imshow((mask0>0).astype(np.int),'gray',vmin=-1,vmax=2)
        for k,c in enumerate(colormap_cells):
            visboundaries(L==(k+1),color=c)
        plt.savefig(save_folder+'/example_results_ongt_'+nametmp + '.png')
        plt.savefig(save_folder+'/example_results_ongt_'+nametmp + '.svg', transparent=True)
        plt.show()    
        plt.close()
        
            
            
            
        L=colorize_notouchingsamecolor(mask0>0)
        
        plt.figure(figsize=(10,10))
        plt.imshow(img0,'gray',vmin=-0.5,vmax=1.7)
        for k,c in enumerate(colormap_cells):
            visboundaries(L==(k+1),color=c)
        plt.savefig(save_folder+'/example_gt_onimg_'+nametmp + '.png')
        plt.savefig(save_folder+'/example_gt_onimg_'+nametmp + '.svg', transparent=True)
        plt.show()    
        plt.close()
            
            
            
            
            
        
        img = img.detach().cpu().numpy()[0,0,:,:]
        mask = mask.detach().cpu().numpy()[0,0,:,:]
        
        
        imsave(save_folder+'/input_'+nametmp + '.tif', mat2gray(img,[-0.5,1.7]))
        
        if method!='pretraining':
            imsave(save_folder+'/gt_output_'+nametmp + '.tif', mask)
        else:
            imsave(save_folder+'/gt_output_'+nametmp + '.tif', mat2gray(mask,[-0.5,1.7]))
        
        try:
            gt = gt.detach().cpu().numpy()[0,0,:,:]
            imsave(save_folder+'/gt_'+nametmp + '.tif', gt)
        except:
            pass
            
        











