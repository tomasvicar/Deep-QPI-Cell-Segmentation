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


def test_fcn(config,segmenter_name,names_test):
    
    
    save_folder = '../outputs/' + config.method + str(config.border_width)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    
    segmenter = pickle.load( open( segmenter_name, "rb" ) )
    
    
    
    names = names_test
                      
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
        
    
        
        shape=np.shape(img)
        img=torch.from_numpy(img.reshape((1,1,shape[0],shape[1])).astype(np.float32))
        
    
        
        imgs=segmenter.predict_imgs(img)
        tmp=segmenter.get_segmentation(imgs)
        
        img=img.detach().cpu().numpy()[0,0,:,:]
        
        
        imsave(save_folder+'/img_'+str(name_num).zfill(5) + '.tif', img)
        imsave(save_folder+'/gt_'+str(name_num).zfill(5) + '.tif', mask)
        imsave(save_folder+'/result_'+str(name_num).zfill(5) + '.tif', tmp)
        
        
        jacards=jacards+get_jacards_cell_with_fp(mask,tmp)
        
        
        binary_gts.append(mask)
        binary_res.append(tmp)
        
        aps_jac,aps_fp,aps_fn = overlap_for_ap(mask,tmp)
            
        for_aps_jac.append(aps_jac)
        for_aps_fp.append(aps_fp)
        for_aps_fn.append(aps_fn)
        
        
        
        colormap_cells=[[1,0,0],[0,1,0],[0,0,1],[0.8314,0.8314,0.0588],[1,0,1],[1,0.5,0],[0.00,1.00,1.00],[0.45,0.00,0.08]]

        
        L=colorize_notouchingsamecolor(tmp>0)
        
        plt.figure(figsize=(10,20))
        plt.subplot(121)
        plt.imshow(img)
        for k,c in enumerate(colormap_cells):
            visboundaries(L==(k+1),color=c)
        
        plt.subplot(122)
        plt.imshow((mask0>0).astype(np.int),'gray',vmin=-1,vmax=2)
        for k,c in enumerate(colormap_cells):
            visboundaries(L==(k+1),color=c)
        plt.savefig(save_folder+'/example_'+str(name_num).zfill(5) + '.png')
        plt.show()    

        
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
    
        



