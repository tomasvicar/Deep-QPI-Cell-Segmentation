import torch
import os
from skimage.io import imread,imsave
import numpy as np
import matplotlib.pyplot as plt
from utils.visboundaries import visboundaries
from utils.colorize_notouchingsamecolor import colorize_notouchingsamecolor
import pickle
from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp



save_folder = '../vysledky_pro_vaska'

segmenter_name='best_models/dt_segmeter_0.6316848369709267.p'



segmenter = pickle.load( open( segmenter_name, "rb" ) )


data_path='../data_train_valid_test/test'

names=[]
for root, dirs, files in os.walk(data_path):
    for name in files:
        if name.endswith("_img.tif"):
            name=name.replace('_img.tif','')
            names.append(root + os.sep +name)
                  
jacards=[]
for name_num,name in enumerate(names):
    
    print(str(name_num)+'/'+str(len(names)))
    
    
    img=imread(name + "_img.tif")
    img=img.astype(np.float32)
    img0=img.copy()
    
    mask=imread(name + "_mask.png")
    mask0=mask.copy()
    

    
    shape=np.shape(img)
    img=torch.from_numpy(img.reshape((1,1,shape[0],shape[1])).astype(np.float32))
    

    
    imgs=segmenter.predict_imgs(img)
    tmp=segmenter.get_segmentation(imgs)
    
    img=img.detach().cpu().numpy()[0,0,:,:]
    
    
    imsave(save_folder+'/img_'+str(name_num).zfill(5) + '.tif', img0)
    imsave(save_folder+'/gt_'+str(name_num).zfill(5) + '.tif', mask)
    imsave(save_folder+'/result_'+str(name_num).zfill(5) + '.tif', tmp)
    
    imsave(save_folder+'/estimatedDT_'+str(name_num).zfill(5) + '.tif', imgs[0])
    imsave(save_folder+'/estimatedFG_'+str(name_num).zfill(5) + '.tif', imgs[1].astype(np.float32))

    jacards=jacards+get_jacards_cell_with_fp(mask,tmp)
            
    
    colormap_cells=[[1,0,0],[0,1,0],[0,0,1],[0.8314,0.8314,0.0588],[1,0,1],[1,0.5,0],[0.00,1.00,1.00],[0.45,0.00,0.08]]
    # colormap_cells=[[0.0000,0.4470,0.7410],[0.8500,0.3250,0.0980],[0.9290,0.6940,0.1250],
    #                 [0.4940,0.1840,0.5560],[0.4660,0.6740,0.1880],[0.3010,0.7450,0.9330],
    #                 [0.6350,0.0780,0.1840],[0.7500,0.0000,0.7500]]
    
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
    
    
    
    
    
    
    
mean_jacards=np.mean(jacards)
      
print(mean_jacards)
