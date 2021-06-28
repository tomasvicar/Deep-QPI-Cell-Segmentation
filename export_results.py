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


from scipy.ndimage  import distance_transform_edt
# from scipy.ndimage.measurements import label
from skimage.measure import label
from scipy.ndimage.morphology import binary_dilation,binary_closing,binary_erosion             
from skimage.morphology import disk


data_path='../data_train_valid_test/test'

segmenter_names = ['boundary_line_0.6696171116230956.p' ,'loewke_1','loewke_2','ndt_0.6538656675812204.p','dt_0.6640225926514111.p',
                   'cell_border_0.562846843054472.p']

cell_types = ['PNT1A','PC3']


all_jacards=[]
all_binary_jacards=[]
all_aps = []



for segmenter_name in segmenter_names:

    for cell_type  in cell_types:
        
        save_folder = '../'  + segmenter_name.split('.')[0] + '/' + cell_type
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        if not (segmenter_name=='loewke_1' or segmenter_name=='loewke_2'):
            segmenter = pickle.load( open( 'best_models/' +segmenter_name, "rb" ) )
        
        
        names=[]
        for root, dirs, files in os.walk(data_path):
            for name in files:
                if name.endswith("_img.tif") and name.find(cell_type)>0:
                    name=name.replace('_img.tif','')
                    names.append(root + os.sep +name)
                    
        
        
        
        jacards=[]
        
        binary_gts=[]
        binary_res=[]
        
        for_aps_jac = []
        for_aps_fp = []
        for_aps_fn = []
        
        for name_num,name in enumerate(names):
            
            
            img=imread(name + "_img.tif")
            img0=img.astype(np.float32)
            img=img0.copy()
            
            mask=imread(name + "_mask.png")
            mask0=mask.copy()
            
            
            qqq=segmenter_name.split('.')[0]
            
            if qqq =='cell_border_0':
                labeled_array=mask
                
                mask_new=np.zeros(mask.shape)
                strel=disk(6).astype(np.bool)
                u=np.unique(labeled_array)
                u=u[u>0]
                for k in u:
                    cell=labeled_array==k
                    nucs=binary_erosion(cell,strel)
                    mask_new=mask_new+nucs
                mask=mask_new
                
                
            
            if qqq =='boundary_line_0':
                labeled_array = mask
                 
                tmp=np.zeros(mask.shape,dtype=np.int)
                strel = disk(5) #np.ones((5,5),dtype=bool)
                N=int(np.max(labeled_array))
                for k in range(N):
                    tmp = tmp +  binary_dilation(labeled_array==(k+1),strel).astype(np.int)
                    
                lines = tmp>1
                
                strel=disk(2).astype(np.bool)
                lines = binary_dilation(lines,strel)
    
                mask=lines
            
            
                
            if qqq =='dt_0':
                labeled_array=mask
                u=np.unique(labeled_array)
                u=u[u>0]
                mask_new=np.zeros(mask.shape)
                for k in u:
                    cell=labeled_array==k
                    dt=distance_transform_edt(cell)
                    mask_new=mask_new+dt/np.max(dt)
                mask=mask_new
            
            if qqq =='ndt_0':
                labeled_array=mask
                u=np.unique(labeled_array)
                u=u[u>0]
                
                mask_new=np.zeros(mask.shape)
                
                strel = disk(5) #np.ones((5,5),dtype=bool)
                tmp_mask=mask>0
                tmp_mask=binary_closing(tmp_mask,strel)
                
                for k in u:
                    cell=binary_dilation(labeled_array==k,strel)
                    other_cells=(mask>0)
                    other_cells[cell]=0
                    max_val=20
                    dt=distance_transform_edt(other_cells==0)
                    dt=max_val-dt
                    dt=dt*cell
                    dt[dt<0]=0
                    dt=dt/max_val
                    mask_new=mask_new+dt
                mask_new[tmp_mask==0]=0
                mask_new[mask_new>1]=1
                
                mask=mask_new
                
            if not (segmenter_name=='loewke_1' or segmenter_name=='loewke_2'):
                gt = mask.copy()
                mask = mask0.copy()
                
                shape=np.shape(img)
                img=torch.from_numpy(img.reshape((1,1,shape[0],shape[1])).astype(np.float32))
                
    
                imgs=segmenter.predict_imgs(img)
                tmp=segmenter.get_segmentation(imgs)
                
                img=img.detach().cpu().numpy()[0,0,:,:]
                
            elif segmenter_name=='loewke_1':
                mask = mask0.copy()
                gt = mask.copy()
                img = img.copy()
                imgs = [img,img]
                tmp = imread('../loewke_res/' + name.split('\\')[-1] + '_img.png')
            
                tmp=label(tmp>0,connectivity=1)
                
            elif segmenter_name=='loewke_2' :
                mask = mask0.copy()
                gt = mask.copy()
                img = img.copy()
                imgs = [img,img]
                tmp = imread('../loewke_res/' + name.split('\\')[-1] + '_img.png')
            
                tmp=label(tmp>0,connectivity=2)
                
                
                
        

            
            imsave(save_folder+'/img_'+str(name_num).zfill(5) + '.tif', img)
            imsave(save_folder+'/gt_'+str(name_num).zfill(5) + '.tif', mask)
            imsave(save_folder+'/whatpredict_'+str(name_num).zfill(5) + '.tif', gt.astype(np.int))
            imsave(save_folder+'/whatpredicted_'+str(name_num).zfill(5) + '.tif', imgs[0])
            imsave(save_folder+'/whatpredictedfg_'+str(name_num).zfill(5) + '.tif', imgs[1].astype(np.float32))
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
    
            
    
        
            
        mean_jacard=np.mean(jacards)
        all_jacards.append(mean_jacard)
        print(mean_jacard)


        tmp_gt = np.stack(binary_gts,axis=2)>0
        tmp_res = np.stack(binary_res,axis=2)>0
        mean_binary_jacard = np.sum(tmp_gt & tmp_res)/np.sum(tmp_gt | tmp_res)
        all_binary_jacards.append(mean_binary_jacard)
        print(mean_binary_jacard)
        
        
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
        
        
        all_aps.append(aps)
        
        


np.save('all_jacards.npy', all_jacards)
np.save('all_binary_jacards.npy', all_binary_jacards)
np.save('all_aps.npy', all_aps)





metrics = ['DAP','BIoU','OIoU']


    
index = -1
all_types_res = [np.zeros((len(segmenter_names),len(metrics))),np.zeros((len(segmenter_names),len(metrics)))]
for segmenter_num,segmenter_name in enumerate(segmenter_names):   
    for cell_type_num,cell_type in enumerate(cell_types):
        
        index = index+1
        for metric_num,metric in enumerate(metrics):
            if metric == 'DAP':
                number = all_aps[index][0]
            if metric == 'BIoU':
                number = all_binary_jacards[index]
            if metric == 'OIoU':
                number = all_jacards[index]
            
            
            
            all_types_res[cell_type_num][segmenter_num,metric_num] = number
        
        
        
    
            
            
        
all_types_res = np.concatenate(all_types_res,axis=1)


all_types_res = all_types_res[[1,4,3,0,5],:]


# segmenter_names = ['boundary_line_0.6696171116230956.p' ,'loewke_1','loewke_2','ndt_0.6538656675812204.p','dt_0.6640225926514111.p',
#                    'cell_border_0.562846843054472.p']