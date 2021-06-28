import numpy as np
import torch
from scipy.ndimage.morphology import grey_dilation
from skimage.morphology import h_maxima
from skimage.measure import regionprops
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp
from skimage.morphology import binary_erosion,disk
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from multiprocessing import Pool
from itertools import chain
import time
from itertools import starmap

def evaluate_one_dt(dt_seg,gt,params,get_segmentation):
    res_L=get_segmentation(dt_seg,params)
    gt_L=label(gt>0)
    return get_jacards_cell_with_fp(gt_L,res_L)

class Wrapper_dt(object):
    def __init__(self, params,get_segmentation):
        self.params = params
        self.get_segmentation = get_segmentation
    def __call__(self, dt_seg,gt):
        return evaluate_one_dt(dt_seg,gt, self.params,self.get_segmentation)


class DtSegmenter():
    
    def __init__(self):
    
        self.param_names=['min_dist','min_value','min_h','min_size']
        self.bounds_lw=[1,   0,      0.05,      20]
        self.bounds_up=[80,  0.9,    0.9,    400]
    
    def save_detection_params(self,params):
                              
        min_dist,min_value,min_h,min_size=params
        self.min_dist=min_dist
        self.min_value=min_value
        self.min_h=min_h  
        self.min_size=min_size
    
    def save_models_name(self,dt_model_name,seg_model_name):
        
        self.dt_model_name=dt_model_name
        self.seg_model_name=seg_model_name
        
    def predict_imgs(self,img):
        
        device = torch.device('cuda:0')


        model=torch.load(self.dt_model_name)
        model.eval()
        model=model.to(device)
        
        
        model_semantic=torch.load(self.seg_model_name)
        model_semantic.eval()
        model_semantic=model_semantic.to(device)
        
        img = img.to(device)
        
        img = F.pad(img,(0,40,0,40),'reflect')
        res=model(img)
        res = res[:,:,:-40,:-40]
        dt=res.detach().cpu().numpy()[0,0,:,:]
        
        res=model_semantic(img)
        res = res[:,:,:-40,:-40]
        seg=res.detach().cpu().numpy()[0,0,:,:]>0
        
        
        return [dt,seg]
        
        
        
    
    def get_segmentation(self,dt_seg,params=None):

        dt,seg=dt_seg      

       
        if params==None:
            min_dist= self.min_dist
            min_value = self.min_value
            min_h = self.min_h
            min_size = self.min_size
            
        else:
            min_dist,min_value,min_h,min_size=params
        
            
            
        min_dist=int(min_dist)
        
        p1=peak_local_max(dt,min_distance=min_dist,threshold_abs=min_value)
        p2=h_maxima(dt,min_h)
        final=np.zeros(dt.shape)
        for p in p1:
            final[int(p[0]),int(p[1])]=p2[int(p[0]),int(p[1])]
        
        
        final=label(final>0,connectivity=1)
        
        dt_int=np.round((dt*500)).astype(np.int)
        
        w = watershed(-dt_int,markers=final,mask=seg, watershed_line=True)
        
        w=remove_small_objects(w,min_size)
        
        w=label(w,connectivity=1)
        
        return w
    
    
    def get_mean_jacard(self,gts,dts_segs,params):
    

        
        # with Pool() as pool:
        #     jacards = pool.starmap(Wrapper_dt(params,self.get_segmentation), zip(dts_segs,gts))
        
        jacards = starmap(Wrapper_dt(params,self.get_segmentation), zip(dts_segs,gts))    


        jacards = list(chain.from_iterable(jacards))
        
        mean_jacards=np.mean(jacards)
        
        
        
        return mean_jacards
    
    
    
    
    
    
    
    
    
    
class NdtSegmenter():
    
    def __init__(self):
    
        self.param_names=['min_dist','er_size','min_h','min_size']
        self.bounds_lw=[1,   1,      0.05,      20]
        self.bounds_up=[80,  20,    0.9,    400]
    
    def save_detection_params(self,params):
                              
        min_dist,er_size,min_h,min_size=params
        self.min_dist=min_dist
        self.er_size=er_size
        self.min_h=min_h  
        self.min_size=min_size
    
    def save_models_name(self,dt_model_name,seg_model_name):
        
        self.dt_model_name=dt_model_name
        self.seg_model_name=seg_model_name
        
    def predict_imgs(self,img):
        
        device = torch.device('cuda:0')


        model=torch.load(self.dt_model_name)
        model.eval()
        model=model.to(device)
        
        
        model_semantic=torch.load(self.seg_model_name)
        model_semantic.eval()
        model_semantic=model_semantic.to(device)
        
        img = img.to(device)
        img = F.pad(img,(0,40,0,40),'reflect')
        res=model(img)
        res = res[:,:,:-40,:-40]
        dt=res.detach().cpu().numpy()[0,0,:,:]
        
        res=model_semantic(img)
        res = res[:,:,:-40,:-40]
        seg=res.detach().cpu().numpy()[0,0,:,:]>0
        
        
        return [dt,seg]
        
        
        
    
    def get_segmentation(self,dt_seg,params=None):

        dt,seg=dt_seg      

       
        if params==None:
            min_dist= self.min_dist
            er_size = self.er_size
            min_h = self.min_h
            min_size = self.min_size
            
        else:
            min_dist,er_size,min_h,min_size=params
            
              
        min_dist=int(min_dist)
        er_size=int(er_size)
        
        strel=disk(er_size)
        seg_er=binary_erosion(seg,strel)
        
        dt_er=-dt.copy()
        dt_er[seg_er==0]=-np.inf
        
        p1=peak_local_max(dt_er,min_distance=min_dist)
        p2=h_maxima(dt_er,min_h)
        final=np.zeros(dt.shape)
        for p in p1:
            final[int(p[0]),int(p[1])]=p2[int(p[0]),int(p[1])]
        
        
        final=label(final>0)
        
        dt_int=np.round((dt*500)).astype(np.int)
        
        
        w = watershed(dt_int,markers=final,mask=seg, watershed_line=True)
        
        
        
        
        w=remove_small_objects(w>0,min_size)
        
        w=label(w,connectivity=1)
        
        return w
    
    
    def get_mean_jacard(self,gts,dts_segs,params):
    
        
        final_res=[]
        for k in range(len(gts)):
            
            gt=gts[k]
            dt_seg=dts_segs[k]
            
            
            tmp=self.get_segmentation(dt_seg,params)
            final_res.append(tmp)
    
        
        jacards=[]
        for k in range(len(final_res)):
        
            gt_L=label(gts[k]>0)
            res_L=final_res[k]
            
            jacards=jacards+get_jacards_cell_with_fp(gt_L,res_L)
            
    
        mean_jacards=np.mean(jacards)
        
        return mean_jacards
    
    
    
    
    
    
    
    
class CellBorderSegmenter():
    
    def __init__(self):
    
        self.param_names=['min_dist','min_h','min_size1','min_size2']
        self.bounds_lw=[1,       0.05,      10,     10]
        self.bounds_up=[100,     10,    500,    500]
    
    def save_detection_params(self,params):
                              
        min_dist,min_h,min_size1,min_size2=params
        self.min_dist=min_dist
        self.min_h=min_h  
        self.min_size1=min_size1
        self.min_size2=min_size2
    
    def save_models_name(self,dt_model_name,seg_model_name):
        
        self.dt_model_name=dt_model_name
        self.seg_model_name=seg_model_name
        
    def predict_imgs(self,img):
        
        device = torch.device('cuda:0')


        model=torch.load(self.dt_model_name)
        model.eval()
        model=model.to(device)
        
        
        model_semantic=torch.load(self.seg_model_name)
        model_semantic.eval()
        model_semantic=model_semantic.to(device)
        
        img = img.to(device)
        img = F.pad(img,(0,40,0,40),'reflect')
        res=model(img)
        res = res[:,:,:-40,:-40]
        dt=res.detach().cpu().numpy()[0,0,:,:]
        
        res=model_semantic(img)
        res = res[:,:,:-40,:-40]
        seg=res.detach().cpu().numpy()[0,0,:,:]>0
        
        img = img[:,:,:-40,:-40]
        img=img.detach().cpu().numpy()[0,0,:,:]>0
        
        
        return [dt,seg,img]
        
        
        
    
    def get_segmentation(self,dt_seg_img,params=None):

        dt,seg,img=dt_seg_img   

       
        if params==None:
            min_dist= self.min_dist
            min_h = self.min_h
            min_size1 = self.min_size1
            min_size2 = self.min_size2
            
        else:
            min_dist,min_h,min_size1,min_size2=params
        
            
            
        min_dist=int(min_dist)
        
        cell_centers=dt
        
        
        
        cell_centers=cell_centers>0
        
        dt = distance_transform_edt(cell_centers)
        
        
        p1=peak_local_max(dt,min_distance=min_dist)
        
        
        p2=h_maxima(dt,min_h)
        final=np.zeros(dt.shape)
        for p in p1:
            final[int(p[0]),int(p[1])]=p2[int(p[0]),int(p[1])]
        
        
        final=label(final>0,connectivity=1)
        
        dt_int=np.round((dt*500)).astype(np.int)
        
        w = watershed(-dt_int,markers=final,mask=cell_centers, watershed_line=True)
        
        w=remove_small_objects(w,min_size1)
        
        
        
        
        cell_centers=remove_small_objects(cell_centers,min_size1)
        cell_centers=label(cell_centers,connectivity=1)
        N=np.max(cell_centers)
        for cell_num in range(N):
            if np.sum((cell_centers==(cell_num+1)).astype(np.float)*w)==0:
                w[cell_centers==(cell_num+1)] = 1
        
        
        markers=label(w,connectivity=1)
        
        w = watershed(-np.round((img*500)).astype(np.int),markers=markers,mask=seg, watershed_line=True)
        
        w=label(w,connectivity=1)
        
        N=np.max(w)
        for cell_num in range(N):
            if np.sum((w==(cell_num+1)).astype(np.float)*markers)==0:
                w[w==(cell_num+1)] = 0
        
        
        w=remove_small_objects(w>0,min_size2)
        
        w=label(w>0,connectivity=1)


        
        return w
    
    
    def get_mean_jacard(self,gts,dts_segs,params):
    
        
        final_res=[]
        for k in range(len(gts)):
            
            gt=gts[k]
            dt_seg=dts_segs[k]
            
            
            tmp=self.get_segmentation(dt_seg,params)
            final_res.append(tmp)
    
        
        jacards=[]
        for k in range(len(final_res)):
        
            gt_L=label(gts[k]>0)
            res_L=final_res[k]
            
            jacards=jacards+get_jacards_cell_with_fp(gt_L,res_L)
            
    
        mean_jacards=np.mean(jacards)
        
        return mean_jacards
        
    
    

class BoundaryLineSegmenter():
    
    def __init__(self):
    
        self.param_names=['min_dist','er_size','min_size1','min_size2']
        self.bounds_lw=[1,       0,      10,    10]
        self.bounds_up=[100,     6,    500,     500]
    
    def save_detection_params(self,params):
                              
        min_dist,er_size,min_size1,min_size2=params
        self.min_dist=min_dist
        self.er_size=er_size  
        self.min_size1=min_size1
        self.min_size2=min_size2
    
    def save_models_name(self,dt_model_name,seg_model_name):
        
        self.dt_model_name=dt_model_name
        self.seg_model_name=seg_model_name
        
    def predict_imgs(self,img):
        
        device = torch.device('cuda:0')


        model=torch.load(self.dt_model_name)
        model.eval()
        model=model.to(device)
        
        
        model_semantic=torch.load(self.seg_model_name)
        model_semantic.eval()
        model_semantic=model_semantic.to(device)
        
        img = img.to(device)
        img = F.pad(img,(0,40,0,40),'reflect')
        res=model(img)
        res = res[:,:,:-40,:-40]
        dt=res.detach().cpu().numpy()[0,0,:,:]
        
        res=model_semantic(img)
        res = res[:,:,:-40,:-40]
        seg=res.detach().cpu().numpy()[0,0,:,:]>0
        
        img = img[:,:,:-40,:-40]
        img=img.detach().cpu().numpy()[0,0,:,:]>0
        
        
        return [dt,seg,img]
        
        
        
    
    def get_segmentation(self,dt_seg_img,params=None):

        dt,seg,img=dt_seg_img   

       
        if params==None:
            min_dist= self.min_dist
            er_size = self.er_size
            min_size1 = self.min_size1
            min_size2 = self.min_size2
            
        else:
            min_dist,er_size,min_size1,min_size2=params
        
            
            
        border_lines = dt>0.5

        min_dist=int(min_dist)
        er_size=int(er_size)
        
        
        
        w = seg.copy()
        w = binary_erosion(w,disk(er_size))
        w[border_lines] = 0
        
        
        tofilt=label(w,connectivity=1)
        area_points=np.zeros_like(tofilt)
        
        props = regionprops(tofilt)
        for prop in props:
            centroid = prop.centroid
            area = prop.area
            area_points[int(centroid[0]),int(centroid[1])] = area
        
        
        p1=peak_local_max(area_points,min_distance=min_dist)
        area_points=np.zeros(area_points.shape)
        for p in p1:
            area_points[int(p[0]),int(p[1])]=1
        
        
        w = label(w,connectivity=1)
        N=np.max(w)
        for cell_num in range(N):
            if np.sum(area_points[w==(cell_num+1)])==0:
                w[w==(cell_num+1)]=0
        
        
        
        w=remove_small_objects(w,min_size1)
        
        seg=remove_small_objects(seg,min_size1)
        seg=label(seg,connectivity=1)
        N=np.max(seg)
        for cell_num in range(N):
            if np.sum((seg==(cell_num+1)).astype(np.float)*w)==0:
                w[seg==(cell_num+1)] = 1
        
        
        markers=label(w,connectivity=1)
        
        w = watershed(-np.round((img*500)).astype(np.int),markers=markers,mask=seg, watershed_line=True)
        
        w=label(w,connectivity=1)
        
        N=np.max(w)
        for cell_num in range(N):
            if np.sum((w==(cell_num+1)).astype(np.float)*markers)==0:
                w[w==(cell_num+1)] = 0
        
        w=remove_small_objects(w>0,min_size2)
        
        w=label(w>0,connectivity=1)


        
        return w
    
    
    def get_mean_jacard(self,gts,dts_segs,params):
    
        
        final_res=[]
        for k in range(len(gts)):
            
            gt=gts[k]
            dt_seg=dts_segs[k]
            
            
            tmp=self.get_segmentation(dt_seg,params)
            final_res.append(tmp)
    
        
        jacards=[]
        for k in range(len(final_res)):
        
            gt_L=label(gts[k]>0)
            res_L=final_res[k]
            
            jacards=jacards+get_jacards_cell_with_fp(gt_L,res_L)
            
    
        mean_jacards=np.mean(jacards)
        
        return mean_jacards    
    
    
    
    
    
min_dist=3
er_size=1
min_h=0.01
min_size=50
lam=1
    
class MixedSegmenter():
    
    def __init__(self):
    
        self.param_names=['min_dist','er_size','min_h','min_size','lam']
        self.bounds_lw=[1,       0,    0.05,      10,    0]
        self.bounds_up=[100,     6,    10,       500,    1]
        
    
    def save_detection_params(self,params):
                              
        min_dist,er_size,min_size=params
        self.min_dist=min_dist
        self.er_size=er_size 
        self.min_h=min_h 
        self.min_size=min_size
        self.lam=lam 
    
    def save_models_name(self,dt_model_name,seg_model_name):
        
        self.dt_model_name=dt_model_name
        self.seg_model_name=seg_model_name
        
    def predict_imgs(self,img):
        
        device = torch.device('cuda:0')


        model=torch.load(self.dt_model_name)
        model.eval()
        model=model.to(device)
        
        
        model_semantic=torch.load(self.seg_model_name)
        model_semantic.eval()
        model_semantic=model_semantic.to(device)
        
        img = img.to(device)
        img = F.pad(img,(0,40,0,40),'reflect')
        res=model(img)
        res = res[:,:,:-40,:-40]
        dt=res.detach().cpu().numpy()[0,0,:,:]
        
        res=model_semantic(img)
        res = res[:,:,:-40,:-40]
        seg=res.detach().cpu().numpy()[0,0,:,:]>0
        
        img = img[:,:,:-40,:-40]
        img=img.detach().cpu().numpy()[0,0,:,:]>0
        
        
        return [dt,seg,img]
        
        
        
    
    def get_segmentation(self,dt_seg_img,params=None):

        dt,seg,img=dt_seg_img   

       
        if params==None:
            min_dist= self.min_dist
            er_size = self.er_size
            min_size = self.min_size
            
        else:
            min_dist,er_size,min_size=params
        
            
            
        border_lines = dt>0.5

        min_dist=int(min_dist)
        er_size=int(er_size)
        
        
        
        w = seg.copy()
        w = binary_erosion(w,disk(er_size))
        w[border_lines] = 0
        
        
        tofilt=label(w,connectivity=1)
        area_points=np.zeros_like(tofilt)
        
        props = regionprops(tofilt)
        for prop in props:
            centroid = prop.centroid
            area = prop.area
            area_points[int(centroid[0]),int(centroid[1])] = area
        
        
        p1=peak_local_max(area_points,min_distance=min_dist)
        area_points=np.zeros(area_points.shape)
        for p in p1:
            area_points[int(p[0]),int(p[1])]=1
        
        
        w = label(w,connectivity=1)
        N=np.max(w)
        for cell_num in range(N):
            if np.sum(area_points[w==(cell_num+1)])==0:
                w[w==(cell_num+1)]=0
        
        
        
        w=remove_small_objects(w,min_size)
        
        seg=remove_small_objects(seg,min_size)
        seg=label(seg,connectivity=1)
        N=np.max(seg)
        for cell_num in range(N):
            if np.sum((seg==(cell_num+1)).astype(np.float)*w)==0:
                w[seg==(cell_num+1)] = 1
        
        
        markers=label(w,connectivity=1)
        
        w = watershed(-np.round((img*500)).astype(np.int),markers=markers,mask=seg, watershed_line=True)
        
        w=label(w,connectivity=1)
        
        N=np.max(w)
        for cell_num in range(N):
            if np.sum((w==(cell_num+1)).astype(np.float)*markers)==0:
                w[w==(cell_num+1)] = 0
        
        w=remove_small_objects(w>0,min_size)
        
        w=label(w>0,connectivity=1)


        
        return w
    
    
    def get_mean_jacard(self,gts,dts_segs,params):
    
        
        final_res=[]
        for k in range(len(gts)):
            
            gt=gts[k]
            dt_seg=dts_segs[k]
            
            
            tmp=self.get_segmentation(dt_seg,params)
            final_res.append(tmp)
    
        
        jacards=[]
        for k in range(len(final_res)):
        
            gt_L=label(gts[k]>0)
            res_L=final_res[k]
            
            jacards=jacards+get_jacards_cell_with_fp(gt_L,res_L)
            
    
        mean_jacards=np.mean(jacards)
        
        return mean_jacards    