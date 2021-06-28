from torch.utils import data
import numpy as np
import torch 
import os
from skimage.io import imread
from glob import glob

from scipy.ndimage  import distance_transform_edt
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation,binary_closing,binary_erosion
from skimage.morphology import disk

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import laplace

import matplotlib.pyplot as plt
import cv2



def augmentation(img,mask,config):
    
    def rand():
        return torch.rand(1).numpy()[0]
    
    cols=img.shape[0]
    rows=img.shape[1]
    sr=0.2
    gr=0.05
    tr=0
    dr=100
    rr=180
    #sr = scales
    #gr = shears
    #tr = tilt
    #dr = translation
    sx=1+sr*rand()
    if rand()>0.5:
        sx=1/sx
    sy=1+sr*rand()
    if rand()>0.5:
        sy=1/sy
    gx=(0-gr)+gr*2*rand()
    gy=(0-gr)+gr*2*rand()
    tx=(0-tr)+tr*2*rand()
    ty=(0-tr)+tr*2*rand()
    dx=(0-dr)+dr*2*rand()
    dy=(0-dr)+dr*2*rand()
    t=(0-rr)+rr*2*rand()
    
    M=np.array([[sx, gx, dx], [gy, sy, dy],[tx, ty, 1]])
    R=cv2.getRotationMatrix2D((cols / 2, rows / 2), t, 1)
    R=np.concatenate((R,np.array([[0,0,1]])),axis=0)
    matrix= np.matmul(R,M)

    img = cv2.warpPerspective(img,matrix, (cols,rows),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    if not config.method == 'pretraining':
        mask = cv2.warpPerspective(mask,matrix, (cols,rows),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)
    
    r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
    if r[0]:
        img=np.fliplr(img)
        if not config.method == 'pretraining':
            mask=np.fliplr(mask)
    if r[1]:
        img=np.flipud(img)
        if not config.method == 'pretraining':
            mask=np.flipud(mask) 
    img=np.rot90(img,k=r[2]) 
    if not config.method == 'pretraining':
        mask=np.rot90(mask,k=r[2])    
    
    
    multipy=0.2 
    multipy=1+rand()*multipy
    if rand()>0.5:
        img=img*multipy
    else:
        img=img/multipy
       
    add=0.2     
    add=(1-2*rand())*add
    img=img+add
    
    
    
    
    
    bs_r=(-0.5,0.5)
    r=1-2*rand()
    if r<=0:
        par=bs_r[0]*r
        img=img-par*laplace(img)
    if r>0:
        par=bs_r[1]*r
        img=gaussian_filter(img,par)

    
    return img,mask







class Dataset(data.Dataset):


    def __init__(self, names,augment,crop,config,crop_same=False):
       
        self.names = names
        self.augment = augment
        self.crop = crop
        self.config = config
        self.crop_same = crop_same
        
        

    def __len__(self):
        return len(self.names)


    def __getitem__(self, idx):

        name = self.names[idx]
        
        if not self.config.method == 'pretraining':
            mask=imread(name.replace("_img.tif","_mask.png"))
            mask=mask.astype(np.float32)
            
        else:
            mask = None
            
        img=imread(name)
        img=img.astype(np.float64)
        
        
        if self.augment:
            img,mask = augmentation(img,mask,self.config)
        
        
        in_size=img.shape
        out_size=[self.config.patch_size,self.config.patch_size]
        
        
        if self.crop:
            r1=torch.randint(in_size[0]-out_size[0],(1,1)).view(-1).numpy()[0]
            r2=torch.randint(in_size[1]-out_size[1],(1,1)).view(-1).numpy()[0]
            r=[r1,r2]
            
            if self.crop_same:
                r = [100,100]
            
            
            img=img[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]
            if not self.config.method == 'pretraining':
                mask=mask[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]    
         

            
                
            
        shape=np.shape(img)
        if not self.config.method == 'pretraining':
            gt=torch.from_numpy(mask.reshape((1,shape[0],shape[1])).astype(np.float32))
        else:
            gt = 0
        
         
        if self.config.method =='cell_border':
            labeled_array=mask
            
            mask_new=np.zeros(mask.shape)
            strel=disk(self.config.border_width).astype(np.bool)
            u=np.unique(labeled_array)
            u=u[u>0]
            for k in u:
                cell=labeled_array==k
                nucs=binary_erosion(cell,strel)
                mask_new=mask_new+nucs
            mask=mask_new
            
            
        
        if self.config.method =='boundary_line':
            labeled_array = mask
             
            tmp=np.zeros(mask.shape,dtype=np.int)
            strel = disk(5) #np.ones((5,5),dtype=bool)
            N=int(np.max(labeled_array))
            for k in range(N):
                tmp = tmp +  binary_dilation(labeled_array==(k+1),strel).astype(np.int)
                
            lines = tmp>1
            
            strel=disk(self.config.border_width).astype(np.bool)
            lines = binary_dilation(lines,strel)

            mask=lines
        
        
            
        if self.config.method =='dt':
            labeled_array=mask
            u=np.unique(labeled_array)
            u=u[u>0]
            mask_new=np.zeros(mask.shape)
            for k in u:
                cell=labeled_array==k
                dt=distance_transform_edt(cell)
                mask_new=mask_new+dt/np.max(dt)
            mask=mask_new
        
        if self.config.method =='ndt':
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
            
        if self.config.method =='semantic':
            mask=mask>0
            
        if self.config.method =='pretraining':
            mask = img.copy()
            
            def rand():
                return torch.rand(1).numpy()[0]
            
            img_shape = img.shape
            
            if self.config.pretrain_noise_std_fraction:
                img = img + torch.randn(img_shape).numpy()*self.config.pretrain_std*self.config.pretrain_noise_std_fraction
                
            if self.config.pretrain_noise_pixel_p:
                
                tmp = torch.rand(img_shape).numpy()<self.config.pretrain_noise_pixel_p
                img[tmp] = 0
                img = img + (tmp).astype(np.float32) * torch.randn(img_shape).numpy() * self.config.pretrain_std*self.config.pretrain_noise_pixel_std_fraction
                
                
                
            block_types = ['chess' for _ in range(self.config.pretrain_chessboard_num_blocks)] + ['rot' for _ in range(self.config.pretrain_rot_num_blocks)] + ['del' for _ in range(self.config.pretrain_num_blocks)]
                
            p = torch.randperm(len(block_types)).numpy()
            
            block_types = [block_types[k] for k in p]
                
            used = np.zeros(img_shape)
            for k,block_type in enumerate(block_types):
                
                if block_type == 'del':
                    block_sizex = int(np.ceil(rand()*self.config.pretrain_max_block_size))
                    block_sizey = int(np.ceil(rand()*self.config.pretrain_max_block_size))
                elif block_type == 'chess':
                    block_sizex = int(np.ceil(self.config.pretrain_chessboard_max_block_size/2 + rand()*self.config.pretrain_chessboard_max_block_size/2)/2)*2
                    # block_sizey = int(np.ceil(self.config.pretrain_chessboard_max_block_size/2 + rand()*self.config.pretrain_chessboard_max_block_size/2)/2)*2
                    block_sizey = block_sizex
                    
                else:
                    block_sizex = int(np.ceil(self.config.pretrain_chessboard_max_block_size/2 + rand()*self.config.pretrain_chessboard_max_block_size/2))
                    block_sizey = block_sizex
                
                posx = int(np.round(rand()*(img_shape[0]-block_sizex)))
                posy = int(np.round(rand()*(img_shape[1]-block_sizey)))
                
                if np.sum(used[posx:posx+block_sizex,posy:posy+block_sizey])==00:
                    used[posx:posx+block_sizex,posy:posy+block_sizey] = 1
                    
                    
                    if block_type == 'del':
                        block = torch.randn([block_sizex,block_sizey]).numpy()*self.config.pretrain_std + self.config.pretrain_mean
                    
                    if block_type == 'rot':
                        
                        block = img[posx:posx+block_sizex,posy:posy+block_sizey]
                        # if rand()>0.5:
                        #     block = block + 0.2
                        # else:
                        #     block = block - 0.2
                        
                        
                        r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
                        
                        if r[0]:
                            block=np.fliplr(block)
        
                        if r[1]:
                            block=np.flipud(block)
        
                        block=np.rot90(block,k=r[2]) 


                    if block_type == 'chess':
                        
                        block = img[posx:posx+block_sizex,posy:posy+block_sizey]
                        # if rand()>0.5:
                        #     block = block + 0.2
                        # else:
                        #     block = block - 0.2
                        
                        
                        p = torch.randperm(4).numpy()
                        
                        x_split = int(block_sizex/2)
                        y_split = int(block_sizey/2)
                        
                        sub_blocks = []

                        sub_blocks.append(block[:x_split,:y_split].copy())
                        sub_blocks.append(block[x_split:,:y_split].copy())
                        sub_blocks.append(block[:x_split,y_split:].copy())
                        sub_blocks.append(block[x_split:,y_split:].copy())
                        
                        block[:x_split,:y_split] = sub_blocks[p[0]]
                        block[x_split:,:y_split] = sub_blocks[p[1]]
                        block[:x_split,y_split:] = sub_blocks[p[2]]
                        block[x_split:,y_split:] = sub_blocks[p[3]]
                    
                    
                    img[posx:posx+block_sizex,posy:posy+block_sizey] = block
                    
                    

            
        mask=torch.from_numpy(mask.reshape((1,shape[0],shape[1])).astype(np.float32))
        img=torch.from_numpy(img.reshape((1,shape[0],shape[1])).astype(np.float32))
        
        
        return img,mask,gt







if __name__ == "__main__":
    from config import Config    
    
    config = Config()
    config.method = 'pretraining'
    config.border_width = 10
    
    config.pretrain_num_blocks = 20
    config.pretrain_max_block_size = 40
    config.pretrain_noise_std_fraction = 0
    config.pretrain_noise_pixel_p = 0
    config.pretrain_chessboard_num_blocks = 20
    config.pretrain_chessboard_max_block_size = 40
    config.pretrain_rot_num_blocks = 20
    config.pretrain_rot_max_block_size = 40
    
    
    
    names_train = glob(config.data_pretrain_train_valid_path+ os.sep+'train' + '/**/*.tif',recursive=True)
    train_generator = Dataset(names_train,augment=True,crop=True,config=config)
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers= 0, shuffle=True,drop_last=True)
    

    for img,mask,gt in train_generator:
        
        plt.imshow(img[0,0,:,:],vmin=-0.5,vmax=2)
        plt.show()
        plt.imshow(mask[0,0,:,:],vmin=-0.5,vmax=2)
        plt.show()
        
        break


