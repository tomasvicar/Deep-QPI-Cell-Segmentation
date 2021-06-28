import os
import numpy as np
import torch
from torch.utils import data
import nvidia_smi
import matplotlib.pyplot as plt
from shutil import copyfile
import segmentation_models_pytorch as smp
from shutil import rmtree

from dataset import Dataset
from unet import Unet
from utils.log import Log
from utils.training_fcns import l1_loss,l2_loss,dice_loss_logit


def get_gpu_memory(nvidia_smi):
    
    try:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        return info.used/1000000000
    except:
        return 0



def train(config,names_train,names_valid):

    
    device = torch.device(config.device)
    
    try:
        nvidia_smi.nvmlInit()
        measured_gpu_memory = []
        measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
    except:
        measured_gpu_memory = []

    train_generator = Dataset(names_train,augment=True,crop=True,config=config)
    train_generator = data.DataLoader(train_generator,batch_size=config.train_batch_size,num_workers= config.train_num_workers, shuffle=True,drop_last=True)

    valid_generator = Dataset(names_valid,augment=False,crop=True,config=config)
    valid_generator = data.DataLoader(valid_generator,batch_size=config.valid_batch_size, num_workers=config.valid_num_workers, shuffle=True,drop_last=True)

    
    if config.model_name_load == 'imagenet':
        model = smp.Unet(
            encoder_name="efficientnet-b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model.config = config
    
    elif config.model_name_load:
        model=torch.load(config.model_name_load)
    else:
        # model=Unet(config, filters=config.filters,in_size=config.in_size,out_size=config.out_size)
        model = smp.Unet(
            encoder_name="efficientnet-b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        model.config = config
        
        
        
    model=model.to(device)
    
    model.log =Log()
    
    
    
    optimizer = torch.optim.AdamW(model.parameters(),lr =config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_changes_list, gamma=config.gamma, last_epoch=-1)

    model_names=[]
    for epoch in range(config.max_epochs):
        
        model.train()
        for img,mask,gt in train_generator:
            
            img = img.to(torch.device(config.device))
            
            res=model(img)
            
            
            if config.method == 'dt' or config.method == 'ndt' or config.method == 'pretraining':
                loss=l2_loss(res,mask)
            else:
                loss=dice_loss_logit(res,mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.log.save_tmp_log(loss,'train')
            measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
                
            
        model.eval()
        with torch.no_grad():
            for img,mask,gt in valid_generator:
                
                img = img.to(torch.device(config.device))
                
                res=model(img)
                
                
                if config.method == 'dt' or config.method == 'ndt' or config.method == 'pretraining':
                    loss=l2_loss(res,mask)
                else:
                    loss=dice_loss_logit(res,mask)
                
                model.log.save_tmp_log(loss,'valid')
                measured_gpu_memory.append(get_gpu_memory(nvidia_smi))
            
        
        model.log.save_log_data_and_clear_tmp()
        
        model.log.plot_training()
        
        
        
        
        if not (config.method == 'dt' or config.method == 'ndt' or config.method == 'pretraining'):
            res=torch.sigmoid(res)
            
        res = res.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        for k in range(res.shape[0]):
            plt.imshow(np.concatenate((img[k,0,:,:],res[k,0,:,:],mask[k,0,:,:]),axis=1),vmin=0,vmax=1)
            plt.show()
            plt.close()
    
    
        xstr = lambda x:"{:.5f}".format(x)
        lr=optimizer.param_groups[0]['lr']
        info= '_' + str(epoch) + '_' + xstr(lr) + '_gpu_' + xstr(np.max(measured_gpu_memory)) + '_train_'  + xstr(model.log.train_loss_log[-1]) + '_valid_' + xstr(model.log.valid_loss_log[-1]) 
        print(info)
        
        model_name=config.model_save_dir+ os.sep + config.method + info  + '.pt'
        
        model_names.append(model_name)
        
        torch.save(model,model_name)
        
        model.log.plot_training(model_name.replace('.pt','loss.png'))
        
        scheduler.step()
    
    
    
    best_model_ind=np.argmin(model.log.valid_loss_log)
    best_model_name= model_names[best_model_ind]   
    best_model_name_new=best_model_name.replace(config.model_save_dir,config.best_models_dir)
    
    copyfile(best_model_name,best_model_name_new)
    
    if os.path.isdir(config.model_save_dir):
        rmtree(config.model_save_dir) 
    if not os.path.isdir(config.model_save_dir):
            os.mkdir(config.model_save_dir)


    return best_model_name_new
