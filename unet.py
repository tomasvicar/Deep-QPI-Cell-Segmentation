import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=1,do_batch=1):
        super().__init__()
        
        self.do_batch=do_batch
    
        self.conv=nn.Conv2d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)


        # dov=0.1
        # self.do=nn.Sequential(nn.Dropout(dov),nn.Dropout2d(dov))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        
        if self.do_batch:
            outputs = self.bn(outputs)          
        outputs=F.relu(outputs)
        # outputs = self.do(outputs)

        return outputs
    
    
    

class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=2,pad=1,out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_size, out_size,filter_size,stride=stride, padding=pad, output_padding=out_pad)
        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs=F.relu(outputs)
        return outputs
    
    
    
    
    

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()

        self.up = unetConvT2(in_size, out_size )
        
#        self.up=nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, inputs1, inputs2):
        
       
        inputs2 = self.up(inputs2)
        
        shape1=list(inputs1.size())
        shape2=list(inputs2.size())
        
        pad=(0,shape1[-2]-shape2[-2],0,shape1[-1]-shape2[-1])
        
        inputs2=F.pad(inputs2,pad)

        return torch.cat([inputs1, inputs2], 1)




class Unet(nn.Module):
    def __init__(self,config, filters=[64, 128, 256, 512,1024],in_size=3,out_size=1):
        super().__init__()
        
        self.config=config
        
        
        self.out_size = out_size
        
        self.filters = filters
        
        
        
        self.conv1 = nn.Sequential(unetConv2(in_size, filters[0]),unetConv2(filters[0], filters[0]),unetConv2(filters[0], filters[0]))

        if len(filters)>=3:
            self.conv2 =  nn.Sequential(unetConv2(filters[0], filters[1] ),unetConv2(filters[1], filters[1] ),unetConv2(filters[1], filters[1] ))

        if len(filters)>=4:
            self.conv3 = nn.Sequential(unetConv2(filters[1], filters[2] ),unetConv2(filters[2], filters[2] ),unetConv2(filters[2], filters[2] ))

        if len(filters)>=5:
            self.conv4 = nn.Sequential(unetConv2(filters[2], filters[3] ),unetConv2(filters[3], filters[3] ),unetConv2(filters[3], filters[3] ))


        self.center = nn.Sequential(unetConv2(filters[-2], filters[-1] ),unetConv2(filters[-1], filters[-1] ))


        # upsampling
        if len(filters)>=5:
            self.up_concat4 = unetUp(filters[4], filters[4] )
            self.up_conv4=nn.Sequential(unetConv2(filters[3]+filters[4], filters[3] ),unetConv2(filters[3], filters[3] ))

        
        if len(filters)>=4:
            self.up_concat3 = unetUp(filters[3], filters[3] )
            self.up_conv3=nn.Sequential(unetConv2(filters[2]+filters[3], filters[2] ),unetConv2(filters[2], filters[2] ))



        if len(filters)>=3:
            self.up_concat2 = unetUp(filters[2], filters[2] )
            self.up_conv2=nn.Sequential(unetConv2(filters[1]+filters[2], filters[1] ),unetConv2(filters[1], filters[1] ))


        self.up_concat1 = unetUp(filters[1], filters[1])
        self.up_conv1=nn.Sequential(unetConv2(filters[0]+filters[1], filters[0] ),unetConv2(filters[0], filters[0],do_batch=0 ))
        
        
        
        self.final = nn.Conv2d(filters[0], self.out_size, 1)
        

        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            
        
        
        
    def forward(self, inputs):
        
        
        required_divisibility=2**len(self.filters)
        shape=list(inputs.size())
        shape1=shape[-2]
        shape2=shape[-1]
        pad1=int((np.ceil(shape1/required_divisibility)*required_divisibility)-shape1)
        pad2=int((np.ceil(shape2/required_divisibility)*required_divisibility)-shape2)
        inputs=F.pad(inputs,(0,pad1,0,pad2),mode='reflect')
        
        cuda_check = next(self.parameters())
        if cuda_check.is_cuda:
            cuda_device = cuda_check.get_device()
            device = torch.device('cuda:' + str(cuda_device) )
        else:
            device = torch.device('cpu')

        inputs=inputs.to(device)
        


        conv1 = self.conv1(inputs)
        x = F.max_pool2d(conv1,2,2)

        if len(self.filters)>=3:
            conv2 = self.conv2(x)
            x = F.max_pool2d(conv2,2,2)

        if len(self.filters)>=4:
            conv3 = self.conv3(x)
            x = F.max_pool2d(conv3,2,2)

        if len(self.filters)>=5:
            conv4 = self.conv4(x)
            x = F.max_pool2d(conv4,2,2)

        x = self.center(x)

        if len(self.filters)>=5:
            x = self.up_concat4(conv4, x)
            x=self.up_conv4(x)
        
        if len(self.filters)>=4:
            x = self.up_concat3(conv3, x)
            x = self.up_conv3(x)

        if len(self.filters)>=3:
            x = self.up_concat2(conv2, x)
            x=self.up_conv2(x)
        
        x = self.up_concat1(conv1, x)
        x=self.up_conv1(x)
        

        x = self.final(x)
        
#        x=torch.sigmoid(x)

        x=x[:,:,:shape1,:shape2]
        
        return x
    

    def save_log(self,log):
        self.log=log
        

        
    