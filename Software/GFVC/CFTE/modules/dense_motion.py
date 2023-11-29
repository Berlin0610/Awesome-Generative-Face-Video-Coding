# -*- coding: utf-8 -*-
# + {}
from torch import nn
import torch.nn.functional as F
import torch
from GFVC.CFTE.modules.util import * 
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from GFVC.CFTE.modules.vggloss import *
from GFVC.CFTE.modules.flowwarp import *

# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda:0" if USE_CUDA else "cpu")
# -

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks,num_down_blocks, max_features, num_kp, num_channels, num_bottleneck_blocks,
                 estimate_occlusion_map=False,scale_factor=1, kp_variance=0.01):
        
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features= num_channels + 1,
                                   max_features=max_features, num_blocks=num_blocks)

        self.flow = nn.Conv2d(self.hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3))
        
        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.num_down_blocks=num_down_blocks

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        ###heatmap_difference upscale    
        up_blocks = []
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(num_kp, num_kp, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
     
    
        ####sparse motion warp---downscale-->upscale
        
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        
        motiondown_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            motiondown_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motiondown_blocks = nn.ModuleList(motiondown_blocks)

        motionup_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            motionup_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motionup_blocks = nn.ModuleList(motionup_blocks)    

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))        
        
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
          
            
    def create_heatmap_representations(self, source_image, heatmap_driving, heatmap_source):
        """
        Eq 6. in the paper H_k(z)
        8*8 Feature-->upscale-->64*64 Feature---->Feature Difference ####torch.Size([40, 1, 64, 64])
        """

        #heatmap = heatmap_driving['value']  - heatmap_source['value'] 
        bs, _, h, w = source_image.shape
        heatmap_d=heatmap_driving['value']
        heatmap_s=heatmap_source['value'] 
        
        for i in range(self.num_down_blocks):
            heatmap_d = self.up_blocks[i](heatmap_d)
            heatmap_s = self.up_blocks[i](heatmap_s)
            
        heatmap = heatmap_d  - heatmap_s
        return heatmap
    

       ###Gunnar Farneback算法计算稠密光流 
    def create_sparse_motions(self, source_image, heatmap_driving, heatmap_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """      

        #feature map-->img-->sparse motion detecion point p0 and p1     
        heatmap_source_lt = heatmap_source['value'] 
        heatmap_source_lt = heatmap_source_lt.cuda().data.cpu().numpy()         
        heatmap_source_lt = (heatmap_source_lt - np.min(heatmap_source_lt))/(np.max(heatmap_source_lt) - np.min(heatmap_source_lt)) *255.0  #转为0-255  
        heatmap_source_lt=np.round(heatmap_source_lt)   #转换数据类型
        heatmap_source_lt=heatmap_source_lt.astype(np.uint8)

        heatmap_driving_lt = heatmap_driving['value']
        heatmap_driving_lt = heatmap_driving_lt.cuda().data.cpu().numpy()         
        heatmap_driving_lt = (heatmap_driving_lt - np.min(heatmap_driving_lt))/(np.max(heatmap_driving_lt) - np.min(heatmap_driving_lt)) *255.0  #转为0-255  
        heatmap_driving_lt=np.round(heatmap_driving_lt)   #转换数据类型
        heatmap_driving_lt=heatmap_driving_lt.astype(np.uint8)       
        
        bs, _, h, w = source_image.shape  ##bs=40
        
        GFflow=[]
        for tensorchannel in range(0,bs):
        
            heatmap_source_lt11=heatmap_source_lt[tensorchannel].transpose([1,2,0])      #取出其中一张并转换维度
            heatmap_driving_lt11=heatmap_driving_lt[tensorchannel].transpose([1,2,0])      #取出其中一张并转换维度
            #flow = cv2.calcOpticalFlowFarneback(heatmap_driving_lt11,heatmap_source_lt11,None, 0.5, 2, 15, 3, 5, 1.2, 0)  ####
            flow = cv2.calcOpticalFlowFarneback(heatmap_source_lt11,heatmap_driving_lt11,None, 0.5, 2, 15, 3, 5, 1.2, 0)  ####            
            GFflow.append(flow)
            
        tmp_flow=torch.Tensor(np.array(GFflow)).cuda()#.to(device)         
        return tmp_flow    


    def create_deformed_source_image(self, source_image, sparse_motion):
        ''' [bs, 3, 64, 64])-->[bs, 64, 64, 64]-->[bs, 128, 32, 32]-->[bs, 256, 16, 16]-->[bs, 512, 8, 8]
        '''
        # Encoding (downsampling) part

        out = self.first(source_image)  
        for i in range(self.num_down_blocks):
            out = self.motiondown_blocks[i](out) 
        
        ########
        # warping
        out=warp(out, sparse_motion)
        
        # Decoding part
        out = self.bottleneck(out)
        for i in range(self.num_down_blocks):
            out = self.motionup_blocks[i](out)
        
        ##deformed image
        out = self.final(out)  
        return out
    

    def forward(self, source_image,heatmap_source,heatmap_driving):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            
        bs, c, h, w = source_image.shape   
        out_dict = dict()
        
        heatmap_representation = self.create_heatmap_representations(source_image, heatmap_driving, heatmap_source)  
        sparse_motion = self.create_sparse_motions(source_image, heatmap_source, heatmap_driving) 
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        
        out_dict['sparse_motion'] = sparse_motion
        out_dict['sparse_deformed'] = deformed_source
        
        heatmap_representation=heatmap_representation.unsqueeze(1).view(bs,1, -1, h, w)
        deformed_source=deformed_source.unsqueeze(1).view(bs,1, -1, h, w)
        
        input =torch.cat([heatmap_representation, deformed_source], dim=2)  #####
        input = input.view(bs, -1, h, w)   ##([40, 4, 64, 64])
        
        prediction = self.hourglass(input)   ##([40, 68, 64, 64])

        ###dense flow
        deformation=self.flow(prediction)  ##([40, 2, 64, 64])
        deformation = deformation.permute(0, 2, 3, 1)  ##([40, 64, 64, 2])
        #print(deformation.shape)  
        out_dict['deformation'] = deformation
        
        # occulusion map
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))  ##([40, 1, 64, 64])
            #print(occlusion_map.shape)  
            out_dict['occlusion_map'] = occlusion_map        
        
        return out_dict
