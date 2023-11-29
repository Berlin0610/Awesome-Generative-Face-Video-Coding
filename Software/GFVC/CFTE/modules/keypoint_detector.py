# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn.functional as F
from GFVC.CFTE.modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
from .GDN import GDN
import math
import numpy as np
import cv2


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        #self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(1, 1),padding=pad)

        
        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

            
        ##feature map/heatmap—> latent code (Analysis net)
        self.conv1 = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + num_kp) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(num_kp)
        
        self.conv2 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(num_kp)
                 
        self.conv3 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(num_kp)
        
        self.conv4 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.gdn4 = GDN(num_kp)        
        
    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)  #torch.Size([bs, 35, 64, 64])
        prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, numkp, 32, 32])
        prediction_gdn2=self.gdn2(self.conv2(prediction_gdn1))  #torch.Size([bs, numkp, 16, 16]) 
        prediction_gdn3=self.gdn3(self.conv3(prediction_gdn2))  #torch.Size([bs, numkp, 8, 8])         
        prediction=self.gdn4(self.conv4(prediction_gdn3))  #torch.Size([bs, numkp, 4, 4])  
        
        #print("prediction")
        #print(prediction.shape)         

        out = {'value': prediction}  
        return out    

