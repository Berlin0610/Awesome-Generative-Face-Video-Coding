# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from torch import nn
import torch.nn.functional as F
import torch
from GFVC.CFTE.modules.util import * 
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms


def warp(x, flo):

    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow  
    if x=img2, flow(pre,cur) x warp flow==pre
    if x=img1, flow(cur,pre) x warp flow==cur
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    
    #x = x
    
    if torch.cuda.is_available():  
        x = x.cuda()#to('cuda:0')
        
        grid = grid.cuda()  #to('cuda:0')
        
    #flo = flo.cuda().data.cpu()
    flo = flo.permute(0,3,1,2)#from B,H,W,2 -> B,2,H,W
    
    #pixel flow motion
    vgrid = Variable(grid) + flo # B,2,H,W

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 
    vgrid = vgrid.permute(0,2,3,1)     #from B,2,H,W -> B,H,W,2

    output = F.grid_sample(x, vgrid,mode="bilinear",padding_mode="zeros")

    return output
