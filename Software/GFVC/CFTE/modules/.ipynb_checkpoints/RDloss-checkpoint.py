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

# +
import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter

# # new added start
# import compressai
# from compressai.zoo import models
# # new added end
# from compressai.models import CompressionModel
# from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# #from compressai.layers import GDN, MaskedConv2d
# #from compressai.models.utils import conv, deconv, update_registered_buffers

import struct
# import torchac

#from modules.selfcompressai.entropy_models import EntropyBottleneck, GaussianConditional
# -

class VideoCompressor(nn.Module):
    def __init__(self, out_channel_mv):
        super(VideoCompressor, self).__init__()
        
        self.out_channel_mv = out_channel_mv
                
        self.entropy_bottleneck = EntropyBottleneck(self.out_channel_mv)
        
        #self.gaussian_conditional = GaussianConditional(None)    
        
    def forward(self, Driving_Feature, Source_Feature):

        fs = Driving_Feature['value'] - Source_Feature['value']
        
        bs, c, h , w= fs.shape  ## bs,1,4,4

        fs_hat, fs_likelihoods = self.entropy_bottleneck(fs)
        #print(fs_hat)

        total_bits = torch.log(fs_likelihoods).sum() / (-math.log(2))     
        #print(total_bits)       

        quant_driving = fs_hat + Source_Feature['value']
                
        quant_driving = {'value': quant_driving}  ####
        
        return total_bits, quant_driving
