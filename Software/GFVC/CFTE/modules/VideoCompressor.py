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
# from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# #from compressai.layers import GDN, MaskedConv2d
# #from compressai.models.utils import conv, deconv, update_registered_buffers

import struct
# import torchac

from modules.selfcompressai.entropy_models import EntropyBottleneck, GaussianConditional


# -

class VideoCompressor(nn.Module):
    #def __init__(self, out_channel_mv,IsTraining):
    def __init__(self, out_channel_mv):

        super(VideoCompressor, self).__init__()
        
        self.out_channel_mv = out_channel_mv
        
        #self.istraining = IsTraining
        
        self.entropy_bottleneck = EntropyBottleneck(self.out_channel_mv)
        
        #self.gaussian_conditional = GaussianConditional(None)    
        
    def forward(self, Driving_Feature, Source_Feature):

        fs = Driving_Feature['value'] - Source_Feature['value']


        self.entropy_bottleneck.eval()
        self.entropy_bottleneck.update(force=True)
       

        fs_strings = self.entropy_bottleneck.compress(fs)
    
        return fs_strings 
