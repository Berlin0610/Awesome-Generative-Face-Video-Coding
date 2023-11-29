import torch
from torch import nn
import torch.nn.functional as F
from GFVC.CFTE.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from GFVC.CFTE.modules.dense_motion import DenseMotionNetwork
from GFVC.CFTE.modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from .GDN import GDN
import math
from GFVC.CFTE.modules.flowwarp import *


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()


        self.temperature =0.1

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return warp(inp, deformation) #F.grid_sample(inp, deformation)  #########

    def forward(self, source_image,heatmap_source,heatmap_driving):  
        
        # Encoding (downsampling) part
        out = self.first(source_image) 
        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}

        dense_motion = self.dense_motion_network(source_image=source_image,heatmap_source=heatmap_source,
                                                 heatmap_driving=heatmap_driving)


        occlusion_map = dense_motion['occlusion_map']  #64*64*1
        output_dict['occlusion_map'] = occlusion_map  
        
        deformation = dense_motion['deformation'] #64*64*2
        output_dict['deformation'] = deformation
        
        deformed_sparse_source = dense_motion['sparse_deformed'] #64*64*3
        output_dict['sparse_deformed'] = deformed_sparse_source
        
        sparse_motion = dense_motion['sparse_motion']  ###8*8*2
        output_dict['sparse_motion'] = sparse_motion

        
        output_dict["deformed"] = self.deform_input(source_image, deformation)   #### deformation 64*64 interpolate 256*256

        out = self.deform_input(out, deformation)

#         ###for a comparison about dense flow
#         out_dense = self.bottleneck(out)
#         for i in range(len(self.up_blocks)):
#             out_dense = self.up_blocks[i](out_dense)
#         out_dense = self.final(out_dense)
#         out_dense = F.sigmoid(out_dense)        
#         output_dict["deformed"] = out_dense  

        ###real-part
        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
