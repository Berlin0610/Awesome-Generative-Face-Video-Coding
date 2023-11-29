import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F
import struct, time
from pathlib import Path

from GFVC.CFTE.sync_batchnorm import DataParallelWithCallback
from GFVC.CFTE.modules.util import *
from GFVC.CFTE.modules.generator import OcclusionAwareGenerator ###
from GFVC.CFTE.modules.keypoint_detector import KPDetector ###
from GFVC.CFTE.animate import normalize_kp

def load_CFTE_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'],strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector']) ####

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator


def make_CFTE_prediction(reference_frame, kp_reference, kp_current, generator, relative=False, adapt_movement_scale=False, cpu=False):
        
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    out = generator(reference_frame, kp_reference, kp_norm)
    # summary(generator, reference_frame, kp_reference, kp_norm)
    
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction



