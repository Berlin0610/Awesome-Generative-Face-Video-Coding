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

from GFVC.FV2V.animate import normalize_kp
from GFVC.FV2V.sync_batchnorm import DataParallelWithCallback
from GFVC.FV2V.modules.generator import OcclusionAwareGenerator ###
from GFVC.FV2V.modules.keypoint_detector import KPDetector, HEEstimator



##########
def make_FV2V_prediction(reference_frame, kp_reference, kp_current, generator, relative=False,adapt_movement_scale=False,
                    estimate_jacobian=False, cpu=False, free_view=False, yaw=0, pitch=0, roll=0):
        
    
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=(estimate_jacobian & relative),
                           adapt_movement_scale=adapt_movement_scale) 
    
    out = generator(reference_frame,kp_source=kp_reference,kp_driving= kp_norm)
    
    
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction


def load_FV2V_checkpoints(config_path, checkpoint_path, cpu=False):

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
        
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    if not cpu:
        he_estimator.cuda()        

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'],strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector']) ####
    he_estimator.load_state_dict(checkpoint['he_estimator'])
        
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        he_estimator = DataParallelWithCallback(he_estimator)

    generator.eval()
    kp_detector.eval()
    he_estimator.eval()
    
    return kp_detector, he_estimator, generator






def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)
    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)
    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)
    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)
    rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

    return rot_mat


def keypoint_transformation_source(kp_canonical, he, estimate_jacobian=False, free_view=False, yaw=0, pitch=0, roll=0):
    kp = kp_canonical['value']
    
    if not free_view:
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).cuda()
        else:
            yaw = he['yaw']
            yaw = headpose_pred_to_degree(yaw)
        if pitch is not None:
            pitch = torch.tensor([pitch]).cuda()
        else:
            pitch = he['pitch']
            pitch = headpose_pred_to_degree(pitch)
        if roll is not None:
            roll = torch.tensor([roll]).cuda()
        else:
            roll = he['roll']
            roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    #print(kp_rotated.shape)
    
    
    t, exp = he['t'], he['exp']
    # keypoint translation

    t = t.unsqueeze_(1)
    t = t.repeat(1, kp.shape[1], 1)
    #t = t.repeat_interleave(kp.shape[1], dim=1)
    
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}


def keypoint_transformation(kp_canonical, he, estimate_jacobian=False, free_view=False, yaw=0, pitch=0, roll=0):
    kp = kp_canonical['value']


    rot_mat = he['rot_mat']
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    #print(kp_rotated.shape)
    
    
    t, exp = he['t'], he['exp']
    # keypoint translation

    # t = t.unsqueeze_(1)
    t = t.unsqueeze(1)
    
    t = t.repeat(1, kp.shape[1], 1)
    #t = t.repeat_interleave(kp.shape[1], dim=1)
    
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}






