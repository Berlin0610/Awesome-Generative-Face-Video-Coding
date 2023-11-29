import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import imageio
from skimage import img_as_ubyte
import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch

import time
import random
import pandas as pd
import collections
import itertools
from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import math
import torch.nn.functional as F


from arithmetic.value_encoder import *
from arithmetic.value_decoder import *

from GFVC.utils import *
from GFVC.FV2V_utils import *





    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=256, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--Iframe_QP", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--Iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    Iframe_format=opt.Iframe_format    
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    

    ## FV2V
    FV2V_config_path='./GFVC/FV2V/checkpoint/FV2V-256.yaml'
    FV2V_checkpoint_path='./GFVC/FV2V/checkpoint/FV2V-checkpoint.pth.tar'         
    FV2V_Analysis_Model_Detector, FV2V_Analysis_Model_Estimator, FV2V_Synthesis_Model = load_FV2V_checkpoints(FV2V_config_path, FV2V_checkpoint_path, cpu=False)
    modeldir = 'FV2V' 
    model_dirname='./experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   

      
############################################                   
            
    driving_kp = model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'

    dir_dec=model_dirname+'/dec/'

    os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
    decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'

    dir_enc = model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm       
            
    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)         
    
    f_dec=open(decode_seq,'w') 



    seq_kp_integer=[]               #  the quantilized compact feature list of the whole sequence

    start=time.time() 
    generate_time = 0

    sum_bits = 0


    for frame_idx in range(0, frames):            
        frame_idx_str = str(frame_idx).zfill(4)   
        if frame_idx in [0]:      # I-frame                      
          
            if Iframe_format=='YUV420':
                os.system("./vtm/decode.sh "+dir_enc+'frame'+frame_idx_str)
                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits

                #  read the rec frame (yuv420) and convert to rgb444
                rec_ref_yuv=yuv420_to_rgb444(dir_enc+'frame'+frame_idx_str+'_dec.yuv', width, height, 0, 1, False, False) 
                img_rec = rec_ref_yuv[frame_idx]
                img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
                img_rec.tofile(f_dec)                         
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      

            elif Iframe_format=='RGB444':
                os.system("./vtm/decode_rgb444.sh "+dir_enc+'frame'+frame_idx_str)
                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits

                f_temp=open(dir_enc+'frame'+frame_idx_str+'_dec.rgb','rb')
                img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                img_rec.tofile(f_dec) 
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1      
                
            with torch.no_grad(): 
                reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                reference = reference.cuda()    # require GPU


                kp_canonical = FV2V_Analysis_Model_Detector(reference)  ####reference 
                he_source = FV2V_Analysis_Model_Estimator(reference)  
                kp_reference = keypoint_transformation_source(kp_canonical, he_source, estimate_jacobian=False,
                                                           free_view=False, yaw=0, pitch=0, roll=0) ###### I frame                         


                kp_cur = FV2V_Analysis_Model_Estimator(reference)  
                ####
                ###  yaw+pttch+roll-->rot mat
                yaw=kp_cur['yaw']
                pitch=kp_cur['pitch']
                roll=kp_cur['roll']
                yaw = headpose_pred_to_degree(yaw)
                pitch = headpose_pred_to_degree(pitch)
                roll = headpose_pred_to_degree(roll)
                kp_rot = get_rotation_matrix(yaw, pitch, roll)            
                kp_rot_list=kp_rot.tolist()
                kp_rot_list=str(kp_rot_list)
                kp_rot_list="".join(kp_rot_list.split())                               

                kp_t=kp_cur['t'] 
                kp_t_list=kp_t.tolist()
                kp_t_list=str(kp_t_list)
                kp_t_list="".join(kp_t_list.split())  

                kp_exp=kp_cur['exp'] 
                kp_exp_list=kp_exp.tolist()
                kp_exp_list=str(kp_exp_list)
                kp_exp_list="".join(kp_exp_list.split())                          

                rot_frame=json.loads(kp_rot_list)###torch.Size([1, 3, 3])
                rot_frame= eval('[%s]'%repr(rot_frame).replace('[', '').replace(']', ''))
                t_frame=json.loads(kp_t_list)  ###torch.Size([1, 3])
                t_frame= eval('[%s]'%repr(t_frame).replace('[', '').replace(']', ''))
                exp_frame=json.loads(kp_exp_list)  ###torch.Size([1, 45])
                exp_frame= eval('[%s]'%repr(exp_frame).replace('[', '').replace(']', ''))
                kp_integer=rot_frame+t_frame+exp_frame ###9+3+45=57
                kp_integer=str(kp_integer)

                seq_kp_integer.append(kp_integer)


        else:
            frame_index=str(frame_idx).zfill(4)
            bin_save=driving_kp+'/frame'+frame_index+'.bin'            
            kp_dec = final_decoder_expgolomb(bin_save)

            ## decoding residual
            kp_difference = data_convert_inverse_expgolomb(kp_dec)
            ## inverse quanzation
            kp_difference_dec=[i/Qstep for i in kp_difference]
            kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  

            kp_previous=seq_kp_integer[frame_idx-1]
            kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', '').replace("'", ""))   

            kp_integer,kp_mat_value,kp_t_value,kp_exp_value=listformat_kp_mat_exp_FV2V(kp_previous, kp_difference_dec) #######
            seq_kp_integer.append(kp_integer)


            dict={}                  
            kp_mat_value=json.loads(kp_mat_value)
            kp_current_mat=torch.Tensor(kp_mat_value).to('cuda:0')          
            dict['rot_mat']=kp_current_mat  

            kp_t_value=json.loads(kp_t_value)
            kp_current_t=torch.Tensor(kp_t_value).to('cuda:0')          
            dict['t']=kp_current_t  

            kp_exp_value=json.loads(kp_exp_value)
            kp_current_exp=torch.Tensor(kp_exp_value).to('cuda:0')          
            dict['exp']=kp_current_exp  

            kp_current_matrix=dict  

            kp_current = keypoint_transformation(kp_canonical, kp_current_matrix, estimate_jacobian=False, 
                                                 free_view=False, yaw=0, pitch=0, roll=0)    

            # generated frame
            generate_start = time.time()    

            prediction = make_FV2V_prediction(reference, kp_reference, kp_current, FV2V_Synthesis_Model) #######################

            generate_end = time.time()
            generate_time += generate_end - generate_start                        

            pre=(prediction*255).astype(np.uint8)  
            pre.tofile(f_dec)                              

            frame_index=str(frame_idx).zfill(4)
            bin_save=driving_kp+'/frame'+frame_index+'.bin'
            bits=os.path.getsize(bin_save)*8
            sum_bits += bits

    f_dec.close()     
    end=time.time()                    


    print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,generate_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=generate_time   
    
    np.savetxt(dir_bit+seq+'_QP'+str(QP)+'.txt', totalResult, fmt = '%.5f')            



                    
                    
                    
                    
                    
    
    