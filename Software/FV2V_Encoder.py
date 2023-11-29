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

from GFVC.utils import *
from GFVC.FV2V_utils import *

from arithmetic.value_encoder import *
from arithmetic.value_decoder import *



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
    original_seq=opt.original_seq
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    

    
    ## FV2V
    FV2V_config_path='./GFVC/FV2V/checkpoint/FV2V-256.yaml'
    FV2V_checkpoint_path='./GFVC/FV2V/checkpoint/FV2V-checkpoint.pth.tar'         
    FV2V_Analysis_Model_Detector, FV2V_Analysis_Model_Estimator, FV2V_Synthesis_Model = load_FV2V_checkpoints(FV2V_config_path, FV2V_checkpoint_path, cpu=False)
    modeldir = 'FV2V' 
    model_dirname='./experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
            
##########################
    start=time.time()

    f_org=open(original_seq,'rb')

    listR,listG,listB=RawReader_planar(original_seq,width, height,frames)

    dir_enc =model_dirname+'/enc/'+'/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm      

    kp_path=model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(kp_path,exist_ok=True)     # the frames to be compressed by vtm   


    start=time.time() 

    sum_bits = 0            
    seq_kp_integer = []

    for frame_idx in range(0, frames):            

        frame_idx_str = str(frame_idx).zfill(4)   
        
        if frame_idx in [0]:      # I-frame      
            
            if Iframe_format=='YUV420':
                
                # wtite ref and cur (rgb444) to file (yuv420)
                f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.yuv','w')
                img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                img_input_yuv.tofile(f_temp)
                f_temp.close()            

                os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+QP+" "+str(width)+" "+str(height))   ########################

                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits
                
                #  read the rec frame (yuv420) and convert to rgb444
                rec_ref_yuv=yuv420_to_rgb444(dir_enc+'frame'+frame_idx_str+'_rec.yuv', width, height, 0, 1, False, False) 
                img_rec = rec_ref_yuv[frame_idx]
                img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                 
            
            elif Iframe_format=='RGB444':
                # wtite ref and cur (rgb444) 
                f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
                img_input_rgb.tofile(f_temp)
                f_temp.close()

                os.system("./vtm/encode_rgb444.sh "+dir_enc+'frame'+frame_idx_str+" "+QP+" "+str(width)+" "+str(height))   ########################
                
                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits
                
                f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                  
            
                                  
            with torch.no_grad(): 
                reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                reference = reference.cuda()    # require GPU       

                headpose_image = FV2V_Analysis_Model_Estimator(reference)

                yaw=headpose_image['yaw']
                pitch=headpose_image['pitch']
                roll=headpose_image['roll']
                yaw = headpose_pred_to_degree(yaw)
                pitch = headpose_pred_to_degree(pitch)
                roll = headpose_pred_to_degree(roll)

                rot_mat = get_rotation_matrix(yaw, pitch, roll)            
                rot_mat_list=rot_mat.tolist()
                rot_mat_list=str(rot_mat_list)
                rot_mat_list="".join(rot_mat_list.split())               

                t=headpose_image['t']
                t_list=t.tolist()
                t_list=str(t_list)
                t_list="".join(t_list.split())

                exp=headpose_image['exp']
                exp_list=exp.tolist()
                exp_list=str(exp_list)
                exp_list="".join(exp_list.split())   


                with open(kp_path+'/frame'+frame_idx_str+'.txt','w')as f:
                    f.write(rot_mat_list)
                    f.write('\n'+t_list)  
                    f.write('\n'+exp_list)

                rot_frame=json.loads(rot_mat_list)###torch.Size([1, 3, 3])
                rot_frame= eval('[%s]'%repr(rot_frame).replace('[', '').replace(']', ''))

                t_frame=json.loads(t_list)  ###torch.Size([1, 3])
                t_frame= eval('[%s]'%repr(t_frame).replace('[', '').replace(']', ''))

                exp_frame=json.loads(exp_list)  ###torch.Size([1, 45])
                exp_frame= eval('[%s]'%repr(exp_frame).replace('[', '').replace(']', ''))

                kp_integer=rot_frame+t_frame+exp_frame ###9+3+45=57
                seq_kp_integer.append(kp_integer)        




        else:

            interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
            #print(source_image)        
            interframe = resize(interframe, (width, height))[..., :3]

            with torch.no_grad(): 
                interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

                interframe = interframe.cuda()    # require GPU      


                headpose_image = FV2V_Analysis_Model_Estimator(interframe)

                yaw=headpose_image['yaw']
                pitch=headpose_image['pitch']
                roll=headpose_image['roll']
                yaw = headpose_pred_to_degree(yaw)
                pitch = headpose_pred_to_degree(pitch)
                roll = headpose_pred_to_degree(roll)
                rot_mat = get_rotation_matrix(yaw, pitch, roll)            
                rot_mat_list=rot_mat.tolist()
                rot_mat_list=str(rot_mat_list)
                rot_mat_list="".join(rot_mat_list.split())               

                t=headpose_image['t']
                t_list=t.tolist()
                t_list=str(t_list)
                t_list="".join(t_list.split())

                exp=headpose_image['exp']
                exp_list=exp.tolist()
                exp_list=str(exp_list)
                exp_list="".join(exp_list.split())   



                #frame_idx = str(frame).zfill(4)
                with open(kp_path+'/frame'+frame_idx_str+'.txt','w')as f:
                    f.write(rot_mat_list)
                    f.write('\n'+t_list)  
                    f.write('\n'+exp_list)

                rot_frame=json.loads(rot_mat_list)###torch.Size([1, 3, 3])
                rot_frame= eval('[%s]'%repr(rot_frame).replace('[', '').replace(']', ''))

                t_frame=json.loads(t_list)  ###torch.Size([1, 3])
                t_frame= eval('[%s]'%repr(t_frame).replace('[', '').replace(']', ''))

                exp_frame=json.loads(exp_list)  ###torch.Size([1, 45])
                exp_frame= eval('[%s]'%repr(exp_frame).replace('[', '').replace(']', ''))

                kp_integer=rot_frame+t_frame+exp_frame ###9+3+45=57
                #print(len(kp_value_jocobi))
                seq_kp_integer.append(kp_integer)        



    rec_sem=[]
    for frame in range(1,frames):
        frame_idx = str(frame).zfill(4)
        if frame==1:
            rec_sem.append(seq_kp_integer[0])

            ### residual
            kp_difference=(np.array(seq_kp_integer[frame])-np.array(seq_kp_integer[frame-1])).tolist()
            ## quantization

            kp_difference=[i*Qstep for i in kp_difference]
            kp_difference= list(map(round, kp_difference[:]))

            frame_idx = str(frame).zfill(4)
            bin_file=kp_path+'/frame'+str(frame_idx)+'.bin'

            final_encoder_expgolomb(kp_difference,bin_file)     

            bits=os.path.getsize(bin_file)*8
            sum_bits += bits          

            #### decoding for residual
            res_dec = final_decoder_expgolomb(bin_file)
            res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

            ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

            res_difference_dec=[i/Qstep for i in res_difference_dec]

            rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()

            rec_sem.append(rec_semantics)

        else:

            ### residual
            kp_difference=(np.array(seq_kp_integer[frame])-np.array(rec_sem[frame-1])).tolist()

            ## quantization
            kp_difference=[i*Qstep for i in kp_difference]
            kp_difference= list(map(round, kp_difference[:]))

            frame_idx = str(frame).zfill(4)
            bin_file=kp_path+'/frame'+str(frame_idx)+'.bin'

            final_encoder_expgolomb(kp_difference,bin_file)     

            bits=os.path.getsize(bin_file)*8
            sum_bits += bits          

            #### decoding for residual
            res_dec = final_decoder_expgolomb(bin_file)
            res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

            ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame
            res_difference_dec=[i/Qstep for i in res_difference_dec]
            rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()
            rec_sem.append(rec_semantics)


    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   




