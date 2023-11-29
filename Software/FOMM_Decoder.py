import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch

import json
import time
import cv2

from GFVC.utils import *
from GFVC.FOMM_utils import *

from arithmetic.value_encoder import *
from arithmetic.value_decoder import *



    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
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

    
    ## FOMM
    FOMM_config_path='./GFVC/FOMM/checkpoint/FOMM-256.yaml'
    FOMM_checkpoint_path='./GFVC/FOMM/checkpoint/FOMM-checkpoint.pth.tar'         
    FOMM_Analysis_Model, FOMM_Synthesis_Model = load_FOMM_checkpoints(FOMM_config_path, FOMM_checkpoint_path, cpu=False)
    modeldir = 'FOMM' 
    model_dirname='./experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
###################################################

    driving_kp =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'   

    dir_dec=model_dirname+'/dec/'
    os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
    decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'

    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm     

    dir_bit=model_dirname+'/resultBit/'
    os.makedirs(dir_bit,exist_ok=True)        


    f_dec=open(decode_seq,'w') 
    seq_kp_integer=[]

    start=time.time() 
    gene_time = 0
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

                kp_reference = FOMM_Analysis_Model(reference) 

                ####
                kp_value = kp_reference['value']
                kp_value_list = kp_value.tolist()
                kp_value_list = str(kp_value_list)
                kp_value_list = "".join(kp_value_list.split())

                kp_jacobian=kp_reference['jacobian'] 
                kp_jacobian_list=kp_jacobian.tolist()
                kp_jacobian_list=str(kp_jacobian_list)
                kp_jacobian_list="".join(kp_jacobian_list.split())  

                kp_value_frame=json.loads(kp_value_list)###20
                kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                kp_jacobian_frame=json.loads(kp_jacobian_list)  ###40
                kp_jacobian_frame= eval('[%s]'%repr(kp_jacobian_frame).replace('[', '').replace(']', ''))
                kp_integer=kp_value_frame+kp_jacobian_frame ###20+40 
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

            kp_previous=seq_kp_integer[frame_idx-1]    #json.loads(str(seq_kp_integer[frame_idx-1]))      
            kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', '').replace("'", ""))  

            kp_integer,kp_value,kp_jocobi=listformat_kp_jocobi_FOMM(kp_previous, kp_difference_dec) #######
            seq_kp_integer.append(kp_integer)                    

            dict={}
            kp_value=json.loads(kp_value)
            kp_current_value=torch.Tensor(kp_value).to('cuda:0')          
            dict['value']=kp_current_value  

            kp_jocobi=json.loads(kp_jocobi)
            kp_current_jocobi=torch.Tensor(kp_jocobi).to('cuda:0')          
            dict['jacobian']=kp_current_jocobi  

            kp_current=dict   


            # generated frame

            gene_start = time.time()

            prediction = make_FOMM_prediction(reference, kp_current, kp_reference, FOMM_Synthesis_Model) #######################

            gene_end = time.time()
            gene_time += gene_end - gene_start
            pre=(prediction*255).astype(np.uint8)  
            pre.tofile(f_dec)                              

            frame_index=str(frame_idx).zfill(4)
            bin_save=driving_kp+'/frame'+frame_index+'.bin'
            bits=os.path.getsize(bin_save)*8
            sum_bits += bits

    f_dec.close()     
    end=time.time()

    print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=gene_time   
    
    np.savetxt(dir_bit+seq+'_QP'+str(QP)+'.txt', totalResult, fmt = '%.5f')            


