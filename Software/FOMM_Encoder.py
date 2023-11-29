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
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *

from GFVC.utils import *
from GFVC.FOMM_utils import *



    

if __name__ == "__main__":
   
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=64, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--Iframe_QP", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--Iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    original_seq=opt.original_seq
    Iframe_format=opt.Iframe_format

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    
    ## FOMM
    FOMM_config_path='./GFVC/FOMM/checkpoint/FOMM-256.yaml'
    FOMM_checkpoint_path='./GFVC/FOMM/checkpoint/FOMM-checkpoint.pth.tar'         
    FOMM_Analysis_Model, FOMM_Synthesis_Model = load_FOMM_checkpoints(FOMM_config_path, FOMM_checkpoint_path, cpu=False)
    modeldir = 'FOMM' 
    model_dirname='./experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
    

###################
    start=time.time()
    driving_kp =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'    
    os.makedirs(driving_kp,exist_ok=True)     # the frames to be compressed by vtm                 

    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm                      

    listR,listG,listB=RawReader_planar(original_seq,width, height,frames)
    f_org=open(original_seq,'rb')

    seq_kp_integer = []
    sum_bits = 0
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

                kp_reference = FOMM_Analysis_Model(reference) ################ 

                kp_value = kp_reference['value']
                kp_value_list = kp_value.tolist()
                kp_value_list = str(kp_value_list)
                kp_value_list = "".join(kp_value_list.split())

                kp_jacobian=kp_reference['jacobian'] 
                kp_jacobian_list=kp_jacobian.tolist()
                kp_jacobian_list=str(kp_jacobian_list)
                kp_jacobian_list="".join(kp_jacobian_list.split())            


                with open(driving_kp+'/frame'+frame_idx_str+'.txt','w')as f:
                    f.write(kp_value_list)  
                    f.write('\n'+kp_jacobian_list)  

                kp_value_frame=json.loads(kp_value_list)###20
                kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                kp_jacobian_frame=json.loads(kp_jacobian_list)  ###40
                kp_jacobian_frame= eval('[%s]'%repr(kp_jacobian_frame).replace('[', '').replace(']', ''))
                kp_value_jocobi=kp_value_frame+kp_jacobian_frame ###20+40
                seq_kp_integer.append(kp_value_jocobi)                    


        else:

            interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
            interframe = resize(interframe, (width, height))[..., :3]

            with torch.no_grad(): 
                interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                interframe = interframe.cuda()    # require GPU          

                ###extraction
                kp_image = FOMM_Analysis_Model(interframe) ################
                kp_value = kp_image['value']
                kp_value_list = kp_value.tolist()
                kp_value_list = str(kp_value_list)
                kp_value_list = "".join(kp_value_list.split())

                kp_jacobian=kp_image['jacobian'] 
                kp_jacobian_list=kp_jacobian.tolist()
                kp_jacobian_list=str(kp_jacobian_list)
                kp_jacobian_list="".join(kp_jacobian_list.split())            

                with open(driving_kp+'/frame'+frame_idx_str+'.txt','w')as f:
                    f.write(kp_value_list)  
                    f.write('\n'+kp_jacobian_list)  

                kp_value_frame=json.loads(kp_value_list)###20
                kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                kp_jacobian_frame=json.loads(kp_jacobian_list)  ###40
                kp_jacobian_frame= eval('[%s]'%repr(kp_jacobian_frame).replace('[', '').replace(']', ''))
                kp_value_jocobi=kp_value_frame+kp_jacobian_frame ###20+40
                #print(len(kp_value_jocobi))
                seq_kp_integer.append(kp_value_jocobi)



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
            bin_file=driving_kp+'/frame'+str(frame_idx)+'.bin'

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
            bin_file=driving_kp+'/frame'+str(frame_idx)+'.bin'

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






