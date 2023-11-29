# +
import numpy as np
import os
import sys
import glob
from utils import *




seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

qplist=[ "37", "42", "47", "52"]


Inputformat='YUV420' # 'RGB444' OR 'YUV420'
testingdata_name='CFVQA' # 'CFVQA' OR 'VOXCELEB'
if testingdata_name=='CFVQA':
    frames=125
if testingdata_name=='VOXCELEB':
    frames=250
width=256
height=256    


if Inputformat=='RGB444':
    os.makedirs("./experiment/RGB444/Dec/",exist_ok=True)     
    for qp in qplist:
        for seq in seqlist:
            os.system("./decode_rgb444.sh "+testingdata_name+'_'+seq+" "+qp+" "+"./experiment/RGB444/EncodeResult/"+" "+"./experiment/RGB444/Dec/"+' &')   ########################
            print(seq+"_"+qp+" submiited")
            

elif Inputformat=='YUV420':
    os.makedirs("./experiment/YUV420/Dec/",exist_ok=True)     
    
    os.makedirs("./experiment/YUV420/YUVtoRGB/",exist_ok=True)     
    
    for qp in qplist:
        for seq in seqlist:
            os.system("./decode_yuv420.sh "+testingdata_name+'_'+seq+" "+qp+" "+"./experiment/YUV420/EncodeResult/"+" "+"./experiment/YUV420/Dec/")   ########################
            print(seq+"_"+qp+" submiited")    
    
    
            decode_seq="./experiment/YUV420/YUVtoRGB/"+testingdata_name+'_'+seq+'_qp'+str(qp)+'_dec.rgb'         
            f_dec=open(decode_seq,'w') 

            #  read the rec frame (yuv420) and convert to rgb444
            rec_ref_yuv=yuv420_to_rgb444("./experiment/YUV420/Dec/"+testingdata_name+'_'+seq+"_qp"+qp+'_dec.yuv', width, height, 0, frames, False, False) 
            
            for frame_idx in range(0, frames):            
                img_rec = rec_ref_yuv[frame_idx]
                img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
                img_rec.tofile(f_dec)     
            f_dec.close()     



