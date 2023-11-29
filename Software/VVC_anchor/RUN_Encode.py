# +
import numpy as np
import os
import sys
import glob
from utils import *




seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

qplist=[ "37", "42", "47", "52"]

inputdir='./testing_sequence/' ###You should download the testing sequence and modify the dir.

Inputformat='YUV420' # 'RGB444' OR 'YUV420'
testingdata_name='CFVQA' # 'CFVQA' OR 'VOXCELEB' ## 'CFVQA' OR 'VOXCELEB'  ###You should choose which dataset to be encoded.
if testingdata_name=='CFVQA':
    frames=125
if testingdata_name=='VOXCELEB':
    frames=250
width=256
height=256    




if Inputformat=='RGB444':
    os.makedirs("./experiment/RGB444/",exist_ok=True)     
    os.makedirs("./experiment/RGB444/EncodeResult/",exist_ok=True)     

    for qp in qplist:
        for seq in seqlist:
            original_seq=inputdir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb' 
            os.system("./encode_rgb444.sh "+qp+" "+str(frames)+" "+str(width)+" "+str(height)+" "+original_seq+" "+"./experiment/RGB444/EncodeResult/"+" "+testingdata_name+'_'+seq+" "+' &') 
            
            print(seq+"_"+qp+" submiited")
            
            
            
elif Inputformat=='YUV420':
    os.makedirs("./experiment/YUV420/",exist_ok=True)     
    os.makedirs("./experiment/YUV420/OriYUV/",exist_ok=True)     
    os.makedirs("./experiment/YUV420/EncodeResult/",exist_ok=True)     
    
    for seq in seqlist:
            
        original_seq=inputdir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
        listR,listG,listB=RawReader_planar(original_seq, width, height,frames)

        # wtite ref and cur (rgb444) to file (yuv420)
        oriyuv_path="./experiment/YUV420/OriYUV/"+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_420.yuv'
        f_temp=open(oriyuv_path,'w')            
        for frame_idx in range(0, frames):            

            img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
            img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
            img_input_yuv.tofile(f_temp)
        f_temp.close()     
        
    for qp in qplist:
        for seq in seqlist:            
            oriyuv_path="./experiment/YUV420/OriYUV/"+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_420.yuv'
            os.system("./encode_yuv420.sh "+qp+" "+str(frames)+" "+str(width)+" "+str(height)+" "+oriyuv_path+" "+"./experiment/YUV420/EncodeResult/"+" "+testingdata_name+'_'+seq+" "+' &')   
            
            ########################       
            print(seq+"_"+qp+" submiited")

            
