# +
# get file size in python
import os
import numpy as np

def get_all_file(dir_path):
    global files
    for filepath in os.listdir(dir_path):
        tmp_path = os.path.join(dir_path,filepath)
        if os.path.isdir(tmp_path):
            get_all_file(tmp_path)
        else:
            files.append(tmp_path)
    return files

def calc_files_size(files_path):
    files_size = 0
    for f in files_path:
        files_size += os.path.getsize(f)
    return files_size



seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

qplist=[ "22", "32", "42", "52"]


testingdata_name='VOXCELEB' # 'CFVQA' OR 'VOXCELEB'
if testingdata_name=='CFVQA':
    frames=125
if testingdata_name=='VOXCELEB':
    frames=250
    

Model='CFTE'             ## 'FV2V' OR 'FOMM' OR 'CFTE'
Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444'


input_bin_file_path='./experiment/'+str(Model)+'/Iframe_'+str(Iframe_format)
save_path='./experiment/'+str(Model)+'/Iframe_'+str(Iframe_format)+"/resultBit/"


totalResult=np.zeros((len(seqlist)+1,len(qplist)))

for seqIdx, seq in enumerate(seqlist):
    for qpIdx, QP in enumerate(qplist):  
        overall_bits=0
        for frame_idx in range(0, frames):            

            frame_idx_str = str(frame_idx).zfill(4)           
   
            if frame_idx in [0]:      # I-frame bitstream 
        
                Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_QP'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'
                overall_bits=overall_bits+os.path.getsize(Iframepath)*8 
                
            else:  ## Feature bitstream
                
                interpath=input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_QP'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'      
                overall_bits=overall_bits+os.path.getsize(interpath)*8 

        print(overall_bits)
        totalResult[seqIdx][qpIdx]=overall_bits   
        

        
        
# summary the bitrate
for qp in range(len(qplist)):
    for seq in range(len(seqlist)):
        totalResult[-1][qp]+=totalResult[seq][qp]
    totalResult[-1][qp] /= len(seqlist)

np.set_printoptions(precision=5)
totalResult = totalResult/1000
seqlength = frames/25
totalResult = totalResult/seqlength

np.savetxt(save_path+testingdata_name+'_'+'resultBit.txt', totalResult, fmt = '%.5f')                    
        
