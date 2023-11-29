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

qplist=[ "37", "42", "47", "52"]


Inputformat='YUV420' # 'RGB444' OR 'YUV420'
testingdata_name='VOXCELEB' # 'CFVQA' OR 'VOXCELEB'
if testingdata_name=='CFVQA':
    frames=125
if testingdata_name=='VOXCELEB':
    frames=250
    
input_bin_file_path='./experiment/'+Inputformat+'/EncodeResult/'
save_path='./experiment/'+Inputformat+'/'


totalResult=np.zeros((len(seqlist)+1,len(qplist)))

for seqIdx, seq in enumerate(seqlist):
    for qpIdx, QP in enumerate(qplist):  
   
            
        path = input_bin_file_path+testingdata_name+'_'+str(seq)+'_qp'+str(QP)+'.bin'
        overall_bits=os.path.getsize(path)*8 
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

np.savetxt(save_path+testingdata_name+'_resultBit.txt', totalResult, fmt = '%.5f')                    
        

    
    
