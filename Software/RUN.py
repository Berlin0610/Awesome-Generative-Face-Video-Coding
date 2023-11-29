# +
import os
import sys


seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

qplist=[ "22", "32", "42", "52"]

    
Sequence_dir='testing_sequence'  ###You should download the testing sequence and modify the dir.

testingdata_name='VOXCELEB' ## 'CFVQA' OR 'VOXCELEB'  ###You should choose which dataset to be encoded.
if testingdata_name=='CFVQA':
    frames=125
if testingdata_name=='VOXCELEB':
    frames=250
    
height=256
width=256

Model='FV2V'             ## 'FV2V' OR 'FOMM' OR 'CFTE' ###You should choose which GFVC model to be uesed.
if Model=='FV2V':
    quantization_factor=256
if Model=='CFTE':
    quantization_factor=4
if Model=='FOMM':
    quantization_factor=64
    
Mode='Encoder'           ## "Encoder" OR 'Decoder'   ###You need to define whether to encode or decode a sequence.
Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444' ###You need to define what color format to use for encoding the first frame.



for qp in qplist:
    for seq in seqlist:
        original_seq=Sequence_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
        os.system("./RUN.sh "+Model+" "+Mode+" "+original_seq+" "+str(frames)+" "+str(quantization_factor)+" "+str(qp)+" "+str(Iframe_format))  
        
        print(Model+"_"+Mode+"_"+seq+"_"+qp+" Finished")
