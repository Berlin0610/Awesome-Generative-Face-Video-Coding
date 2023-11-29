import os, sys
import yaml
from tqdm import tqdm
import imageio
import numpy as np
import scipy.io as io
import json
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB

def splitlist(list): 
    alist = []
    a = 0 
    for sublist in list:
        try: #用try来判断是列表中的元素是不是可迭代的，可以迭代的继续迭代
            for i in sublist:
                alist.append (i)
        except TypeError: #不能迭代的就是直接取出放入alist
            alist.append(sublist)
    for i in alist:
        if type(i) == type([]):#判断是否还有列表
            a =+ 1
            break
    if a==1:
        return printlist(alist) #还有列表，进行递归
    if a==0:
        return alist  
    
    
###只能读取yuv420 8bit 
def yuv420_to_rgb444(yuvfilename, W, H, startframe, totalframe, show=False, out=False):
    # 从第startframe（含）开始读（0-based），共读totalframe帧
    arr = np.zeros((totalframe,H,W,3), np.uint8)
    
    plt.ion()
    with open(yuvfilename, 'rb') as fp:
        seekPixels = startframe * H * W * 3 // 2
        fp.seek(8 * seekPixels) #跳过前startframe帧
        for i in range(totalframe):
            #print(i)
            oneframe_I420 = np.zeros((H*3//2,W),np.uint8)
            for j in range(H*3//2):
                for k in range(W):
                    oneframe_I420[j,k] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
            oneframe_RGB = cv2.cvtColor(oneframe_I420,cv2.COLOR_YUV2BGR_I420)
            if show:
                plt.imshow(oneframe_RGB)
                plt.show()
                plt.pause(5)
            if out:
                outname = yuvfilename[:-4]+'_'+str(startframe+i)+'.png'
                cv2.imwrite(outname,oneframe_RGB[:,:,::-1])
            arr[i] = oneframe_RGB
    return arr