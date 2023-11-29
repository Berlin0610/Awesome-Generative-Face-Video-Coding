import sys
from arithmetic.arithmeticcoding import *
from arithmetic.ppmmodel import *
import numpy as np
import random
import pandas as pd
import collections
import itertools
import json
import os
import struct
from collections import Counter
from arithmetic.expgolomb_encode_decode import *

############################# unary code#######################################
##对解码得到的数字list进行分割函数
def list_deal(list_ori,p):   
    list_new=[]				#处理后的列表，是一个二维列表
    list_short=[]			#用于存放每一段列表
    for i in list_ori:
        if i!=p:		
            list_short.append(i)
        else:
            list_new.append(list_short)
            list_short=[]
    list_new.append(list_short)   #最后一段遇不到切割标识，需要手动放入
    return list_new

##对手动切割的list进行相关的数字个数统计函数
def  count_list(std:list,tongji):
    name=Counter()
    for  num in std:
        name[num]+=1
    #print(name[tongji])
    return name[tongji]

## final decode: input: bin file; output: the 0/1 value
def final_decoder_unary(inputfile,MODEL_ORDER = 0):
    # Must be at least -1 and match ppm-compress.py. Warning: Exponential memory usage at O(257^n).
    # Perform file decompression
    with open(inputfile, "rb") as inp:
        bitin = BitInputStream(inp)  #arithmeticcoding.

        dec = ArithmeticDecoder(256, bitin) ##############arithmeticcoding.
        model = PpmModel(MODEL_ORDER, 3, 2) #######ppmmodel.
        history = []
        
        datares_rec=[]
                    
        while True:
            symbol = decode_symbol(dec, model, history)
            if symbol ==2:
                break

            model.increment_contexts(history, symbol)
            datares_rec.append(symbol)
            if model.model_order >= 1:
                # Prepend current symbol, dropping oldest symbol if necessary
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, symbol) ####
        return datares_rec

###对解码得到的0/1字符串进行以0为分割符号的list切割，并且统计每个子list中1的个数，输出原始对应的值，
###并且进行相关的数字反非负值处理，变成 inter residual

def data_convert_inverse_unary(datares_rec):


     ##按照数字0进行对list进行截断，并且统计每个sub_list里面1的个数，还原出真实的数字
    list_new=list_deal(datares_rec,0) #按照0进行切割
    #print(list_new)
    #print(len(list_new))

    true_ae_number=[]
    for subnum in range(len(list_new)-1):
        num=count_list(std=list_new[subnum],tongji=1)
        #print(num)
        true_ae_number.append(num)
    #print(true_ae_number)

    ##进行相关的恢复
    for i in range(len(true_ae_number)):
        true_ae_number[i] = true_ae_number[i]-1
    #print(true_ae_number)

    #把解码后的残差变会原先的数值 （0，1，2，3，4——》0，1，-1，2，-2)
    for ii in range(len(true_ae_number)):
        if true_ae_number[ii] ==0:
            true_ae_number[ii]=0
        elif  true_ae_number[ii] >0 and true_ae_number[ii] %2 ==0:
            true_ae_number[ii]=-(int(true_ae_number[ii]/2))
        else:
            true_ae_number[ii]=int((true_ae_number[ii]+1)/2)
    #print(true_ae_number)
    return true_ae_number

################################ 0-order exponential coding###############
## final decode: input: bin file; output: the 0/1 value
def final_decoder_expgolomb(inputfile,MODEL_ORDER = 0):
    # Must be at least -1 and match ppm-compress.py. Warning: Exponential memory usage at O(257^n).
    # Perform file decompression
    with open(inputfile, "rb") as inp:
        bitin = BitInputStream(inp)  #arithmeticcoding.

        dec = ArithmeticDecoder(256, bitin) ##############arithmeticcoding.
        model = PpmModel(MODEL_ORDER, 3, 2) #######ppmmodel.
        history = []
        
        datares_rec=[]
                    
        while True:
            symbol = decode_symbol(dec, model, history)
            if symbol ==2:
                break

            model.increment_contexts(history, symbol)
            datares_rec.append(symbol)
            if model.model_order >= 1:
                # Prepend current symbol, dropping oldest symbol if necessary
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, symbol) ####
        return datares_rec

###对0阶指数格伦布的编码方式，对这一串二进制字符串进行有效的切分
###并且进行相关的数字反非负值处理，变成 inter residual

def data_convert_inverse_expgolomb(datares_rec):


    ##按照0-order所定义的解码方式进行数字划分切割
    list_new= expgolomb_split(datares_rec)
    #print(list_new)
    #print(len(list_new))
    
    true_ae_number=[]
    for subnum in range(len(list_new)):
        num=exponential_golomb_decode(list_new[subnum])
        #print(num)
        true_ae_number.append(num)
    #print(true_ae_number)

    #把解码后的残差变会原先的数值 （0，1，2，3，4——》0，1，-1，2，-2)
    for ii in range(len(true_ae_number)):
        if true_ae_number[ii] ==0:
            true_ae_number[ii]=0
        elif  true_ae_number[ii] >0 and true_ae_number[ii] %2 ==0:
            true_ae_number[ii]=-(int(true_ae_number[ii]/2))
        else:
            true_ae_number[ii]=int((true_ae_number[ii]+1)/2)
    #print(true_ae_number)
    return true_ae_number




############################################

def listformat_adptive_CFTE(refframetensor, true_ae_number, num_channel,N_size):    
    length=N_size*N_size ##############
    listnum=N_size  ###########
    slidelist=int(length/listnum)
    #print(slidelist)
    ### 第二帧（帧间重建第一帧 based on VVC frame）    

    reallatentvalue=(np.array(refframetensor)+np.array(true_ae_number)).tolist()
    #print(len(reallatentvalue))
    #print(reallatentvalue)

    #按照模型所需格式重组
    latentformat=[]
    
    for channel_size in range(num_channel):
        
        latentformat_channel=[]
        for slideformat in range(0,slidelist):
            latentformatslide=reallatentvalue[slideformat*listnum:(slideformat+1)*listnum]
            latentformat_channel.extend([latentformatslide])
        del reallatentvalue[0:length]
        #latentformat=[[latentformat]]
        latentformat.extend([latentformat_channel])
    #print(latentformat)
    latentformat=[latentformat]
    
    ###保存恢复的真实的数据格式
    latentformat=str(latentformat)
    latentformat="".join(latentformat.split())
        
    return latentformat



def listformat_kp_mat_exp_FV2V(refframetensor, true_ae_number):
    
    group=57
    ##按照模型所需格式进行重组的相关参数设置
    listnum=3
    ##对恢复得到的dataRec进行切片，每16个组成一组，并进行帧间补偿    

    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    reallatentvalue=(np.array(refframetensor)+np.array(true_ae_number)).tolist()

    #按照模型所需格式重组
    
    ###kp_mat 3*3 matrix, e.g. [[[63,25,28],[39,63,30],[36,33,64]]] 
    reallatentvalue_split1=reallatentvalue[0:9]
    latentformat1=[]
    for slideformat1 in range(0,3):
        latentformatslide1=reallatentvalue_split1[slideformat1*listnum:(slideformat1+1)*listnum]
        latentformat1.extend([latentformatslide1])
    kp_mat_value=[latentformat1]
    
    ###kp_t_value 1*3 matrix e.g. [[30,25,38]]
    reallatentvalue_split2=reallatentvalue[9:12]
    kp_t_value=[reallatentvalue_split2]
    
    ###kp_exp_value 1*45 matrix e.g. [[30,25,38]]
    reallatentvalue_split3=reallatentvalue[12:57]
    kp_exp_value=[reallatentvalue_split3] 

    ###保存恢复的真实的数据格式
    kp_mat_value=str(kp_mat_value)
    kp_mat_value="".join(kp_mat_value.split())
    kp_t_value=str(kp_t_value)
    kp_t_value="".join(kp_t_value.split())    
    kp_exp_value=str(kp_exp_value)
    kp_exp_value="".join(kp_exp_value.split())        
    

    return reallatentvalue,kp_mat_value,kp_t_value,kp_exp_value



def listformat_kp_jocobi_FOMM(refframetensor, true_ae_number):
    
    group=60
    ##按照模型所需格式进行重组的相关参数设置
    listsplit=int(group/(1+2))
    listnum=2
    slidelistnum=int((group/(1+2)*1)/listnum)
    ##对恢复得到的dataRec进行切片，每16个组成一组，并进行帧间补偿    

    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    reallatentvalue=(np.array(refframetensor)+np.array(true_ae_number)).tolist()

    #按照模型所需格式重组
    ###kp
    reallatentvalue_split1=reallatentvalue[0*listsplit:1*listsplit]
    latentformat1=[]
    for slideformat1 in range(0,slidelistnum):
        latentformatslide1=reallatentvalue_split1[slideformat1*listnum:(slideformat1+1)*listnum]
        latentformat1.extend([latentformatslide1])
    latentformat1=[latentformat1]
    #print(latentformat1)        

    ###jocobi
    reallatentvalue_split2=reallatentvalue[1*listsplit:3*listsplit]
    latentformat2=[]
    for slideformat2 in range(0,int((group/(1+2)*2)/listnum)):
        latentformatslide2=reallatentvalue_split2[slideformat2*listnum:(slideformat2+1)*listnum]
        latentformat2.extend([latentformatslide2])
    
    latentformat2_final=[]
    for sub in range(int(len(latentformat2)/2)):
        latentformatslide2_final= [latentformat2[sub*2],latentformat2[sub*2+1]]
        latentformat2_final.append(latentformatslide2_final)
    
    latentformat2_final=[latentformat2_final]

    ###保存恢复的真实的数据格式
    latentformat1=str(latentformat1)
    latentformat1="".join(latentformat1.split())
    latentformat2_final=str(latentformat2_final)
    latentformat2_final="".join(latentformat2_final.split())        
    return reallatentvalue,latentformat1,latentformat2_final


