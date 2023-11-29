import contextlib, sys
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
from functools import reduce
from arithmetic.expgolomb_encode_decode import *


############################# unary code#######################################
###10进制数转二进制数（一元码）
def unary(q):
    code1 = []
    for i in range(q):
        code1.append(1)
    code1.append(0)
    code2 = [str(i) for i in code1]
    code = "".join(code2)
    return code

###数组中数值处理（非负整数及正整数）
def dataconvert_unary(symbol):

    ##把负数的残差变换为正整数 （0，1，-1，2，-2——》0，1，2，3，4)
    for i in range(len(symbol)):
        if symbol[i] <=0:
            symbol[i]=(-symbol[i])*2
        else:
            symbol[i]=symbol[i]*2-1

    #print(symbol)

    ## 正整数直接+1（0，1，2，3，4——》1，2，3，4，5)
    for i in range(len(symbol)):
        symbol[i]=symbol[i]+1
    #print(symbol)
        
    return symbol
    
###正整数10进制转二进制一元码，并且将其顺序合并成0/1二进制字符串
def list_binary_unary(symbol):
    ### 10进制转为2进制
    for i in range(len(symbol)):                
        n = symbol[i]
        symbol[i]=unary(n)
        
    #print(symbol)
    
    ##将list的所有数字拼接成一个
    m=''
    for x in symbol:
        m=m+str(x)
    #print(int(m))
    return m

##encoder: input:Inter-frame residual output: bin file
def final_encoder_unary(datares,outputfile, MODEL_ORDER = 0):
     # Must be at least -1 and match ppm-decompress.py. Warning: Exponential memory usage at O(257^n).
    with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:  #arithmeticcoding.
    
        enc = ArithmeticEncoder(256, bitout) #########arithmeticcoding.
        #print(enc)
        model = PpmModel(MODEL_ORDER, 3, 2)  ##########ppmmodel. ppmmodel.
        #print(model)
        history = []

        # Read and encode one byte
        symbol=datares
        #print(symbol)
        
        # 数值转换
        symbol = dataconvert_unary(symbol)
        #print(symbol)
        
        # 二进制0/1字符串
        symbollist = list_binary_unary(symbol)
        #print(int(symbollist))
             
        
        ###依次读取这串拼接的数，从左到右输出
        for ii in symbollist:
            #print(ii)
            i_number=int(ii)
            
            encode_symbol(model, history, i_number, enc)

            model.increment_contexts(history, i_number)
            if model.model_order >= 1:
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, i_number) ###########
            #print(history)
        encode_symbol(model, history, 2, enc)  # EOF ##########
        enc.finish()  #
        
        
################################ 0-order exponential coding###############
###数组中数值处理（非负整数及正整数）
def dataconvert_expgolomb(symbol):

    ##把负数的残差变换为正整数 （0，1，-1，2，-2——》0，1，2，3，4)
    for i in range(len(symbol)):
        if symbol[i] <=0:
            symbol[i]=(-symbol[i])*2
        else:
            symbol[i]=symbol[i]*2-1

    return symbol
    
###正整数10进制转二进制一元码，并且将其顺序合并成0/1二进制字符串
def list_binary_expgolomb(symbol):
    ### 10进制转为2进制
    for i in range(len(symbol)):
        n = symbol[i]
        symbol[i]=exponential_golomb_encode(n)
                    
    #print(symbol)
    
    ##将list的所有数字拼接成一个
    m='1' ##1:标识符
    for x in symbol:
        m=m+str(x)
    #print(int(m))
    return m

##encoder: input:Inter-frame residual output: bin file
def final_encoder_expgolomb(datares,outputfile, MODEL_ORDER = 0):
     # Must be at least -1 and match ppm-decompress.py. Warning: Exponential memory usage at O(257^n).
    with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:  #arithmeticcoding.
    
        enc = ArithmeticEncoder(256, bitout) #########arithmeticcoding.
        #print(enc)
        model = PpmModel(MODEL_ORDER, 3, 2)  ##########ppmmodel.
        #print(model)
        history = []

        # Read and encode one byte
        symbol=datares
        #print(symbol)
        
        # 数值转换
        symbol = dataconvert_expgolomb(symbol)
        #print(symbol)
        
        # 二进制0/1字符串
        symbollist = list_binary_expgolomb(symbol)
        #print(int(symbollist))
             
        
        ###依次读取这串拼接的数，从左到右输出
        for ii in symbollist:
            #print(ii)
            i_number=int(ii)
            
            encode_symbol(model, history, i_number, enc)

            model.increment_contexts(history, i_number)
            if model.model_order >= 1:
                if len(history) == model.model_order:
                    history.pop()
                history.insert(0, i_number) ###########
            #print(history)
        encode_symbol(model, history, 2, enc)  # EOF ##########
        enc.finish()  #        
        
##encoder: input:Inter-frame residual output: bin file
def final_encoder_expgolomb_count_proposal(datares):

    # Read and encode one byte
    symbol=datares
    #print(symbol)

    # 数值转换
    symbol = dataconvert_expgolomb(symbol)
    #print(symbol)

    # 二进制0/1字符串
    symbollist = list_binary_expgolomb(symbol)
    return len(symbollist)

