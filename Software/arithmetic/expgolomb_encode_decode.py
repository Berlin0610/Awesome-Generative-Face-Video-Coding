import numpy as np
import math
from collections import Counter
import itertools

###0-order
def get_digits(num):
    result = list(map(int,str(num)))
    return result

def exponential_golomb_encode(n):
    unarycode = ''
    golombCode =''
    ###Quotient and Remainder Calculation
    groupID = np.floor(np.log2(n+1))
    temp_=groupID
    #print(groupID)
    
    while temp_>0:
        unarycode = unarycode + '0'
        temp_ = temp_-1
    unarycode = unarycode#+'1'

    index_binary=bin(n+1).replace('0b','')
    golombCode = unarycode + index_binary
    return golombCode
        

### golombcode : 00100 real input[0,0,1,0,0]
def exponential_golomb_decode(golombcode):

    code_len=len(golombcode)

    ###Count the number of 1's followed by the first 0
    m= 0 ### 
    for i in range(code_len):
        if golombcode[i]==0:
            m=m+1
        else:
            ptr=i  ### first 0
            break

    offset=0
    for ii in range(ptr,code_len):
        num=golombcode[ii]
        offset=offset+num*(2**(code_len-ii-1))
    decodemum=offset-1
    
    return decodemum


def expgolomb_split(expgolomb_bin_number):

    x_list=expgolomb_bin_number
    
    del(x_list[0]) 
    x_len=len(x_list)
    
    sublist=[]
    while (len(x_list))>0:

        count_number=0
        i=0
        if x_list[i]==1:
            sublist.append(x_list[0:1])
            del(x_list[0])            
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            count_number=count_number+num_times_zeros[0]
            sublist.append(x_list[0:(count_number*2+1)])
            del(x_list[0:(count_number*2+1)])
    return sublist






    


