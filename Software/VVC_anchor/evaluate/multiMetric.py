import cv2
import numpy as np
import math
import os
import torch
import sys
import time
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import inspect
from utils import downsample
from PIL import Image
from utils import prepare_image
from torchvision import transforms
from skimage.transform import resize
from scipy.signal import convolve2d

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        # a = torch.hann_window(5,periodic=False)
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    '''
    Refer to https://github.com/dingkeyan93/DISTS
    '''
    def __init__(self, channels=3, load_weights=True):
        assert channels == 3
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(DISTS),'..','weights/DISTS.pt')))
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, as_loss=True, resize = True):
        assert x.shape == y.shape
        if resize:
            x, y = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if as_loss:
            return score.mean()
        else:
            return score

class LPIPSvgg(torch.nn.Module):
    def __init__(self, channels=3):
        # Refer to https://github.com/richzhang/PerceptualSimilarity

        assert channels == 3
        super(LPIPSvgg, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,128,256,512,512]
        self.weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(LPIPSvgg),'..','weights/LPIPSvgg.pt')))
        self.weights = list(self.weights.items())
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        for k in range(len(outs)):
            outs[k] = F.normalize(outs[k])
        return outs

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        score = 0 
        for k in range(len(self.chns)):
            score = score + (self.weights[k][1]*(feats0[k]-feats1[k])**2).mean([2,3]).sum(1)
        if as_loss:
            return score.mean()
        else:
            return score

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def cacl_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
 
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))

        
def cacl_psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2.0 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2.0/mse)

        
if __name__ == "__main__":
       
    width=256
    height=256
    

        

    seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

    qplist=[ "37", "42", "47", "52"]


    testingdata_name='VOXCELEB' # 'CFVQA' OR 'VOXCELEB'
    if testingdata_name=='CFVQA':
        frames=125
    if testingdata_name=='VOXCELEB':
        frames=250
    
    
    Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444'
    
    
    result_dir = '../experiment/'+Iframe_format+'/evaluation/'

    

    metric = {'psnr': 1, 'ssim': 0, 'lpips': 1, 'dists': 1}

    if metric['psnr']:
        totalResult_PSNR=np.zeros((len(seqlist)+1,len(qplist)))
    if metric['ssim']:
        totalResult_SSIM=np.zeros((len(seqlist)+1,len(qplist)))
    if metric['lpips']:
        lpips = LPIPSvgg().cuda()
        totalResult_LPIPS=np.zeros((len(seqlist)+1,len(qplist)))
    if metric['dists']:    
        dists = DISTS().cuda()
        totalResult_DISTS=np.zeros((len(seqlist)+1,len(qplist)))

    seqIdx=0
    for seq in seqlist:
        qpIdx=0
        for qp in qplist:
            start=time.time()    
            if not os.path.exists(result_dir):
                os.makedirs(result_dir) 


            f_org=open('/mnt/workspace/code/GFVC/testing_sequence/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb','rb')

            f_test=open('../experiment/'+Iframe_format+'/YUVtoRGB/'+testingdata_name+'_'+str(seq)+'_qp'+qp+'_dec.rgb','rb')    #seq+'_256x256_QP'+qp    

            # f_test=open('../experiment/'+Iframe_format+'/Dec/'+testingdata_name+'_'+str(seq)+'_qp'+qp+'_dec.rgb','rb')    #seq+'_256x256_QP'+qp    

            if metric['psnr']:
                f_psnr = open(result_dir+testingdata_name+'_'+str(seq)+'_QP'+str(qp)+'_psnr.txt','w')
                sum_PSNR=0
            if metric['ssim']:
                f_ssim = open(result_dir+testingdata_name+'_'+str(seq)+'_QP'+str(qp)+'_ssim.txt','w')
                sum_SSIM=0
            if metric['lpips']:
                f_lpips = open(result_dir+testingdata_name+'_'+str(seq)+'_QP'+str(qp)+'_lpips.txt','w')
                sum_LPIPS=0
            if metric['dists']:    
                f_dists = open(result_dir+testingdata_name+'_'+str(seq)+'_QP'+str(qp)+'_dists.txt','w')
                sum_DISTS=0

            for frame in range(0,frames):                 
                img_org = np.fromfile(f_org,np.uint8,3*height*width).reshape((3,height,width))
                img_test = np.fromfile(f_test,np.uint8,3*height*width).reshape((3,height,width))
                ## PSNR SSIM
                if metric['psnr']:
                    R_psnr = cacl_psnr(img_org[0], img_test[0])
                    G_psnr = cacl_psnr(img_org[1], img_test[1])
                    B_psnr = cacl_psnr(img_org[2], img_test[2])
                    PSNR = (R_psnr+G_psnr+B_psnr)/3
                    f_psnr.write(str(PSNR))
                    f_psnr.write('\n')
                    sum_PSNR=sum_PSNR+PSNR
                if metric['ssim']:
                    R_ssim = cacl_ssim(img_org[0], img_test[0])
                    G_ssim = cacl_ssim(img_org[1], img_test[1])
                    B_ssim = cacl_ssim(img_org[2], img_test[2])
                    SSIM = (R_ssim+G_ssim+B_ssim)/3 
                    f_ssim.write(str(SSIM))
                    f_ssim.write('\n')
                    sum_SSIM=sum_SSIM+SSIM
                ## DISTS LPIPS
                img_org = resize(img_org, (3, height, width))
                img_test = resize(img_test, (3, height, width))
                img_org = torch.tensor(img_org[np.newaxis].astype(np.float32)).cuda()
                img_test = torch.tensor(img_test[np.newaxis].astype(np.float32)).cuda() 
                if metric['lpips']:
                    LPIPS = lpips(img_org,img_test, as_loss=False) 
                    f_lpips.write(str(LPIPS.cpu().detach().numpy()))
                    f_lpips.write('\n')
                    sum_LPIPS=sum_LPIPS+LPIPS
                if metric['dists']:
                    DISTS = dists(img_org,img_test, as_loss=False)
                    f_dists.write(str(DISTS.cpu().detach().numpy()))
                    f_dists.write('\n')
                    sum_DISTS=sum_DISTS+DISTS

            if metric['psnr']:
                sum_PSNR=sum_PSNR/frames
                totalResult_PSNR[seqIdx][qpIdx]=sum_PSNR
                f_psnr.close()
            if metric['ssim']:
                sum_SSIM=sum_SSIM/frames
                totalResult_SSIM[seqIdx][qpIdx]=sum_SSIM
                f_ssim.close()
            if metric['lpips']:
                sum_LPIPS=sum_LPIPS/frames
                totalResult_LPIPS[seqIdx][qpIdx]=sum_LPIPS
                f_lpips.close()
            if metric['dists']:
                sum_DISTS=sum_DISTS/frames
                totalResult_DISTS[seqIdx][qpIdx]=sum_DISTS
                f_dists.close()

            f_org.close()
            f_test.close()

            end=time.time()
            print(seq+'_QP'+str(qp)+'.rgb',"success. Time is %.4f"%(end-start))
            qpIdx+=1
        seqIdx+=1

    np.set_printoptions(precision=5)

    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            if metric['psnr']:
                totalResult_PSNR[-1][qp]+=totalResult_PSNR[seq][qp]
            if metric['ssim']:
                totalResult_SSIM[-1][qp]+=totalResult_SSIM[seq][qp]
            if metric['lpips']:
                totalResult_LPIPS[-1][qp]+=totalResult_LPIPS[seq][qp]
            if metric['dists']:
                totalResult_DISTS[-1][qp]+=totalResult_DISTS[seq][qp]    
        if metric['psnr']:
            totalResult_PSNR[-1][qp] /= len(seqlist)
        if metric['ssim']:
            totalResult_SSIM[-1][qp] /= len(seqlist)
        if metric['lpips']:
            totalResult_LPIPS[-1][qp] /= len(seqlist)
        if metric['dists']:
            totalResult_DISTS[-1][qp] /= len(seqlist)

    if metric['psnr']:
        np.savetxt(result_dir+testingdata_name+'_result_psnr.txt', totalResult_PSNR, fmt = '%.5f')
    if metric['ssim']:
        np.savetxt(result_dir+testingdata_name+'_result_ssim.txt', totalResult_SSIM, fmt = '%.5f')
    if metric['lpips']:
        np.savetxt(result_dir+testingdata_name+'_result_lpips.txt', totalResult_LPIPS, fmt = '%.5f')
    if metric['dists']:
        np.savetxt(result_dir+testingdata_name+'_result_dists.txt', totalResult_DISTS, fmt = '%.5f')



