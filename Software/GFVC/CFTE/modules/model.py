# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import *
import numpy as np
from torch.autograd import grad
from .GDN import GDN
import math
from modules.vggloss import *
from modules.dists import *


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator,videocompressor, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.videocompressor = videocompressor
        
        self.train_params = train_params
        self.scale_factor = train_params['scale_factor']
        self.scales = train_params['scales']
        self.temperature =train_params['temperature']
        self.out_channels =train_params['num_kp']       
        self.disc_scales = self.discriminator.scales
        
        self.down = AntiAliasInterpolation2d(generator.num_channels, self.scale_factor)    
            
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

        self.dists = DISTS()
        if torch.cuda.is_available():
            self.dists = self.dists.cuda()        
             
    def forward(self, x, lambda_var):
        
        bs,_,width,height=x['source'].shape
        heatmap_source = self.kp_extractor(x['source']) ###
        heatmap_driving = self.kp_extractor(x['driving'])  ####
        
        lamdaloss = lambda_var
        
        if torch.cuda.is_available():                
            lambda_var = torch.tensor(lambda_var).cuda()                  
        
        total_bits_mv, quant_driving = self.videocompressor(heatmap_driving,heatmap_source)    #####         

        generated = self.generator(x['source'],heatmap_source=heatmap_source,heatmap_driving=quant_driving) #####
        generated.update({'heatmap_source':heatmap_source,'heatmap_driving':quant_driving})   ####     

        loss_values = {}

        pyramide_real = self.pyramid(x['driving']) 
        pyramide_generated = self.pyramid(generated['prediction'])


        driving_image_downsample = self.down(x['driving'])    ### [3,64,64]   
        pyramide_real_downsample = self.pyramid(driving_image_downsample) 
        sparse_deformed_generated = generated['sparse_deformed']  ### [3,64,64]
        sparse_pyramide_generated = self.pyramid(sparse_deformed_generated)  


        ####lambda
        loss_values['lambda'] = lambda_var         
        
        
        ###bpp loss
        bpp_mv = total_bits_mv / (bs * width * height) #####
        loss_values['bpp'] = bpp_mv 

        ### dists loss
        if torch.cuda.is_available():
            prediction=prepare_image(np.transpose(generated['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]).cuda() 
            groundtruth=prepare_image(np.transpose(x['driving'].data.cpu().numpy(), [0, 2, 3, 1])[0]).cuda() 
        dists= self.dists(groundtruth,prediction, as_loss=True)      ######                     
        loss_values['dists'] = dists    
        
        #### rd loss optimization
        rdloss = lamdaloss*bpp_mv + dists  ###
        loss_values['rdloss'] = rdloss

        
        ### Perceptual Loss---Initial
        if sum(self.loss_weights['perceptual_initial']) != 0:
            value_total = 0
            for scale in [1, 0.5, 0.25]:
                x_vgg = self.vgg(sparse_pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real_downsample['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_initial']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_initial'][i] * value
                loss_values['perceptual_64INITIAL'] = value_total        


        ### Perceptual Loss---Final
        if sum(self.loss_weights['perceptual_final']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_final']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_final'][i] * value
                loss_values['perceptual_256FINAL'] = value_total


        ### GAN Loss
        if self.loss_weights['generator_gan'] != 0:

            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)     

            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total        

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator,videocompressor, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.videocompressor= videocompressor
        
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)
        
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
