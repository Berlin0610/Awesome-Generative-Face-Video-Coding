import numpy as np
import torch
import torch.nn.functional as F
import imageio
from torch import nn

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
from GFVC.CFTE.modules.util import make_coordinate_grid
from GFVC.CFTE.flowvisual import *


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
            
    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis] #[[256 256]]
        kp_array = spatial_size * (kp_array + 1) / 2 #(10, 64, 64, 2)
        num_kp = kp_array.shape[0] #10

        for kp_ind, kp in enumerate(kp_array):
            print(kp_ind)
            print(kp)
            
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image 
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source))
        
        # Driving image 
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving))        
            

        #Sparse Flow
        if 'sparse_motion' in out:
            sparseflow = out['sparse_motion'].data.cpu().numpy()

            bs, h, w, c = sparseflow.shape
            flow=[]
            for batch in range(0,bs):
                sf =flow_to_image(sparseflow[batch:batch+1,:,:,:].reshape(h, w, c))
                flow.append(sf)

            sparse_flow= np.array(flow)
            sparse_flow = np.transpose(sparse_flow, [0, 3, 1, 2])
            sparse_flow = torch.from_numpy(sparse_flow).type(source.type())  ###.type(dtype=torch.float64)
            sparse_flow = F.interpolate(sparse_flow, size=source.shape[1:3]).numpy()
            sparse_flow = np.transpose(sparse_flow, [0, 2, 3, 1])   
            images.append(sparse_flow)          
        
        ### sparse motion deformed image
        if 'sparse_deformed' in out:        
            sparse_deformed = out['sparse_deformed'].data.cpu().repeat(1, 1, 1, 1)
            sparse_deformed = F.interpolate(sparse_deformed, size=source.shape[1:3]).numpy()
            sparse_deformed = np.transpose(sparse_deformed, [0, 2, 3, 1])
            images.append(sparse_deformed)


        #Dense Flow
        if 'deformation' in out:
            denseflow = out['deformation'].data.cpu().numpy()

            bs, h, w, c = denseflow.shape
            flow=[]
            for batch in range(0,bs):
                df =flow_to_image(denseflow[batch:batch+1,:,:,:].reshape(h, w, c))
                flow.append(df)

            dense_flow= np.array(flow)
            dense_flow = np.transpose(dense_flow, [0, 3, 1, 2])
            dense_flow = torch.from_numpy(dense_flow).type(source.type()) 
            dense_flow = F.interpolate(dense_flow, size=source.shape[1:3]).numpy()
            dense_flow = np.transpose(dense_flow, [0, 2, 3, 1])
            images.append(dense_flow)              
            
        # denseflow Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)            
            
        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Driving Result 
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)
        
        
        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
