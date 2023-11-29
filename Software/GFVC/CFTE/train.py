# -*- coding: utf-8 -*-
from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train(config, generator, discriminator, kp_detector,videocompressor, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    
    rdlambdas = config['train_params']['loss_weights']['rdlambda']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_videocompressor = torch.optim.Adam(videocompressor.parameters(), lr=train_params['lr_videocompressor'], betas=(0.5, 0.999))
    
    optimizer_aux = torch.optim.Adam(videocompressor.parameters(), lr=train_params['lr_videocompressor'], betas=(0.5, 0.999))   ### 

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,videocompressor,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_videocompressor = MultiStepLR(optimizer_videocompressor, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)   
    
    scheduler_aux = MultiStepLR(optimizer_aux, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)   #####
    
    
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, videocompressor, train_params) #####
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, videocompressor, train_params) #####


#     if torch.cuda.is_available():
#         generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids).cuda()
#         discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids).cuda()    

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full, device_ids=device_ids).to(device_ids[0])
        discriminator_full = torch.nn.DataParallel(discriminator_full, device_ids=device_ids).to(device_ids[0])  


    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:    

                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()                
                optimizer_videocompressor.step()
                optimizer_videocompressor.zero_grad()
                
                optimizer_aux.step() ###
                optimizer_aux.zero_grad() ###


                #lambda_var = abs(rdlambdas)
                lambda_var = rdlambdas                      

                print("lambda_var")
                print(lambda_var)

                losses_generator, generated = generator_full(x,lambda_var) #####

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss.backward(retain_graph=True)  ####

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward(retain_graph=True)
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                print(losses)

                aux_loss = videocompressor.entropy_bottleneck.loss() ###
                print(aux_loss)
                aux_loss.backward(retain_graph=True) ###
                
                
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_videocompressor.step()

            scheduler_aux.step()   ####         
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'videocompressor':videocompressor,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_videocompressor':optimizer_videocompressor}, inp=x, out=generated)
