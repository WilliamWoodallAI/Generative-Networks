# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:12:46 2021

@author: William Woodall
"""

'''
Implementation of NVIDIA’s Progressive GAN
Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
NVIDIA
https://arxiv.org/pdf/1710.10196.pdf
I chose not to only use some of the weighted scaled convolution layers. 
'''

## View Training with GAN_training_display.py module
## Training images saved to ./training_imgs directory
## Model Checkpoints saved to ./models directory 
## Specify dataset directory in ProGANModel.img_dir

## I purposely do not leave comments in code as I believe that Python is a language and should 
#be able to be interpreted as such 

## Preprocess training images by resizing to intended output size before training to speed up preformance

from torchvision.transforms import ColorJitter 
from torchvision.transforms import RandomAffine ##(degrees, translate=None, scale=None, shear=None, 
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, calculate_gain
import numpy as np
import random
from PIL import Image
import cv2
from matplotlib import pyplot as plt

import pickle
import os

device = (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))



class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        
    def forward(self, img):
        return (img - self.mean) / self.std

class Denormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        
    def forward(self, img):
        return img * self.std + self.mean
    
class UpscaleLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = F.interpolate(x, x.shape[2] * self.scale_factor, mode='nearest')
        return x
    
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class SConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
        self.scale = (gain / (channels_in * kernel_size**2)) ** 0.5
        self.bias  = self.conv.bias
        self.conv.bias = None
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

        
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_shapes = [4,8,16,32,64,128,256,512,1024]
        self.conv_channels = [512,512,512,512,256,128,64,32,16]
        self.conv_blocks = 8
        self.scaling_step = 0
    
        self.from_rgb = nn.Sequential(nn.Conv2d(3, self.conv_channels[0], kernel_size=1, stride=1, padding=0), 
                                       nn.LeakyReLU(0.2)
                                      )
        
        self.conv_out = nn.Sequential(
                        SConv2d(self.conv_channels[0] + 1, self.conv_channels[0], kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(self.conv_channels[0], self.conv_channels[0], kernel_size=4, stride=1, padding=0),
                        nn.LeakyReLU(0.2)
                        )
               
        self.fc_out = nn.Linear(self.conv_channels[0], 1)
        
        
    def generate_conv_block(self, ch_in, ch_out):
        conv_block = [nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1), 
                      nn.LeakyReLU(0.2),
                      nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1), 
                      nn.LeakyReLU(0.2),
                      nn.AvgPool2d(kernel_size=2, stride=2),
                      ]
        return conv_block
        
    def scale_network(self):
        self.scaling_step += 1
        counter = 0
        
        self.from_rgb_pass = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        self.from_rgb_pass.add_module('RGB_1', self.from_rgb[0]) 
        self.from_rgb_pass.add_module('ReLU_2', self.from_rgb[1])                                
         
        self.from_rgb = nn.Sequential(nn.Conv2d(3, self.conv_channels[self.scaling_step], kernel_size=1, stride=1, padding=0), 
                                       nn.LeakyReLU(0.2)
                                      )
        if self.scaling_step > 1:
            self.conv_main_ = nn.ModuleList() 
            for i, item in enumerate(self.conv_scale):
                self.conv_main_.add_module(f'added_{self.scaling_step}_{counter}', item)
                counter += 1  
             
            for i, item in enumerate(self.conv_main):
                self.conv_main_.add_module(f'added_{self.scaling_step}_{counter}', item)
                counter += 1   
            self.conv_main = nn.Sequential(*self.conv_main_)
        else:
            self.conv_main = nn.Sequential(nn.Identity())
            
        self.conv_scale = nn.Sequential()
        for item in self.generate_conv_block(self.conv_channels[self.scaling_step], self.conv_channels[self.scaling_step - 1]):
            self.conv_scale.add_module(f'added_{self.scaling_step}_{counter}', item)
            counter += 1
            
    def blending_layer(self, alpha, passed_img, x):
        return alpha * x + (1- alpha) * passed_img
    
    def mini_batch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = torch.cat([x, batch_stats], dim=1)
        return x
        
    def forward(self, x, alpha):
        if self.scaling_step == 0:
            x = self.from_rgb(x)
            x = self.mini_batch_std(x)
            x = self.conv_out(x)
            x = x.view(x.shape[0], -1)
            x = self.fc_out(x)
            return x
        pass_img = self.from_rgb_pass(x)
        x = self.from_rgb(x)
        x = self.conv_scale(x)
        x = self.blending_layer(alpha, pass_img, x)
        x = self.conv_main(x)
        x = self.mini_batch_std(x)
        x = self.conv_out(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_out(x)
        return x
        
class GeneratorNetwork(nn.Module):
    def __init__(self, latent_dims, fc_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.fc_dims = fc_dims
        self.fc_layers = 8
        self.img_shapes = [4,8,16,32,64,128,256,512,1024]
        self.conv_channels = [512,512,512,512,256,128,64,32,16]
        self.scaling_step = 0
        
        self.mapping = nn.Sequential(nn.Linear(self.latent_dims, self.fc_dims),    
                                     nn.LeakyReLU(0.2)
                                     )
        for i in range(self.fc_layers -1):
            self.mapping.add_module('FC{i}', nn.Linear(self.fc_dims, self.fc_dims))
            self.mapping.add_module('LReLU{i}', nn.LeakyReLU(0.2))
        
        self.conv_in = nn.Sequential(
                        PixelNorm(),
                        nn.ConvTranspose2d(512, self.conv_channels[0], kernel_size=4, stride=1, padding=0), 
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(self.conv_channels[0], self.conv_channels[0], kernel_size=3, stride=1, padding=1), 
                        nn.LeakyReLU(0.2),
                        PixelNorm(),
                        )
        self.to_rgb = nn.Conv2d(self.conv_channels[0], 3, kernel_size=1, stride=1, padding=0)
        
    def scale_network(self):
        counter = 0
        self.scaling_step += 1
        
        if self.scaling_step > 1:
            for item in self.conv_scale:
                self.conv_in.add_module(f'added_{self.scaling_step}_{counter}', item)
                counter += 1
            
        self.conv_scale = nn.ModuleList()
        for item in self.generate_conv_block(self.conv_channels[self.scaling_step], self.conv_channels[self.scaling_step + 1]):
            self.conv_scale.append(item.to(device))         
        self.conv_scale = nn.Sequential(*self.conv_scale)
        
        self.to_rgb_pass = nn.Sequential(UpscaleLayer(2),
                                         self.to_rgb,
                                         ).to(device)
                                         
        self.to_rgb = nn.Conv2d(self.conv_channels[self.scaling_step + 1], 3, kernel_size=1, stride=1, padding=0).to(device)
                                 
    def generate_conv_block(self, ch_in, ch_out):
        conv_block = [
                    UpscaleLayer(2),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1), 
                    nn.LeakyReLU(0.2),
                    PixelNorm(),
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1), 
                    nn.LeakyReLU(0.2),
                    PixelNorm()
                    ]
        return conv_block
        
    def blending_layer(self, alpha, passed_img, x):
        return alpha * x + (1- alpha) * passed_img
    
    def forward(self, x, alpha):
        
        if self.scaling_step == 0:
            x = self.conv_in(x)
            x = self.to_rgb(x)
            return x
        
        x = self.conv_in(x)
        pass_latent = x
        x = self.conv_scale(x)    
        x = self.to_rgb(x)
        pass_latent = self.to_rgb_pass(pass_latent)
        x = self.blending_layer(alpha, pass_latent, x)
        return x
            
    
class ProGANModel():
    def __init__(self):
        super().__init__()
        self.latent_size = 512
        self.fc_dims = 512
        self.batch_size = 32
        self.train_steps = 500
        self.epochs = 10_000
        self.upscale_steps = 4
        self.upscale_interval = [50, 150, 250, 400, 600, 700, 650, 750,  850]
        self.image_sizes =      [ 4,   8,  16,  32,  64, 128, 256, 512, 1024]
        self.generator_update = 5
        self.lr = 1e-5
        self.generator = GeneratorNetwork(self.latent_size, self.fc_dims).to(device)
        self.critic = CriticNetwork().to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.0, 0.999))
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.999))
        self.img_size = 256
        self.img_size_full = 256
        self.img_dir = './train_data/'
        self.img_name = 'scene'
        self.img_output = './training_imgs/'
        self.img_ext = '.jpg'
        self.model_name = 'ProGAN'
        self.num_images = 251
        
        self.stats_dir = './stats/'
        self.stats_dict = {'critic_loss': [],
                           'generator_loss': [],
                           'score_real': [],
                           'score_fake': [],
                           'fake_mean': [],
                           'fake_std': [],
                           'real_mean': [],
                           'real_std': [],
                           'epoch': []
                           }
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
    
    
    
    def train(self, check_point=None):
        scaler = torch.cuda.amp.GradScaler()
        
        epoch_start = 0
        if not check_point == None:
            self.load_models(check_point)
            epoch_start = check_point
        
        self.img_size = self.critic.img_shapes[self.critic.scaling_step]
        seeds = self.generate_latent(self.batch_size).to(device)
        alpha = 0
        
        for epoch in range(epoch_start, self.epochs):
            
            if epoch == self.upscale_interval[self.critic.scaling_step] \
                and not self.critic.scaling_step >= self.upscale_steps:
                    
                    print('**** Scaling Network *****')
                    self.critic.scale_network()
                    self.generator.scale_network()
                    self.img_size = self.critic.img_shapes[self.critic.scaling_step]
                    self.generator.to(device)
                    self.critic.to(device)
                    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.0, 0.999))
                    self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.999))
                    alpha = 0
            
            for step in range(self.train_steps):            
                
                with torch.cuda.amp.autocast():
                    batch =         self.generate_batch(self.batch_size).to(device)
                    batch_latent =  self.generate_latent(self.batch_size).to(device)
                    
                    alpha = 2 * ((epoch + 1) % self.upscale_interval[self.critic.scaling_step - 1]) / \
                                (self.upscale_interval[self.critic.scaling_step + 1] - 
                                 self.upscale_interval[self.critic.scaling_step])
                                 
                    alpha = min(alpha, 1)
                    
                    fake = self.generator(batch_latent.detach(), alpha)
                    
                    self.critic_optimizer.zero_grad()
                    self.generator_optimizer.zero_grad()
                    
                    scores_real = self.critic(batch, alpha=alpha)
                    scores_fake = self.critic(fake, alpha=alpha)
                    
                    gradient_penalty =  self.gradient_penalty(batch, fake, alpha)
                    
                    critic_loss = -scores_real.mean() + scores_fake.mean() + 10 * gradient_penalty #+ (0.001 * (scores_real ** 2).mean())
                
                scaler.scale(critic_loss).backward()
                scaler.step(self.critic_optimizer)
                scaler.update()
                
                if step > 0 and not step % self.generator_update:
                    with torch.cuda.amp.autocast():
                        self.generator_optimizer.zero_grad()
                        fake = self.generator(batch_latent, alpha)
                        scores_fake = self.critic(fake, alpha)
                        generator_loss = -scores_fake.mean()
                        
                    scaler.scale(generator_loss).backward()
                    scaler.step(self.generator_optimizer)
                    scaler.update()
                    
                    print(f"Epoch:{epoch} Step:{step}, alpha:{np.round(alpha,3)}, Critic Loss:{critic_loss.item()}, Generator Loss:{generator_loss.item()}")
                    
            self.print_fake(seeds, alpha)
            self.print_real(batch)
            self.save_models(epoch + 1)
            self.update_stats(epoch, critic_loss.item(), generator_loss.item(), 
                            scores_real.mean().item(), scores_fake.mean().item(),
                            fake.mean().item(), fake.std().item(),
                            batch.mean().item(), batch.std().item(),             
                            )
                    
                
                
    def generate_batch(self, batch_size):
        img_idxs = np.random.choice(np.arange(1, self.num_images, 1), batch_size, replace=False)
        batch = []
        for idx in img_idxs:
            img = plt.imread(self.img_dir + self.img_name + f'{idx}' + self.img_ext)
            img = self.process_img(img)
            if len(batch) > 0:
                batch = torch.cat((batch,img), dim=0)
            else:
                batch = img
            batch = transforms.RandomApply(nn.ModuleList([
                            #transforms.CenterCrop(112),
                            #transforms.RandomInvert(p=0.2),
                            RandomAffine(180, fill=-1.8),
                            #ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            ]), p=0.5)(batch)
        #batch = torch.Tensor(batch).to(device)
        return batch
    
    def gradient_penalty(self, real, fake, alpha):
        beta = torch.rand(real.shape[0], 1, 1, 1)
        beta = beta.repeat(1, real.shape[1], real.shape[2], real.shape[2]).to(device)
        mixed_images = real * beta + fake.detach() * (1 - beta)
        
        mixed_images.requires_grad_(True)
        mixed_images = mixed_images.to(device)
        
        mixed_scores = self.critic(mixed_images, alpha)
    
        gradient = torch.autograd.grad(
            inputs=mixed_images, outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True, retain_graph=True)[0]
        
        gradient = gradient.reshape(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def generate_latent(self, batch_size):
        x = torch.randn(batch_size, self.latent_size)
        x = x.view(x.shape[0],x.shape[1], 1, 1)
        return x
    
    def upscale_seeds(self, seeds):
        size = self.critic.img_shapes[self.critic.scaling_step]
        for seed in seeds:
            seed = F.interpolate(seed, size=([size]))
            seed = seed.permute(0,2,1)
            seed = F.interpolate(seed, size=([size]))
            seed = seed.permute(0,2,1)
        return seeds
    
    def print_fake(self, seeds, alpha):
        imgs = self.generator(seeds, alpha)
        imgs = imgs.detach().cpu()
        for i in range(4): 
            img = self.deprocess_img(imgs[i])    
            img = cv2.resize(img, (self.img_size_full, self.img_size_full))
            img = Image.fromarray(img)
            img.save(self.img_output + f"fake_{i}" + self.img_ext)
    
    def print_real(self, batch):
        batch = batch.detach().cpu()
        for i in range(4):
            img = self.deprocess_img(batch[i])
            img = cv2.resize(img, (self.img_size_full, self.img_size_full), interpolation= cv2.INTER_CUBIC)
            img = Image.fromarray(img)
            img.save(self.img_output + f"real_{i}" + self.img_ext)
            
    
    def process_img(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)
        img = transforms.ToTensor()(img)
        img = Normalization()(img)
        return img.unsqueeze(0)
    
    def deprocess_img(self, img):
        img = Denormalization()(img)
        img = transforms.ToPILImage()(img.detach().cpu())
        img = np.array(img)
        return img
    
    
    def save_models(self, epoch):
        torch.save(self.generator, './models/' + self.model_name + f'_gen_E_{epoch}.pth')
        torch.save(self.critic, './models/' + self.model_name + f'_critic_E_{epoch}.pth')
    
    def load_models(self, epoch):
        self.generator = torch.load('./models/' + self.model_name + f'_gen_E_{epoch}.pth')
        self.critic = torch.load('./models/'+ self.model_name + f'_critic_E_{epoch}.pth')   
        
    def update_stats(self, epoch, critic_loss, generator_loss, score_real, score_fake, 
                     fake_mean, fake_std, real_mean, real_std):
        self.stats_dict['epoch'].append(epoch)
        self.stats_dict['critic_loss'].append(critic_loss)
        self.stats_dict['generator_loss'].append(generator_loss)
        self.stats_dict['score_real'].append(score_real)
        self.stats_dict['score_fake'].append(score_fake)
        self.stats_dict['fake_mean'].append(fake_mean)
        self.stats_dict['fake_std'].append(fake_std)
        self.stats_dict['real_mean'].append(real_mean)
        self.stats_dict['real_std'].append(real_std)
        with open(self.stats_dir + 'training_hist.pkl', 'wb') as f:
            pickle.dump(self.stats_dict, f)
    
    
GAN = ProGANModel()    

GAN.train()

        
        
        
        
        
        
        
        
    
    
    
    