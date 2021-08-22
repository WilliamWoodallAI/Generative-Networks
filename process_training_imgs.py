# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 21:17:44 2021

@author: William Woodall
"""

from matplotlib import pyplot as plt
import cv2
from PIL import Image
import os 


img_dir = './scenes/'
img_name = 'scene'
output_dir = './train_data/'
img_ext = '.jpg'

len_data = 250
max_img_size = 128

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Processing.', end="")
for i in range(len_data + 1):
    print('.', end="")
    img = plt.imread(img_dir + img_name + f'{i}' + img_ext)
    img = cv2.resize(img, (max_img_size, max_img_size), interpolation = cv2.INTER_AREA)
    img = Image.fromarray(img)
    img.save(output_dir + img_name + f'{i}' + img_ext)
print('Complete!')
    
    