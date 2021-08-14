# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 01:19:48 2020

@author: William Woodall
"""

from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns

import numpy as np
import pickle

train_stats_dir = './stats/training_hist.pkl'


plt.style.use('fivethirtyeight')
sns.set(style='darkgrid', palette='bright', font_scale=0.9)

fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(9,8) 
ax = fig.add_subplot(gs[:3,:])
ax2 = fig.add_subplot(gs[3:6,:])

ax3 = fig.add_subplot(gs[6:9,:2])
ax4 = fig.add_subplot(gs[6:9,2:4])

ax5 = fig.add_subplot(gs[6:9,4:6])
ax6 = fig.add_subplot(gs[6:9,6:8])
   
def plot_stats_animation(i):
    
    with open(train_stats_dir,'rb') as f:
        stats_dict = pickle.load(f)
        
    stats_dict['critic_average'] = [np.mean(stats_dict['critic_loss'][:i]) for i in range(len(stats_dict['critic_loss']))]
    stats_dict['generator_average'] = [np.mean(stats_dict['generator_loss'][:i]) for i in range(len(stats_dict['generator_loss']))]
    
    ax.cla()
    ax.plot(stats_dict['epoch'], stats_dict['critic_loss'], '.', linewidth=0.5)
    ax.plot(stats_dict['epoch'], stats_dict['critic_loss'], linewidth=0.5, color=(204/255,0/255,255/255), label='Critic Loss')
    ax.plot(stats_dict['epoch'], stats_dict['critic_average'], linewidth=0.9, color=(204/255,0/255,255/255))
    ax.text(stats_dict['epoch'][-1], stats_dict['critic_average'][-1], f"{np.round(stats_dict['critic_average'][-1],4)}", color='blue', alpha=0.6)
    ax.legend()
    ax.set_title('GAN Training History')

    ax2.cla()
    ax2.plot(stats_dict['epoch'], stats_dict['generator_loss'], '.', linewidth=0.5)
    ax2.plot(stats_dict['epoch'], stats_dict['generator_loss'], linewidth=0.5, color='red', label='Generator Loss')
    ax2.plot(stats_dict['epoch'], stats_dict['generator_average'], linewidth=0.8, color='red')
    ax2.text(stats_dict['epoch'][-1], stats_dict['generator_average'][-1], f"{np.round(stats_dict['generator_average'][-1],4)}", color='blue', alpha=0.6)
    ax2.legend()
      
    ax3.cla()
    ax3.set_title('Real Score')
    ax3.plot(stats_dict['epoch'], stats_dict['score_real'], '-', linewidth=0.5)
    ax3.text(stats_dict['epoch'][-1], stats_dict['score_real'][-1], f"{np.round(stats_dict['score_real'][-1],4)}", color='blue', alpha=0.6)
    
    ax4.cla()
    ax4.set_title('Fake Score')
    ax4.plot(stats_dict['epoch'], stats_dict['score_fake'], '-', linewidth=0.5)
    ax4.text(stats_dict['epoch'][-1], stats_dict['score_fake'][-1], f"{np.round(stats_dict['score_fake'][-1],4)}", color='blue', alpha=0.6)
    
    ax5.cla()
    ax5.set_title('Image Mean')
    ax5.plot(stats_dict['epoch'], stats_dict['fake_mean'], '.-', linewidth=0.5)
    ax5.plot(stats_dict['epoch'], stats_dict['real_mean'], '.-', color='green', linewidth=0.5)
    
    ax6.cla()
    ax6.set_title('Image std')
    ax6.plot(stats_dict['epoch'], stats_dict['fake_std'], '.-', linewidth=0.5)
    ax6.plot(stats_dict['epoch'], stats_dict['real_std'], '.-', color='green', linewidth=0.5)
    
ani = animation.FuncAnimation(fig, plot_stats_animation, interval=300)
plt.show()