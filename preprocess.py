# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:44:29 2020

@author: Liu & Chen's
"""
import bokeh.plotting as bpl
import cv2
import glob
import logging
import tifffile as tif
import multiprocessing
from multiprocessing.pool import ThreadPool
import copy
from sklearn.neighbors import KNeighborsRegressor as knnR
import scipy.stats as stats
import time
from scipy.spatial.distance import euclidean
from skimage.feature import register_translation
import numpy as np
from scipy.ndimage.fourier import fourier_shift
import os
import ruptures as rpt
import multiprocessing as mp
from utilities import *


pixel_x=256 # scale setting for preprocessed video
pixel_y=256

### an example ##
Dir = "../Data/" # data dir
SaveDir = "../Output/results/wt/" # dir for saving preprocessed data
Name = ["fish1run1","fish1run2"] # expriment name
if os.path.isdir(SaveDir) == False:
    os.mkdir(SaveDir)

fps = 50

for name in Name:
    fishname = name
    Savefish_Dir = SaveDir+fishname
    if os.path.isdir(Savefish_Dir) == False:
        os.mkdir(Savefish_Dir)
    for i in range(1,5):
        print("%d th is loaded"%i)
        if i ==1:
            video = tif.imread(Dir+name+"/%s(1).tif"%fishname)[:,::4,::4]
            video = np.array([cv2.resize(cv2.flip(video[i,:,:],0),(pixel_x,pixel_y)) for i in range(video.shape[0]) ])
        else:
            video_sub = tif.imread(Dir+name+"/%s(%d).tif"%(fishname,i))[:,::4,::4]
            video_sub = np.array([cv2.resize(cv2.flip(video_sub[i,:,:],0),(pixel_x,pixel_y)) for i in range(video_sub.shape[0]) ])
            video = np.concatenate((video,video_sub))
    img = np.mean(video,0).astype("uint8")
    cv2.imwrite(Savefish_Dir+ "/brain_mean_gad1.jpg",img)
    video = driftcorrect(video)
    video = video.astype("int")
    video_shifted = BackExtract(video)
    print("finish shift")
    video_straight = np.array([video_shifted[i,:,:].reshape((video_shifted.shape[2]*video_shifted.shape[1],)) for i in range(video_shifted.shape[0])]).T
    pool = multiprocessing.Pool(processes=28)
    video_smooth = np.array(pool.map(knnsmooth,video_straight))
    pool.close()
    print("finish smooth!")
    np.save(Savefish_Dir + "/video_smooth.npy",video_smooth)