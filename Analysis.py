# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:44:29 2020

@author: Liu's
"""
 
from skimage.feature import register_translation
import numpy as np
from scipy.ndimage.fourier import fourier_shift
import os
import glob

import tifffile as tf
pixel_x=256
pixel_y=256

def listallfiles(directory):
    currentcwd = os.getcwd()
    if currentcwd != directory:
        os.chdir(directory)
    files = glob.glob('*.tif')
    return files


def readdata(name):
    data =np.zeros((1,int(pixel_x),int(pixel_y)))
    for f in name:
        print(f)
        temp = tf.imread(f)
        data = np.append(data,temp,0)
    data = data[1::]
    return data

def driftcorrect(data):
    z,x,y = data.shape
    data =data.astype('float32')
    final = np.zeros_like(data)
    for i in np.arange(z):
        shift, error, diffphase = register_translation(data[0], data[i],100)
        correctedimg = fourier_shift(np.fft.fftn(data[i]), shift)
        correctedimg = np.fft.ifftn(correctedimg).real
        progress = np.round(i/z*100)
        print(progress)
        final[i] = correctedimg.astype('float32')
    return final


       
