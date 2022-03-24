import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
import tifffile as tif
import tifffile as tf
from scipy import ndimage as ndi
from skimage.feature import register_translation
from scipy.ndimage.fourier import fourier_shift
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import dipy.align.imwarp as imwarp
from dipy.viz import regtools
from dipy.align.imwarp import DiffeomorphicMap
import copy
from scipy.signal import find_peaks
from utilities import *


ROI_names = ["SM", "SFGS", "SAC & SGC","SPV"]

# an example
SaveDirs = ["../Output/results/wt/", "../Output/results/gad1b/",\
            "../Output/results/PTZ/","../Output/results/gad1b+PTZ/"]

Names = [["fish1run2","fish1run3"], ["fish1","fish1run2"],["fish1run1","fish1run2"],["fish1run1","fish1run2"]]

fps_s = [50,50,50,50]
pixel_x=256 # scale of preprocessed video
pixel_y=256

result_dir = "../Output/results/meta analysis/"
groups = ["wt","gad1b","PTZ","gad1b+PTZ"]

PTZ =[]
gad1b =[]
wt = []
gad1b_PTZ=[]

PTZ_2 =[]
gad1b_2 =[]
wt_2 = []
gad1b_PTZ_2=[]

PTZ_freq =[]
gad1b_freq =[]
wt_freq =[]
gad1b_PTZ_freq = []
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()
fig6 = plt.figure()
figs = [fig1,fig2,fig3,fig4,fig5,fig6]
axs = [fig1.add_subplot(1, 1, 1),fig2.add_subplot(1, 1, 1),fig3.add_subplot(1, 1, 1),fig4.add_subplot(1, 1, 1),fig5.add_subplot(1, 1, 1),fig6.add_subplot(1, 1, 1)]
for i in range(4):
    axs[i].set_xlim(3., 25.)
    axs[i].set_ylim(0., 1)
    axs[i].set_xlabel("Frequency(Hz)",fontsize=16)
    axs[i].set_ylabel("Amplitude",fontsize=16)

for nn in range(len(fps_s)):
    fps = fps_s[nn]
    SaveDir = SaveDirs[nn]
    Name = Names[nn]
    group = groups[nn]
    if group == "wt":
        color = '#66CD00'
        shape = 'o'
    elif group == "gad1b":
        color = '#2F88C0'
        shape = '^'
    elif group == "PTZ":
        color = '#ff0066'
        shape = 'v'
    elif group == "gad1b+PTZ":
        color = 'black'
        shape = '+'
    for name in Name:
        print(name)
        Savefish_Dir = SaveDir+name
        if os.path.isfile(Savefish_Dir + '/ROI.npy'):
            ROI_signals = np.load(Savefish_Dir +'/ROI.npy')
            ROIs_std = np.std(ROI_signals,1)
            ROIs_mean = np.std(ROI_signals,1)
            win_size = fps*3
            fft_percents = []
            for i in range(ROI_signals.shape[0]):
                scaled_Hz = np.fft.fftfreq(ROI_signals.shape[1], d = 1/fps)
                fft_percent = getsp(ROI_signals[i,:]).reshape(-1,)
                lenrange= np.int(np.size(fft_percent)*0.5)
                fft_percent = fft_percent[0:lenrange]
                scaled_Hz = scaled_Hz[0:lenrange]
                indx=np.argwhere(scaled_Hz > 3).reshape(-1,)
                scaled_Hz = scaled_Hz[indx]
                fft_percent = fft_percent[indx]
                thresh=np.std(fft_percent)
                indx_value = np.argwhere(fft_percent > 4*thresh)[:,0].reshape(-1,)
                value = fft_percent[indx_value]
                if len(value) >0 :
                    mindx, _ =find_peaks(value, threshold=1*thresh,distance=500)
                    if len(mindx) >0 :
                        axs[i//2].scatter(scaled_Hz[indx_value[mindx]],value[mindx], color=color,marker = shape,s = 17,alpha = 0.8)
                        fft_percents.append(value)
                        freq_ap = [scaled_Hz[indx_value[mindx]][0],value[mindx][0]]
                        if nn==0:
                            wt_freq.append(freq_ap)
                        elif nn ==2:
                            PTZ_freq.append(freq_ap)
                        elif nn ==1:
                            gad1b_freq.append(freq_ap)
                        else:
                            gad1b_PTZ_freq.append(freq_ap)
for i in range(4):
    figs[i].savefig(result_dir+'freq_%s.png'%ROI_names[i])

PTZ_freq = np.array(PTZ_freq)
wt_freq = np.array(wt_freq)
gad1b_freq = np.array(gad1b_freq)
gad1b_PTZ_freq = np.array(gad1b_PTZ_freq)

np.save('freq_PTZ.npy',PTZ_freq)
np.save('freq_wt.npy',wt_freq)
np.save('freq_gad1b.npy',gad1b_freq)
np.save('freq_gad1b_PTZ.npy',gad1b_PTZ_freq)