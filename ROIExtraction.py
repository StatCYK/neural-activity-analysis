from operator import length_hint
import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
import tifffile as tif
from scipy import ndimage as ndi
from skimage.feature import register_translation
from scipy.ndimage.fourier import fourier_shift
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import dipy.align.imwarp as imwarp
from dipy.viz import regtools
from dipy.align.imwarp import DiffeomorphicMap
import copy
from utilities import *


# setting for image registration
metric = EMMetric(dim = 2, smooth=0, step_type="demons") #step_type="demons" seems better than step_type='gauss_newton'      
level_iters_original = [100,70,50,25,5]  
scaling = 20  # Use larger scaling paramter for better performance but slower CPU time
level_iters = [iii * scaling for iii in level_iters_original]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
masks1 = cv2.flip(tif.imread("../atlas/stratum_album_centrale.tif")[281:290,240:670,110:460],1)
masks2 = cv2.flip(tif.imread("../atlas/stratum_fibrosum_et_griseum_superficiale.tif")[281:290,240:670,110:460],1)
masks3 = cv2.flip(tif.imread("../atlas/stratum_griseum_centrale.tif")[281:290,240:670,110:460],1)
masks4 = cv2.flip(tif.imread("../atlas/stratum_marginale.tif")[281:290,240:670,110:460],1)
masks5 = cv2.flip(tif.imread("../atlas/stratum_opticum.tif")[281:290,240:670,110:460],1)
masks6 = cv2.flip(tif.imread("../atlas/periventricular_layer.tif")[281:290,240:670,110:460],1)

masks = [masks4,masks2,(255*((masks1+masks3)>0)+0).astype("uint8"),masks6]
ROI_names = ["SM", "SFGS", "SAC & SGC","SPV"]
atlas = cv2.flip(tif.imread("../atlas/Average_HuCGCaMP5G.tif")[281:290,240:670,110:460],1)


# an example
SaveDirs = ["../Output/results/wt/", "../Output/results/gad1b/",\
            "../Output/results/PTZ/","../Output/results/gad1b+PTZ/"]

Names = [["fish1run2","fish1run3"], ["fish1","fish1run2"],["fish1run1","fish1run2"],["fish1run1","fish1run2"]]

fps_s = [50,50,50,50]
pixel_x=256 # scale of preprocessed video
pixel_y=256

for nn in range(len(SaveDirs)):
    fps = fps_s[nn]
    SaveDir = SaveDirs[nn]
    Name = Names[nn]
    for name in Name:
        print(name)
        Savefish_Dir = SaveDir+name
        if os.path.isfile(Savefish_Dir + "/video_smooth.npy"):
            video_s = np.load(Savefish_Dir + "/video_smooth.npy")
            target_img = np.median(video_s,1).reshape((pixel_x,pixel_y))
            target_img = (50+(target_img-np.min(target_img))/(np.max(target_img)-np.min(target_img))*200).astype("uint8")
            pos = croscor(atlas[:,::6,::6],target_img[::4,::4],1)
            atlas_pos = cv2.resize(atlas[pos],target_img.shape)
            ROI_signals = np.zeros((2*len(ROI_names),video_s.shape[1]-fps*10))
            mapping = Mapping(atlas_pos,target_img)
            np.save(Savefish_Dir +'/target_img.npy',target_img)
            plt.figure(1,figsize=(49,17))
            for i in range(4):
                if i ==3:
                    ylim_interval = (-0.005,0.01)
                else:
                    ylim_interval = (-0.005,0.06)
                mask = masks[i]
                ROI_name = ROI_names[i]
                mask_atlas = cv2.resize(mask[pos],target_img.shape)
                mask_right,mask_left = mask_seg(mapping,mask_atlas)
                mask_right = mask_right/np.sum(mask_right)
                mask_left = mask_left/np.sum(mask_left)
                signal_r = np.dot(mask_right.reshape((1,pixel_x*pixel_y)),video_s).T
                signal_norm_r = normalization(signal_r,fps)
                signal_l = np.dot(mask_left.reshape((1,pixel_x*pixel_y)),video_s).T
                signal_norm_l = normalization(signal_l,fps)
                ROI_signals[2*i,:] = signal_norm_l
                ROI_signals[2*i+1,:] = signal_norm_r
                ax1=plt.subplot2grid((6,12),(i,0),colspan=1,rowspan=1)
                ax1.imshow(colormap(target_img,mask_left))
                ax1.axis("off")
                ax1.set_title(ROI_names[i],fontsize=16)
                ax2=plt.subplot2grid((6,12),(i,1),colspan=4,rowspan=1)
                plt.plot(np.arange(len(signal_norm_l))/fps,signal_norm_l)
                plt.xlabel('Time(Secs.)',fontsize=16)
                plt.ylim(ylim_interval)
                plt.ylabel("ΔF/F",fontsize=16)
                ax3=plt.subplot2grid((6,12),(i,5),colspan=1,rowspan=1)
                ax3.imshow(colormap(target_img,mask_right))
                ax3.axis("off")
                ax3.set_title(ROI_names[i],fontsize=16)
                ax4=plt.subplot2grid((6,12),(i,6),colspan=4,rowspan=1)
                plt.plot(np.arange(len(signal_norm_r))/fps,signal_norm_r)
                plt.xlabel('Time(Secs.)',fontsize=16)
                plt.ylabel("ΔF/F",fontsize=16)
                plt.ylim(ylim_interval)
                np.save(Savefish_Dir +'/ROI%s_left.npy'%ROI_names[i],mask_left)
                np.save(Savefish_Dir +'/ROI%s_right.npy'%ROI_names[i],mask_right)
            plt.savefig(Savefish_Dir +'/ROI.png')
            np.save(Savefish_Dir +'/ROI.npy',ROI_signals)
