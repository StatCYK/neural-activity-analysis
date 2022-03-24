import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import *


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
        ROI_signals = np.load(Savefish_Dir +'/ROI.npy')

        ROI_signals_rerange = np.zeros_like(ROI_signals)
        ROI_signals_rerange[0:4,:] = ROI_signals[2*np.arange(4),:] 
        ROI_signals_rerange[4:8,:] = ROI_signals[2*np.arange(4)+1,:]

        d = pd.DataFrame(data=ROI_signals_rerange.T,columns =  ["SM", "SFGS", "SAC & SGC","SPV","SM", "SFGS", "SAC & SGC","SPV"])
        dcorr = d.corr(method='pearson')
        plt.figure(figsize=(11, 9),dpi=100)
        sns.set(font_scale=2.2) 
        sns.heatmap(data=dcorr,vmin=-1,vmax=1,cmap = plt.get_cmap("RdBu"))
        plt.savefig(Savefish_Dir +'/corr.png')