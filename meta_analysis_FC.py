import numpy as np
import copy
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation
import matplotlib.pyplot as plt
from utilities import *


# an example
SaveDirs = ["../Output/results/wt/", "../Output/results/gad1b/",\
            "../Output/results/PTZ/","../Output/results/gad1b+PTZ/"]

Names = [["fish1run2","fish1run3"], ["fish1","fish1run2"],["fish1run1","fish1run2"],["fish1run1","fish1run2"]]

fps_s = [50,50,50,50]
pixel_x=256 # scale of preprocessed video
pixel_y=256
groups = ["wt","gad1b","PTZ","gad1b+PTZ"]

thresh = 0.6
width = 20
step = 3
connect_within = []
connect_between = []
group = []

for nn in range(len(SaveDirs)):
    fps = fps_s[nn]
    SaveDir = SaveDirs[nn]
    Name = Names[nn]
    gp = groups[nn]
    for name in Name:
        Savefish_Dir = SaveDir+name
        ROI_signals = np.load(Savefish_Dir +'/ROI.npy')

        ROI_signals_rerange = np.zeros_like(ROI_signals)
        ROI_signals_rerange[0:4,:] = ROI_signals[2*np.arange(4),:] 
        ROI_signals_rerange[4:8,:] = ROI_signals[2*np.arange(4)+1,:]

        res = SparsityGraph(ROI_signals_rerange,int(fps),thresh = thresh, width = width,step = step)
        group.append(gp)
        connect_within.append(res[0])
        connect_between.append(res[1])


data = pd.DataFrame({"connection frequency":connect_within,\
                     "group": group})
plt.figure(1,(12,8))
ax = sns.boxplot(x = "group", y = "connection frequency",data=data,palette="Set3")
add_stat_annotation(ax, data=data, x="group", y="connection frequency", 
                    box_pairs=[("gad1b","wt"),
                                 ("gad1b","PTZ"),
                                 ("PTZ","wt"),
                               ("gad1b","gad1b+PTZ"),
                                 ("gad1b+PTZ","PTZ"),
                                 ("gad1b+PTZ","wt")
                                ],
                    test='Mann-Whitney', text_format='simple', loc='inside', verbose=2)

data = pd.DataFrame({"connection frequency":connect_between,\
                     "group": group})
plt.figure(1,(12,8))
ax = sns.boxplot(x = "group", y = "connection frequency",data=data,palette="Set3")
add_stat_annotation(ax, data=data, x="group", y="connection frequency", 
                    box_pairs=[("gad1b","wt"),
                                 ("gad1b","PTZ"),
                                 ("PTZ","wt"),
                               ("gad1b","gad1b+PTZ"),
                                 ("gad1b+PTZ","PTZ"),
                                 ("gad1b+PTZ","wt")
                                ],
                    test='Mann-Whitney', text_format='simple', loc='inside', verbose=2)