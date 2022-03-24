# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:44:29 2020

@author: Yang Liu
"""

from skimage.feature import register_translation
import numpy as np
from scipy.ndimage.fourier import fourier_shift
import os
import glob
import time
import ruptures as rpt
from sklearn.neighbors import KNeighborsRegressor as knnR
import multiprocessing as mp
n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
import tifffile as tf
pixel_x=256
pixel_y=256

def binimg(img):
    scale = 0.25
    m,n = img.shape
    binn_num = np.int(1/scale)
    m_new = np.int(m*scale)
    n_new = np.int(n*scale)
    binned = img.reshape(m_new,binn_num,n_new,binn_num).mean(-1).mean(1).astype('uint8')
    return binned

def listallfiles(directory):
    currentcwd = os.getcwd()
    if currentcwd != directory:
        os.chdir(directory)
    files = glob.glob('*.tif')
    return files

def readandconvert(name):
    data =np.zeros((1,int(pixel_x),int(pixel_y)))
    for f in name:
        print(f)
        temp = tf.imread(f)
        for i in np.arange(temp.shape[0]):
            temp[i] = converttouint8(temp[i])
            temp[i] = binimg(temp[i])
        data = np.append(data,temp,0)
    data = data[1::]
    return data





def converttouint8(data):
    # data = img_as_ubyte(data)
    info = np.iinfo(data.dtype) # Get the information of the incoming image type
    # data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
    # data = 255 * data# Now scale by 255
    # data = data.astype(np.uint8)
    tragetuplimit =  np.iinfo(np.uint8)
    tragetmin = np.around(data.min()/info.max*tragetuplimit.max)
    tragetmax = np.around(data.max()/info.max*tragetuplimit.max)
    data = convert(data,tragetmin,tragetmax,np.uint8)
    # data= ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8) ### scale max 
    return data
   
#test = data.T.reshape((15000,256,256)) convert back to img stack     

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


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


def makestack(data):
    row,col = np.shape(data)
    data = data.T.reshape((int(col),int(np.sqrt(row)),int(np.sqrt(row)))).astype('float32')
    return data

def smoothrow (data):
    start = time.perf_counter()
    row,col =np.shape(data)
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(knnsmooth, args=(data[x,:],15)) for x in np.arange(row)]
    pool.close()  
    stop = time.perf_counter()
    results = np.array(results).astype('float32')
    print(stop-start)
    return results


def knnsmooth(signal, width = 7):#, estimate_step = 1):
    X_estimate = np.array([np.median(signal[max(i-width,0):min(i+width,signal.shape[0])]) for i in range(signal.shape[0])])
    return X_estimate



def svddnoise(data,num_eiganvalue):
    u,s,vh = np.linalg.svd(data.astype('float32'),0,1) # Single Value
    s[num_eiganvalue::]=0
    s = np.diag(s)
    data = np.dot(u, np.dot(s, vh))
    data = makestack(data)
    return data

def ChangePointFind(Signal):
    Signal = Signal.reshape((-1,1))
    algo = rpt.Pelt(model="rbf").fit(Signal)
    result = algo.predict(pen=10)
    return result
def BackExtract(video,BackSize = 30):
    ImageSize = video.shape[1]
    background1 = np.mean(np.mean(video[:,0:BackSize,0:BackSize],1),1)
    background2 = np.mean(np.mean(video[:,(ImageSize-BackSize):ImageSize,0:BackSize],1),1)
    background3 = np.mean(np.mean(video[:,(ImageSize-BackSize):ImageSize,(ImageSize-BackSize):ImageSize],1),1)
    background4 = np.mean(np.mean(video[:,0:BackSize,(ImageSize-BackSize):ImageSize],1),1)
    BackSignal = (background1+background2+background3+background4)/4
    Result = []
    for k in range(len(BackSignal)//2000):
        Result.extend([2000*k+value for value in ChangePointFind(BackSignal[k*2000:(k+1)*2000])[:-1]])
    Result.extend([2000*(len(BackSignal)//2000)+value for value in ChangePointFind(BackSignal[-(len(BackSignal)%2000):])[:-1]])
    shift_matrix = np.zeros_like(video,dtype="int")
    for i in range(0,len(Result)):
        bpt = Result[i]-1
        shift_img = np.median(video[max(bpt-100,0):bpt,:,:],0)-np.median(video[bpt:min(bpt+100,video.shape[0]),:,:],0)
        shift_matrix_extend = np.tile(shift_img,(video.shape[0]-bpt,1,1)).astype("int")
        shift_matrix[bpt:,:,:] += shift_matrix_extend
    shift_matrix = shift_matrix.astype("int")
    video_new = video+shift_matrix
    return(video_new)

def normalization(value): 
    #width is sliding window size in seconds, fps is imaging speed 50 fps#
    width = 10
    num_points = value.shape[0]
    framenum = int(np.round(width*50)-1)
    newvalue = np.zeros((num_points-framenum))
    for i in np.arange(framenum,num_points):
        normdata = value[i-framenum:i]
        #get lower 50% value #
        a = np.percentile(normdata, 50, interpolation='midpoint') 
        b = normdata[normdata<a];
        norvalue = value[i]/np.mean(b)-1;
        newvalue[i-framenum] = norvalue
    return newvalue


### extract signal in ROIs #########
def ROI_extract(video_straight, region_size = 11, num_of_regions = 31**2):
    num_of_pixel = video_straight.shape[0]
    num_of_frame = video_straight.shape[1]
    region_mask = np.zeros((num_of_regions,num_of_pixel))
    count = 0
    for i in range(1,1+int(np.sqrt(num_of_regions))):
        for j in range(1,1+int(np.sqrt(num_of_regions))):
            mask_tmp = np.zeros((int(np.sqrt(num_of_pixel)),int(np.sqrt(num_of_pixel))))
            mask_tmp[(8*i-5):(8*i+6),(8*j-5):(8*j+6)] = 1
            region_mask[count,:] = mask_tmp.reshape((num_of_pixel,))
            count = count+1
    ROIs_signal = np.dot(region_mask, video_straight)/(11**2)
    return ROIs_signal


"""
directory = ""

files = listallfiles(directory)
video = readandconvert(files)
video = driftcorrect(video)
video = BackExtract(video)
video = data.astype("int")
video_shifted = BackExtract(video)
video_straight = np.array([video_shifted[i,:,:].reshape((video_shifted.shape[1]*video_shifted.shape[1],)) for i in range(video_shifted.shape[0])]).T
video_smooth = np.array(pool.map(knnsmooth,video_straight))

ROIs_signal = ROI_extract(video_smooth)
ROIs_normed = np.array(pool.map(normalization,ROIs_signal))


#video = svddnoise(video,num_eiganvalue=20) need to be adjust accordingly
#video_normed = np.array(pool.map(normalization,video_smooth))


if want to check can use makestack save as image stack


"""




