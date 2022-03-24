# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:44:29 2020

@author: Yang Liu, Yongkai Chen
"""



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
    info = np.iinfo(data.dtype) # Get the information of the incoming image type
    tragetuplimit =  np.iinfo(np.uint8)
    tragetmin = np.around(data.min()/info.max*tragetuplimit.max)
    tragetmax = np.around(data.max()/info.max*tragetuplimit.max)
    data = convert(data,tragetmin,tragetmax,np.uint8)
    return data

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
        final[i] = correctedimg.astype('float32')
    return final

def makestack(data):
    row,col = np.shape(data)
    data = data.T.reshape((int(col),int(np.sqrt(row)),int(np.sqrt(row)))).astype('float32')
    return data

def knnsmooth(signal, width = 7):#, estimate_step = 1):
    X_estimate = np.array([np.median(signal[max(i-width,0):min(i+width,signal.shape[0])]) for i in range(0,signal.shape[0])])
    return X_estimate

def smoothrow (data):
    start = time.perf_counter()
    row,col =np.shape(data)
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(knnsmooth, args=(data[x,:],7)) for x in np.arange(row)]
    pool.close()
    stop = time.perf_counter()
    results = np.array(results).astype('float32')
    print(stop-start)
    return results

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

def BackExtract(video,BackSize = 10):
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
    if len(Result)>0:
        for i in range(0,len(Result)):
            bpt = Result[i]-1
            shift_img = np.median(video[max(bpt-100,0):bpt,:,:],0)-np.median(video[bpt:min(bpt+100,video.shape[0]),:,:],0)
            shift_matrix_extend = np.tile(shift_img,(video.shape[0]-bpt,1,1)).astype("int")
            shift_matrix[bpt:,:,:] += shift_matrix_extend
    shift_matrix = shift_matrix.astype("int")
    video_new = video+shift_matrix
    return(video_new)


def F_baseline(value):
    fps = 34
    #width is sliding window size in seconds, fps is imaging speed 50 fps#
    width = 10
    num_points = value.shape[0]
    framenum = int(np.round(width*fps)-1)
    newvalue = np.zeros((num_points-framenum))
    for i in np.arange(framenum,num_points):
        normdata = value[i-framenum:i]
        #get lower 50% value #
        a = np.percentile(normdata, 50, interpolation='midpoint') 
        b = normdata[normdata<a];
        value_base = np.mean(b)
        newvalue[i-framenum] = value_base
    return newvalue


def Grid_extract(video_straight, region_size = 4, num_of_regions = 64*64):
    num_of_pixel = video_straight.shape[0]
    num_of_frame = video_straight.shape[1]
    region_mask = np.zeros((num_of_regions,num_of_pixel))
    count = 0
    hei = 256
    wid = 256
    #for i in range(int(np.sqrt(num_of_regions))):
    for i in range(int(hei/region_size)):
        for j in range(int(wid/region_size)):
            mask_tmp = np.zeros((hei,wid))
            mask_tmp[(region_size*i):(region_size*(i+1)),(region_size*j):(region_size*(j+1))] = 1
            region_mask[count,:] = mask_tmp.reshape((num_of_pixel,))
            count = count+1
    Grids_signal = np.dot(region_mask, video_straight)/(region_size**2)
    return (Grids_signal,region_mask)


def Mapping(atlas, target_img):
    mapping = sdr.optimize(target_img,atlas)
    return mapping

def mask_seg(mapping,mask_atlas):
    mask_left = copy.deepcopy(mask_atlas)
    mask_right = copy.deepcopy(mask_atlas)
    mask_right[:,0:128] = 0
    mask_left[:,128:256] = 0
    corrected_mask_right = mapping.transform(mask_right)
    corrected_mask_left = mapping.transform(mask_left)
    return corrected_mask_right,corrected_mask_left

def croscor(brainmap,data,num):
    z,y,x = np.shape(brainmap)
    length = len(np.arange(0,z,num))
    final = np.zeros([length,y,x])
    n=0
    value =[]
    for i in np.arange(0,z,num):
        final[n]=ndi.filters.convolve(brainmap[i]*1.0,data*1.0)
        value.append(np.max(final[n]))
        n=n+1
    pos = np.argmax(value)
    return pos

def normalization(value,fps = 50): 
    #width is sliding window size in seconds, fps is imaging speed 50 fps#
    width = 10
    num_points = value.shape[0]
    framenum = int(np.round(width*fps))
    newvalue = np.zeros((num_points-framenum))
    for i in np.arange(framenum,num_points):
        normdata = value[i-framenum:i]
        #get lower 50% value #
        a = np.percentile(normdata, 50, interpolation='midpoint') 
        b = normdata[normdata<a];
        norvalue = value[i]/np.mean(b)-1;
        newvalue[i-framenum] = norvalue
    return newvalue

def colormap(img,cmap):
    img = img.astype("uint8")
    rows, cols = img.shape
    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))
    indx = np.argwhere(cmap!=0)
    img_color[indx[:,0],indx[:,1],0]=0#(img_color[indx[:,0],indx[:,1],0]*0.5).astype("uint8")# assign different color according to the value of the msk
    img_color[indx[:,0],indx[:,1],1]=0
    return img_color

def SparsityGraph(ROI_signals,fps,thresh = 0.5, width = 20,step = 10):
    length = ROI_signals.shape[1]
    connect_within = 0
    connect_between = 0
    num_windows = len(range(0,length - width*fps, 3*fps))
    for tp in range(0,length - width*fps,3*fps):
        signal_sub = ROI_signals[:,tp:(tp+width*fps)]
        d = pd.DataFrame(data=signal_sub.T)
        dcorr = abs(np.array(d.corr(method='pearson')))
        connect_within = connect_within + sum(sum(dcorr[0:4,0:4]>thresh))+ sum(sum(dcorr[4:8,4:8]>thresh))-8
        connect_between = connect_between + sum(sum(dcorr[0:4,4:8]>thresh)) + sum(sum(dcorr[4:8,0:4]>thresh))
    return ((connect_within/(24*num_windows),connect_between/(32*num_windows)))

def NumActivity(Signal,mu,sd,wd):
    states = (np.array([np.max(Signal[i*wd:(i+1)*wd])-mu>2*sd for i in range(len(Signal)//wd)]))+0
    state_ch = (states[1:]-states[:-1])==1
    nums = np.sum(state_ch)
    return nums

def DurationActivity(Signal,mu,sd,wd):
    states = (np.array([np.max(Signal[i*wd:(i+1)*wd])-mu>2*sd for i in range(len(Signal)//wd)]))+0
    nums = np.sum(states)
    return nums

def FFT(Signal):
    states = (np.array([np.max(Signal[i*wd:(i+1)*wd])-mu>2*sd for i in range(len(Signal)//wd)]))+0
    nums = np.sum(states)
    return nums

def getsp(value):
    intensities_DC_removed = value - np.mean(value)
    fft_output = np.fft.fft(intensities_DC_removed)
    fft_abs = np.abs(fft_output.real)
    fft_percent = (fft_abs / np.sum(fft_abs)) / np.max((fft_abs / np.sum(fft_abs)))
    thresh = ((np.mean(fft_abs[261:]) + np.std(fft_abs[261:])) / np.sum(fft_abs[261:])) / np.max((fft_abs[261:] / np.sum(fft_abs[261:])))
    return fft_percent
