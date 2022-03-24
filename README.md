# Analysis_Zebrafish

code running example


directory = ""

files = listallfiles(directory)
video = readandconvert(files)
video = driftcorrect(video)
video = BackExtract(video)
video = data.astype("int")
video_shifted = BackExtract(video)
video_straight = np.array([video_shifted[i,:,:].reshape((video_shifted.shape[1]*video_shifted.shape[1],)) for i in range(video_shifted.shape[0])]).T
video_smooth = np.array(pool.map(knnsmooth,video_straight))
video = svddnoise(video,num_eiganvalue=20) need to be adjust accordingly
video_normed = np.array(pool.map(normalization,video_smooth))


if want to check can use makestack save as image stack
