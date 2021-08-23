#!/usr/bin/env python
# coding: utf-8

# # FormantNet Data Handling Functions

# The following functions read in and normalize the data. This code is like that used in the PaPE2021 experiments, and unlike the IAIF-preprocessed data used in the IS2021 experiments, in that it reads in raw wavefiles and then converts them into the model input, which consists of log-scale (dB) smoothed spectral envelopes calculated from each frame of input, as described in the IS2021 paper.

# In[ ]:


import numpy as np
import tensorflow as tf
import sys


# **smooth_spenvl()** is used by **getdata()** to smooth the spectral envelopes extracted from the wavefiles. The number of smoothing passes is controlled by *npasses* (ultimately by cfg.ENV_SMOOTH_PASSES, by default 6).

# In[ ]:


def smooth_spenvl(data, npasses):
    npt=len(data)
    for jp in range(npasses):
        dm1 = data[0]
        dz = data[1]
        for j in range(1, npt-1):
            dp1 = data[j+1]
            if dz < dm1 and dz < dp1:
                data[j] = 0.5 * (dm1 + dp1)
            dm1 = dz
            dz = dp1
            
        dm1 = data[0]
        dz = data[1]
        for j in range(1, npt-1):
            dp1 = data[j+1]
            if dz <= dm1 or dz <= dp1:
                data[j] = 0.25*dm1 + 0.5*dz + 0.25*dp1
            dm1 = dz
            dz = dp1
            
    return data         


# **getdata()** is used to read the input wavefiles and extract the smoothed spectral envelopes. Configuration parameters determine how much pre-emphasis, if any, is used (PREEMPH), window width (WINDOW_LENGTH_MSEC), window spacing (WINDOW_STRIDE_MSEC), the amount of envelope smoothing (ENV_SMOOTH_PASSES), and whether that smoothing is done before or after the linear-scale envelopes are converted to decibels (SMOOTH_LINEAR). The ultimate length of the envelope (number of frequency bins) is controlled by SPECTRUM_NPOINTS: if the user wants to use a frequency range smaller than the Nyquist frequency, then the higher-frequency bins are simply removed from the final envelopes as desired.
# 
# Finally, note that getdata() concatenates all input files into one single NumPy array, with nothing to indicate boundaries between input files; model training sequences may thus start in one file and end in the next. But also note that the filelist may be a single file (which is necessary for evaluation -- see **FN_model.track_files()**).

# In[ ]:


#scaling_constant = 2.0 / tf.reduce_sum(tf.signal.hann_window(window_length=cfg.WINDOW_LENGTH_SAMPLES, 
#                                                             periodic=False)).numpy()

def getdata(filelist, cfg, verbose=1):
    scaling_constant = 2.0 / tf.reduce_sum(tf.signal.hann_window(window_length=cfg.WINDOW_LENGTH_SAMPLES, 
                                                                 periodic=False)).numpy()
    if verbose > 0:
        import time
        start_time = time.perf_counter()
        nfiles = len(filelist)
        tenper = nfiles // 10
    datalist = []
    for j, f in enumerate(filelist):
        audio_binary = tf.io.read_file(f)
        aud0, fs = tf.audio.decode_wav(audio_binary)
        normed_audio = tf.squeeze(aud0, axis = -1)
        audio = normed_audio * 32768.0
        pe_audio = audio.numpy()
        pe_audio[1:] = pe_audio[1:] - cfg.PREEMPH * audio[:-1].numpy()
        audio_padded = np.hstack((np.zeros(( cfg.WINDOW_LENGTH_SAMPLES // 2 )), pe_audio, 
                                  np.zeros(( cfg.FRAME_STRIDE_SAMPLES - 1 ))))
            
        specs = tf.abs(tf.signal.stft(audio_padded, frame_length=cfg.WINDOW_LENGTH_SAMPLES, 
                                      frame_step=cfg.FRAME_STRIDE_SAMPLES, 
                                      fft_length=cfg.WINDOW_LENGTH_SAMPLES, pad_end=False)).numpy()
        
        # Conversion from linear to log scale (before smoothing)
        if not cfg.SMOOTH_LINEAR:
            specs = 20.0 * np.log10(scaling_constant * specs + cfg.FLOOR)
        
        #Spectral envelope smoothing
        for i in range(specs.shape[0]):
            datalist += list(smooth_spenvl(specs[i], cfg.ENV_SMOOTH_PASSES))
            
        if verbose > 0 and (j+1) % tenper == 0 and j+1 < nfiles:
            minutes = (time.perf_counter() - start_time) / 60.0
            print("  Loaded {} ({}%) of {} files (time: {:.2f} min)"
                  .format(j+1, ((j+1)*100)//nfiles, nfiles, minutes))
            sys.stdout.flush()

    dataset = np.array(datalist, dtype=np.float32).reshape(-1, specs.shape[1])
    
    # Conversion from linear to log scale (after smoothing)
    if cfg.SMOOTH_LINEAR:
        dataset = 20.0 * np.log10(scaling_constant * dataset + cfg.FLOOR)

    # If user requested a restricted frequency range, remove the extra frequency bins
    dataset = dataset[:,:cfg.SPECTRUM_NPOINTS]
    
    if (verbose > 0):
        minutes = (time.perf_counter() - start_time) / 60.0
        print("Loaded {} files (Total time: {:.2f} min)".format(nfiles, minutes))
        sys.stdout.flush()
    return specs.shape[1], dataset


# **getstats()** is used to calculate and print some stats on the dataset. Note that the mean and standard deviation of the training set calculated with this function is used to normalize all training and evaluation data prior to being fed to the model.

# In[ ]:


def getstats(d):
    print("Statistics of log-scale data:")
    print("  Shape:", d.shape)
    print("  Range:", np.min(d), "-", np.max(d))
    mean=np.mean(d)
    stdev=np.std(d)
    print("  Mean:", mean)
    print("  Stdev:", stdev)
    sys.stdout.flush()
    return mean, stdev


# **get_batched_data()** is used to complete the entire process of extracting, batching, and normalizing data. The initial extraction is performed by **getdata()**. Then the Keras function tf.keras.preprocessing.timeseries_dataset_from_array() is used to convert the data sets from single NumPy arrays into tf.data.Datasets of batches of **SEQUENCE_LENGTH** sequences. These are then converted by normdata() into 2-tuples of input and evaluation datasets. If a mean and standard deviation is provided to the function, then those are used to normalize the data; otherwise, the dataset's own statistics are used for normalization. Note that the data that will be fed as input to the model are the normalized frames (the first item in the **batched_dset** tuple), but the data used as the targets for evaluation are the original **un**-normalized frames (the second item in the **batched_dset** tuple). Since RNNs output sequences, both input and output data are 3-dimensional (batch_length * sequence_length * features).

# In[ ]:


def get_batched_data(filelist, cfg, trmean=None, trstd=None):
    
    npoints, dat1 = getdata(filelist, cfg)
    mean, std = getstats(dat1)
    
    batched_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=dat1, targets=None, sequence_length=cfg.SEQUENCE_LENGTH, 
        sequence_stride=cfg.SEQUENCE_STRIDE, batch_size=cfg.BATCH_SIZE)
    
    if trmean is None:
        trmean = mean
        trstd = std
        
    def normdata(data, mean=trmean, sd=trstd):
        normed_data = (data - mean) / sd
        return normed_data, data

    batched_dset = batched_dset.map(normdata).cache()
    
    del dat1
    return batched_dset, trmean, trstd

