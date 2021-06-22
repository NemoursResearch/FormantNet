#!/usr/bin/env python
# coding: utf-8

# # Code for PaPE2021 non-IAIF models (RNN model with 1 unidirectional LSTM layer, with or without pre-emphasis, with added delta-frequency loss)
# 
# The Python code (run under TensorFlow 2.4) that was used to train and evaluate the non-IAIF models reported to PaPE 2021 is given below. The code is unaltered, except that (1) comments have been added, and (2) code used solely to evaluate the trained model on non-TIMIT data has been removed.
# 
# Note that the code makes some assumptions based on the circumstances of our computational setup at the time (e.g. file names and locations, etc.) and so cannot be run as-is without the same setup. You may also notice differences in code between the IS2021 models and these models. Some of these differences are due to the necessary differences between the experiments, of course, while other differences are irrelevant to the training and evaluation, and are simply due to the evolution of the code over time, e.g. to make the code more readable and generalizable. We intend to provide a more uniform and user-friendly version of the code for general use soon.
# 
# ### Execution:
# This script was run with five command-line parameters that indicate the number of formants and antiformants; whether spectral smoothing should be done on the linear spectra ("T") or the log spectra ("F"; we recommend "T"); whether to use pre-emphasis or not ("T" or "F"); and finally the delta-frequency loss weighting parameter W as a percentage. The output (consisting of data statistics, model specifications, and script progress reports, including training and validation loss) is saved to an output file, e.g.:
# 
# LSTM1_noIAIF_DFLoss.py 6 1 T F 10 > LSTM1DF.f6z1TF10.out
# 
# It needs to be run on a server with access to Tensorflow 2.3 or above. On a GPU it takes us about 4 hours total to run; on a CPU, it may take half a day or more to run.
# 
# ### Input:
# If the name of the directory in which the script is run is e.g. expdir/, then the script looks for the input data in a sister directory ../data/, where the two input file lists timwavs_ordered2.txt and VTRlist1.txt (described below) should be found. Also in there should be a directory ../data/timit/timitlink/, which holds the TIMIT wavefiles in the same directory structure as in the original corpus, i.e. two subdirectories test/ and train/, each with subdirectories for dialects (dr1-dr8), and each of those with subdirectories for each speaker. Unlike the IAIF-using scripts of IS2021, this script takes the original raw wavefiles as input.
# 
# ### Output:
# The output models and evaluation files are saved to a directory named, e.g. expdir/mvt33_f6z1TF10/ (where "mvt33" was the unique designation for this experiment, and the rest indicates the command-line options chosen). The model files are stored directly in this directory. A subdirectory, expdir/mvt33_f6z1TF10/timit/, in which the output formant track files are stored, one for each input file. These are stored in a format (described below) that was designed to the specific interests of our laboratory, so scripts will be provided that were used to extract the frequencies for evaluation against the VTR-TIMIT database.

# In[ ]:


import numpy as np
import tensorflow as tf
import os
import glob
import subprocess
import sys


# In[ ]:


# The testing parameter is used for quick testing of the code in e.g. a Jupyter Lab window. 
# If testing is set to True, then only a small number of input files are loaded, a few
# iterations of training are run, and only a few test files are evaluated.

#testing = True
testing = False


# In[ ]:


# Variables that differ among subexperiments:
expname = "mvt33"     # A unique designation for the experiment -- used to name the output directories and files below

# Subexperiment parameters, given either in the script or as command-line parameters:
# We tested various numbers of formants and zeros, but ultimately settled on 6 formants and 1 zero for the TIMIT dataset.
# We recommend leaving smooth_lin as "T" (spectral smoothing done on linear rather than log spectra).
# Users may want to experiment with pre-emphasis and the delta-loss weight; see our paper.
if testing is True:
    NFORMANTS = 6
    NZEROS = 1
    smooth_lin = "T"
    preemphasis = "F"
    weightper = 10
else:
    NFORMANTS = int(sys.argv[1])
    NZEROS = int(sys.argv[2])
    smooth_lin = sys.argv[3]
    preemphasis = sys.argv[4]
    weightper = int(sys.argv[5])


# **Input filelist:** The input filelist "timwavs_ordered2.txt" has the following format:
# 
# *timit/timitlink/train/dr1/fecd0/sa1  
# timit/timitlink/train/dr1/fecd0/sa2  
# timit/timitlink/train/dr1/fecd0/si1418  
# timit/timitlink/train/dr1/fecd0/si2048  
# timit/timitlink/train/dr1/fecd0/si788*  
# 
# The code further down below assumes 6300 files, in the order train (4140), validation (480), test (2040).
# The evaluation filelist VTRlist1.txt has the same format, except with the suffix ".wav" added to each line, and it only lists the 516 files included in the VTR-TIMIT corpus, in any order.
# 
# **Sequences:** For RNN models (e.g. LSTMs), a training **SEQUENCE_LENGTH** of 64 is specified; the training set is split into non-overlapping sequences of frames of this length (though the final model can accept sequences of any length for evaluation).

# In[ ]:


NSUM = NFORMANTS + NZEROS
NPARAMS = NFORMANTS*3 + NZEROS*2

#Creation of output directory name (checkpoint_dir)
subexp = "f" + str(NFORMANTS) + "z" + str(NZEROS) + "s" + smooth_lin + "p" + preemphasis + str(weightper)

DIFFWEIGHT = weightper / 100.0

smlog = prelog = False
if smooth_lin == "T":
    smlog = True
if preemphasis == "T":
    prelog = True

if testing is True:
    checkpoint_dir = expname + "_tmp_" + subexp
else:
    checkpoint_dir = expname + "_" + subexp

datadir = "../data/"
filelist = 'timwavs_ordered2.txt'

# Other variables:
n_context_frames = 0  # only used for CNNs (see IS2021 code)
window_length = n_context_frames * 2 + 1  # only used for CNNs (see IS2021 code)
SEQUENCE_LENGTH = 64
BATCH_SIZE = 32
top_activation='sigmoid'
floor = 0.001


# In[ ]:


print("")
print(expname + " OUTPUT:")
print("Formants:", NFORMANTS)
print("Zeros:", NZEROS)
print("Smooth linear rather than log spectra:", smlog)
print("Use preemphasis:", prelog)
print("Differential weight:", DIFFWEIGHT)
print("Number of context spectra on either side:", n_context_frames)
print("Total window length:", window_length)
print("Sequence length:", SEQUENCE_LENGTH)
print("Data from:", datadir)
print("Data list:", filelist)
print("Saved in:", checkpoint_dir)
print("")


# In[ ]:


with open(datadir + filelist) as f:
    allfileset = [i[:-1] + '.wav' for i in list(f)]

with open(datadir + 'VTRlist1.txt') as f:
    vtrfileset = [i[:-1] for i in list(f)]

if testing is True:
    vtrfileset = vtrfileset[:10]


# ### Datasets -- log scale spectra:
# 
# The following code reads in and normalizes the training and validation data. Unlike the IAIF-preprocessed data used in the IS2021 experiments, this code reads in raw wavefiles and then converts them into the model input, which consists of a log-scale (dB) spectral envelope calculated from each frame of input, as described in the IS2021 paper.

# In[ ]:


# This function does the smoothing of the spectral envelope.
def spenvl(data, npasses):
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


# The variable **wlen** below sets the width of the input window (in samples; 512 corresponds to a 32-msec window with 16KHz-sampled speech). The number of data points per frame (the spectral resolution) will half of this number plus one, which for these experiments was kept at a constant 257 points and stored in the variable **npoints** below. **step** determines how often windows are extracted (in samples; 80 samples corresponds to 5 msec), and **pe** sets the degree of pre-emphasis, if it's used (this was also kept constant).

# In[ ]:


wlen=512
step=80
pe=0.98
#scale = tf.cast(2.0 / tf.reduce_sum(tf.signal.hann_window(window_length=wlen, periodic=False)), dtype=tf.float64)
scale = 2.0 / tf.reduce_sum(tf.signal.hann_window(window_length=wlen, periodic=False)).numpy()
#scale


# In[ ]:


# The following function is used to load data and, if needed, add N context frames (ncframes) to each end.
# (Context frames were only used for CNN experiments; see the IS2021 directory.)
# Input include datadir (the common superdirectory for all input files), the filelist, the amount of spectral
# smoothing, and the spectral smoothing and preemphasis flags described agove.
# datdir is prepended to each file in the filelist, and may
# be left as an empty string. The filelist may itself include its own subdirectories. The filelist
# may be a single file (necessary for evaluation, as seen below). Note that getdata() concatenates
# all input files into one single NumPy array, with nothing to indicate boundaries between input files.
# Initial and final context frames (copies of the first and last frames) are added to this entire structure. This is
# necessary because of how tf.keras.preprocessing.timeseries_dataset_from_array() works, which will be
# used to divide the array into input windows (see below).

def getdata(datadir, filelist=[], ncframes=0, smooth=6, verbose=1, smooth_lin=smlog, preemphasis=prelog):
    if verbose > 0:
        import time
        start_time = time.perf_counter()
    datalist = []
    for f in filelist:
        audio_binary = tf.io.read_file(datadir + '/' + f)
        aud0, fs = tf.audio.decode_wav(audio_binary)
        normed_audio = tf.squeeze(aud0, axis = -1)
        audio = normed_audio * 32768.0
        if preemphasis is True:
            pe_audio = audio.numpy()
            pe_audio[1:] = pe_audio[1:] - pe * audio[:-1].numpy()
            audio_padded = np.hstack((np.zeros(( wlen // 2 )), pe_audio, np.zeros(( step - 1 ))))
        else:
            audio_padded = np.hstack((np.zeros(( wlen // 2 )), audio.numpy(), np.zeros(( step - 1 ))))
            
        specs = tf.abs(tf.signal.stft(audio_padded, frame_length=wlen, frame_step=step, fft_length=wlen, pad_end=False)).numpy()
        
        if smooth_lin is False:
            specs = 20.0 * np.log10(scale * specs + floor)
        
        for i in range(specs.shape[0]):
            datalist += list(spenvl(specs[i], smooth))

    dataset = np.array(datalist, dtype=np.float32).reshape(-1, specs.shape[1])
    
    if smooth_lin is True:
        dataset = 20.0 * np.log10(scale * dataset + floor)
    
    if (ncframes > 0):
        x = firstframe = [dataset[0,:]]
        y = lastframe = [dataset[-1,:]]
        for j in range(ncframes-1):
            x = np.concatenate((x, firstframe))
        for j in range(ncframes*2-1):
            y = np.concatenate((y, lastframe))
        dataset = np.concatenate((x, dataset, y))
    
    if (verbose > 0):
        print("Loaded", len(filelist), "files (time:", time.perf_counter() - start_time, ")")
    return specs.shape[1], dataset


# In[ ]:


# A function to get some stats on the dataset. Note that the mean and standard deviation of
# the training set calculated with this function must be used to normalize all training
# and evaluation data prior to being fed to the model.

def getstats(d):
    print("Shape:", d.shape)
    print("Range:", np.min(d), "-", np.max(d))
    mean=np.mean(d)
    stdev=np.std(d)
    print("Mean:", mean)
    print("Stdev:", stdev)
    return mean, stdev


# In[ ]:


print("")
print("Loading validation data ....")
sys.stdout.flush()
if testing is True:
    npoints, val1 = getdata(datadir, allfileset[4140:4200], n_context_frames)
else:
    npoints, val1 = getdata(datadir, allfileset[4140:4620], n_context_frames)
print("Log validation data stats:")
getstats(val1)
print("")


# In[ ]:


print("Loading training data ....")
sys.stdout.flush()
if testing is True:
    len2, train1 = getdata(datadir, allfileset[:50], n_context_frames)
else:
    len2, train1 = getdata(datadir, allfileset[:4140], n_context_frames)
print("Log training data stats:")
trmean, trstd = getstats(train1)
print("")
print("trmean =", trmean)
print("trstd =", trstd)
print("")
sys.stdout.flush()


# We normalize the datasets by the mean and stdev of the training set.
# 

# In[ ]:


def normdata(data, mean=trmean, sd=trstd):
    normed_data = (data - mean) / sd
    return normed_data, data


# The Keras function tf.keras.preprocessing.timeseries_dataset_from_array() is used to convert the training and validation sets from single NumPy arrays into tf.data.Datasets of non-overlapping **SEQUENCE_LENGTH** sequences. Note that the data fed as input to the model are the normalized frames, but the data used as the targets for evaluation are the original **un**-normalized frames. Since RNNs output sequences, both input and output data are 3-dimensional (batch_length * sequence_length * resolution).

# In[11]:


batched_train_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=train1, targets=None, sequence_length=SEQUENCE_LENGTH, sequence_stride=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
batched_val_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=val1, targets=None, sequence_length=SEQUENCE_LENGTH, sequence_stride=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)

batched_train_dset = batched_train_dset.map(normdata)
batched_val_dset = batched_val_dset.map(normdata)

print("")
for batch_input, batch_target in batched_train_dset.take(1):
    print("Input shape:", batch_input.shape)
    print("Target shape:", batch_target.shape)
print("")
sys.stdout.flush()


# In[ ]:


print("Using caching")
batched_train_dset = batched_train_dset.cache()
batched_val_dset = batched_val_dset.cache()
print("")
sys.stdout.flush()


# In[ ]:


del train1, val1
#del train2, val2


# ### Definition of Loss function, etc.
# 
# The functions used to compute the loss are defined here. We tried to write the code so that it could handle variations in sampling rate (srate), frequency range (from 0 to maxfreq), number of formants (FORMANTS), number of anti-formants (ZEROS), spectral resolution (npoints), and the activation type of the final model output layer (myactivation). For the TIMIT experiments, these were all set constant across all experiments: 16K sampling rate, 0-8K frequency range, 6 formants, 1 zero, 257-point spectra, sigmoid activation.

# The formant() function takes the frequency F and bandwidth B of each formant predicted by the model, and generates a corresponding formant: an array of spectrum levels h at each frequency bin f in the spectrum range at the given resolution (see Eqn. (1) of the IS2021 paper). The vtfn() function weights these by their corresponding amplitude factors, and combines them (multiplying or dividing, corresponding to whether it's a pole or zero) to produce a linear-scale spectral envelope.

# In[ ]:


maxfreq=8000
spec1 = tf.cast(np.linspace(0, maxfreq, npoints), dtype=tf.float32)

@tf.function
def formant(freq, bw, nres, npoints=257, maxfreq=8000):
    fex = tf.expand_dims(freq, axis=-1)
    bex = tf.expand_dims(bw, axis=-1)
    bsq = bex**2 * 0.25
    anum = fex**2 + bsq    
    #spec1 = tf.cast(np.linspace(0, maxfreq, npoints), dtype=tf.float32)
    spec2 = tf.tile(spec1, [tf.size(freq)])
    spec = tf.reshape(spec2, [-1, nres, npoints])
    negspec = (spec - fex)**2 + bsq
    posspec = (spec + fex)**2 + bsq
    formants = anum / tf.math.sqrt(negspec * posspec)
    return(formants)

#Note that vtfn returns a LINEAR-scale spectrum
if NZEROS == 0:
    @tf.function
    def vtfn(freqs, bws, amps, npoints=257, srate=16000):
        ax = tf.expand_dims(amps, axis=-1)
        ax = 10.0 ** (ax / 20.0)   #convert log amplitudes to linear
        maxf = srate // 2
        specs = formant(freqs, bws, NFORMANTS, npoints, maxf)
        sumspec = tf.reduce_sum(ax * specs, axis = -2)
        return sumspec
else:
    @tf.function
    def vtfn(freqs, bws, amps, zfreqs, zbws, npoints=257, srate=16000):
        ax = tf.expand_dims(amps, axis=-1)
        ax = 10.0 ** (ax / 20.0)   #convert log amplitudes to linear
        maxf = srate // 2
        fspecs = ax * formant(freqs, bws, NFORMANTS, npoints, maxf)
        sumspecs = tf.reduce_sum(fspecs, axis = -2, keepdims=True)
        zspecs = 1.0 / formant(zfreqs, zbws, NZEROS, npoints, maxf)
        allspecs = tf.concat([sumspecs, zspecs], axis = -2)
        prodspecs = tf.reduce_prod(allspecs, axis = -2)
        return prodspecs


# The rescale_params() function takes the output of the model, and rescales it to the expected scale for formant parameters (e.g. 0-8000 Hz for frequencies). The input scale depends on the output activation of the model; we experimented with linear, tanh, softsign, and ReLU, but found that sigmoid usually works best. Note that this function forces the output of the model to be in the order F1 F2 F3 ... B1 B2 B3 ... A1 A2 A3 .... Note also that this function is needed for evaluation (further below) and any future use of the model.

# In[ ]:


@tf.function(input_signature=(tf.TensorSpec(shape=[None, NPARAMS], dtype=tf.float32),))
def rescale_params(params):
    freqs, bws, amps = tf.split(params, [NSUM, NSUM, NFORMANTS], axis=-1)
    if top_activation == 'sigmoid':    #network produces values between 0 and 1
        freqs = freqs * 8000.0
        bws = (bws * 5000.0) + 20.0
        amps = (amps - 0.5) * 200.0
    elif top_activation == 'softsign' or top_activation == 'tanh':  #network produces values between -1 and 1
        freqs = (freqs + 1.0) * 4000.0
        bws = (bws * 2500.0) + 2520.0
        amps = amps * 100.0
    elif top_activation == 'relu':   #network produces values of 0 or greater. Add 20.0 to avoid division by 0
        bws = bws + 20.0
        amps = amps - 100.0
    return freqs, bws, amps      


# Finally, the model loss is calculated with custom_loss(). First, the batch and sequence dimensions are collapsed. Then the input model parameters are rescaled with rescale_params(). The formants are split into poles and zeros, and sent to vtfn() to compute a linear-scale spectral envelope. The envelope is then converted to decibel scale, and the spectral loss is calculated as the mean square difference between the generated envelope and the original envelope.

# If Delta-frequency loss is being used, it is also calculated here, weighted by the **DIFFWEIGHT** parameter (derived from the weight specified on the command line), and added to the spectral loss to get the final loss. (We also experimented with using delta-loss with bandwidths and amplitudes, but those experiments have been unsuccessful so far.)

# In[ ]:


#Note that the floor is added to the log conversion here.
#def get_custom_loss(myactivation='linear'):
def get_custom_loss():
    if NZEROS == 0:
        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, npoints], dtype=tf.float32),
                                     tf.TensorSpec(shape=[None, None, NPARAMS], dtype=tf.float32)))
        def custom_loss(specs_input, params_pred):
            npoints=specs_input.shape[-1]
            specs_input = tf.reshape(specs_input, [-1, npoints])
            params_pred = tf.reshape(params_pred, [-1, NPARAMS])
            freqs, bws, amps = rescale_params(params_pred)
            specs_pred = vtfn(freqs, bws, amps, npoints=specs_input.shape[-1], srate=16000)
            specs_pred = 20.0 * tf.math.log(floor + specs_pred) / tf.math.log(10.0)
            mse = tf.math.reduce_mean(tf.square(specs_input - specs_pred)) #loss over whole batch
            fdiff = tf.math.reduce_mean(tf.abs(freqs[1:] - freqs[:-1]))
            #bdiff = tf.math.reduce_mean(tf.abs(bws[1:] - bws[:-1]))
            #adiff = tf.math.reduce_mean(tf.abs(amps[1:] - amps[:-1]))
            #return(mse + DIFFWEIGHT * (fdiff + bdiff + 25.0 * adiff))
            return(mse + DIFFWEIGHT * (fdiff)) # + bdiff + 25.0 * adiff))

    else:  
        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, npoints], dtype=tf.float32),
                                     tf.TensorSpec(shape=[None, None, NPARAMS], dtype=tf.float32)))
        def custom_loss(specs_input, params_pred):
            npoints=specs_input.shape[-1]
            specs_input = tf.reshape(specs_input, [-1, npoints])
            params_pred = tf.reshape(params_pred, [-1, NPARAMS])
            freqs, bws, amps = rescale_params(params_pred)
            pfreqs, zfreqs = tf.split(freqs, [NFORMANTS, NZEROS], axis=-1)
            pbws, zbws = tf.split(bws, [NFORMANTS, NZEROS], axis=-1)
            specs_pred = vtfn(pfreqs, pbws, amps, zfreqs, zbws, npoints=specs_input.shape[-1], srate=16000)
            specs_pred = 20.0 * tf.math.log(floor + specs_pred) / tf.math.log(10.0)
            mse = tf.math.reduce_mean(tf.square(specs_input - specs_pred)) #loss over whole batch
            fdiff = tf.math.reduce_mean(tf.abs(freqs[1:] - freqs[:-1]))
            #bdiff = tf.math.reduce_mean(tf.abs(bws[1:] - bws[:-1]))
            #adiff = tf.math.reduce_mean(tf.abs(amps[1:] - amps[:-1]))
            #return(mse + DIFFWEIGHT * (fdiff + bdiff + 25.0 * adiff))
            return(mse + DIFFWEIGHT * (fdiff)) # + bdiff + 25.0 * adiff))
    
    return custom_loss


# ### Build and train model

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dense(NFORMANTS*3 + NZEROS*2, activation=top_activation),
])


# In[17]:


print("Input shape: ", batch_input.shape)
print("Output shape: ", model(batch_input).shape)
print("")
sys.stdout.flush()


# In[18]:


model.summary()


# In[ ]:


myloss = get_custom_loss()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=myloss, metrics=[myloss]
)


# ### If a restart, reload last model

# In[ ]:


reload = True
if testing is True:
    last_epoch = 0
    reload = False


# In[ ]:


checkpoints = glob.glob(checkpoint_dir + "/*.index")
if len(checkpoints) > 0 and reload is True:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    model.load_weights(latest_checkpoint[:-6]) #remove .index suffix
    print("Reloading from", latest_checkpoint)
    sys.stdout.flush()
    last_epoch=int(latest_checkpoint.partition('-')[0][-3:])
else:
    last_epoch=0


# The trained model is saved after every epoch that produces a validation loss lower than that of any previous epoch. Models were trained until the best best validation loss was not improved after 20 epochs (patience=20), or a maximum of 200 epochs.

# In[ ]:


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir + "/weights." + top_activation + "{epoch:03d}-{val_custom_loss:.3f}",
    save_best_only=True, save_weights_only=True,
    monitor='val_custom_loss', mode='min')

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor='val_custom_loss', mode='min')

if testing is True:
    EPOCHS = 3
    VERBOSITY = 1
else:
    EPOCHS = 200
    VERBOSITY = 2

model.fit(batched_train_dset, epochs=EPOCHS, verbose=VERBOSITY,
          callbacks=[model_checkpoint_callback, early_stopping_callback], 
          validation_data=batched_val_dset)


# ## Restore best model and evaluate

# In[ ]:


print("")
sys.stdout.flush()
checkpoints = glob.glob(checkpoint_dir + "/*.index")
latest_checkpoint = max(checkpoints, key=os.path.getctime)
model.load_weights(latest_checkpoint[:-6]) #remove .index suffix
print("Restoring from", latest_checkpoint)
sys.stdout.flush()

train_eval = model.evaluate(batched_train_dset, verbose=0)
print("Training loss:", train_eval[0])
sys.stdout.flush()

val_eval = model.evaluate(batched_val_dset, verbose=0)
print("Validation loss:", val_eval[0])
sys.stdout.flush()


# At this point, the TIMIT test data is loaded and tested. Note that all input data fed to the model must first be normalized using the mean and standard deviation of the original training data, so those values need to be recorded for evaluation (see also below) and any future model use.  (Here they are stored in the function normdata().)

# In[ ]:


if testing is True:
    len3, test1 = getdata(datadir, allfileset[4620:4680], n_context_frames, verbose=1)
else:
    len3, test1 = getdata(datadir, allfileset[4620:6300], n_context_frames, verbose=0)

batched_test_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
    test1, targets=None, sequence_length=SEQUENCE_LENGTH, sequence_stride=SEQUENCE_LENGTH, batch_size=BATCH_SIZE)
batched_test_dset = batched_test_dset.map(normdata)

test_eval = model.evaluate(batched_test_dset, verbose=0)
print("Test loss (TIMIT):", test_eval[0])


# ## Generate TIMIT data
# 
# For evaluation, the models were run on the TIMIT recordings whose formants were measured for the VTR-TIMIT formant database. The output files are generated in a subdirectory /timit/ of the output directory (*checkpoint_dir*). The filename will be prefixed by *checkpoint_dir* and have the suffix ".abs". These files are text files, with one output line for every window of speech extracted from the input file (every 5 msec with the default settings above). The columns of the output file consist of the following in order: the filename, 3 placeholder columns (for the particular needs of our research), the time point in milliseconds, 7 more placeholder columns, and then the total number of resonances (poles plus zeros). This is followed by the parameters (frequency, bandwidth, and amplitude correction factor, in that order) of the poles, in order of increasing mean frequency, and then those of the zeros, in order of increasing mean absolute frequency.
# 
# Other notes:
# * For output interpretation, it's important to remember that the generated "amplitudes" are not actually final formant amplitudes, but rather weighting factors that are used to adjust the initial formant amplitudes generated by formant().
# * The following code changes the frequencies of the zeros to negative values, to distinguish them from the poles. Also, since the zeros don't have their own amplitude correction factors, a placeholder value of "0.0" is inserted (theoretically we should have used 1.0 instead, but this value is not used in any computations).
# * The output code below assumes a frame rate of one per 5 milliseconds, which is the rate we used for our input data. (However, the VTR TIMIT measurements were taken once per 10 milliseconds, so every other output frame was used for evaluation.)
# * Since there is nothing in the custom loss code above that distinguishes one formant from another (aside from poles versus zeros), and any of them can take frequency values between 0 and 8000, the model output neurons may generate the formants in any random order (although that order will be constant from one frame to the next; e.g. if neuron 3 generates F1 for one frame, it does so for all frames and files).  The code below reorders the formants by their mean frequencies over all frames.

# In[ ]:


print("Generating", len(vtrfileset), ".abs files on TIMIT data:")

cmd = ["mkdir", "-p", checkpoint_dir + "/timit"]
subprocess.run(cmd)
datadir = "../data/timit/timitlink/"
for filename in vtrfileset:
        shortname=filename[:-4].replace("/","_")
        outname=checkpoint_dir + "/timit/" + checkpoint_dir + "_" + shortname

        #Note that we feed the files one at a time to getdata()
        lenf, f1 = getdata(datadir, [filename], n_context_frames, verbose=0)
        #Again, the input data must be normalized by the training set statistics
        f2 = (f1 - trmean) / trstd
        #Add a third dimension so that frame sequence can be read directly by the model
        f2 = tf.expand_dims(f2, axis=0)
        y = model.predict(f2)[0]
        # Rescale and reorganize output
        f, b, a = rescale_params(y)
        zf=f.numpy()
        za=a.numpy()
        zb=b.numpy()        

        # Convert zero frequencies to negative numbers and insert placeholder "0.0" for zero amplitudes.
        # Then sort formants and zeroes separately by increasing frequency.
        if NZEROS > 0:
            fp, f0 = tf.split(zf, [NFORMANTS, NZEROS], axis=-1)
            f0 = f0 * -1.0
            zf = tf.concat([fp, f0], axis=-1)
            a0 = tf.zeros([za.shape[0], NZEROS], dtype=tf.float32)
            za = tf.concat([za, a0], axis=-1)    
            ord = np.hstack((np.argsort(np.mean(fp, axis=0)), (np.flip(np.argsort(np.mean(f0, axis=0)) + NFORMANTS))))
        else:
            ord = np.argsort(np.mean(np.abs(zf),axis=0))
        
        print("FILE:", shortname)
        sys.stdout.flush()

        #Re-sort parameters in the order F1 B1 A1, F2 B2 A2, etc. and write to output file
        p = [(i, i+NSUM, i+(NSUM*2)) for i in ord]
        p2=sum(p, ())
        zp=np.hstack((zf, zb, za))
        out1=zp[:,p2]
        ff = open(outname + ".abs", "w")
        for i in range(out1.shape[0]):
            ff.write("{} AA 1 1 {:.1f} 200.0 60 0 0 60 40 2 {}   ".format(shortname, i*5.0, NSUM))
            out1[i,:].tofile(ff, sep=" ")
            ff.write(" \n")
        ff.close()


# In[ ]:


print("")
print("FINISHED script for", checkpoint_dir)

