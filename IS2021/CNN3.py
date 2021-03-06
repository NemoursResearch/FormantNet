#!/usr/bin/env python
# coding: utf-8

# # Code for IS2021 CNN3 model (CNN model with 3 convolutional layers)
# 
# The Python code (run under TensorFlow 2.3) that was used to train and evaluate the CNN3 model submitted to Interspeech 2021 is given below. The code is unaltered, except that (1) comments have been added, and (2) code used solely to evaluate the trained model on non-TIMIT data has been removed.
# 
# Note that the code makes some assumptions based on the circumstances of our computational setup at the time (e.g. file names and locations, etc.) and so cannot be run as-is without the same setup. You may also notice differences in code between the four models. Some of these differences are due to the necessary differences between the 4 experiments, of course, while other differences are irrelevant to the training and evaluation, and are simply due to the evolution of the code over time, e.g. to make the code more readable and generalizable. An updated, generalized, and user-friendly version of the code for general public use has been provided in the **../User/** directory.
# 
# ### Execution:
# This script was run with two command-line parameters that indicate the number of formants and antiformants, and the output (consisting of data statistics, model specifications, and script progress reports, including training and validation loss) is saved to an output file, e.g.:
# 
# CNN3.py 6 1 > CNN3.f6z1.out
# 
# It needs to be run on a server with access to Tensorflow 2.3. On a GPU, the script make take a few hours; on a CPU, the script may take several hours to a few days to run. (In our case, running the script on CPU machines using 24 parallel cores per job, these experiments ran between half a day and 2 days.)
# 
# ### Input:
# If the name of the directory in which the script is run is e.g. expdir/, then the script looks for the input data in a sister directory ../data/, where the two input file lists timit_ordered1.txt and VTRlist0.txt (described below) should be found. Also in there should be a directory ../data/timit/srcflt_r480_ENV_db68/vtspecs/, which holds the individual spectral envelope files derived from each TIMIT wavefile (also described below).
# 
# ### Output:
# The output models and evaluation files are saved to a directory named expdir/mvt24_f6z1/ (where "mvt24" was the unique designation for this experiment, and "f6z1" indicates 6 formants and 1 zero). The model files are stored directly in this directory. A subdirectory, expdir/mvt24_f6z1/timit/, will hold the output formant track files, one for each input file. These are stored in a format (described below) that was designed for the specific interests of our laboratory, so scripts are provided that were used to extract the frequencies for evaluation against the VTR-TIMIT database.

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
expname = "mvt24"    # A unique designation for the experiment -- used to name the output directories and files below

# Subexperiment parameters, given either in the script or as command-line parameters. 
# We tested various numbers of formants and zeros, but ultimately settled on 6 formants and 1 zero for IS2021.
if testing is True:
    NFORMANTS = 6
    NZEROS = 1
else:
    NFORMANTS = int(sys.argv[1])
    NZEROS = int(sys.argv[2])


# **Input filelist:** The input filelist "timit_ordered1.txt" has a format where the directory and suffix have been removed, e.g.:
# 
# *train_dr1_fecd0_sa1  
# train_dr1_fecd0_sa2  
# train_dr1_fecd0_si1418  
# train_dr1_fecd0_si2048  
# train_dr1_fecd0_si788*  
# 
# The code further down below assumes 6300 files, in the order train (4140), validation (480), test (1680).
# The evaluation filelist VTRlist0.txt has the same format, except it only lists the 516 files included in the VTR-TIMIT corpus, in any order.
# 
# **Context frames and sequences:** For CNN models, the model input for each time-step was set up so that it included not only the target frame, but also the N preceding frames and N following frames, for context. This is controlled by the variable **n_context_frames** below, and the total length of the input (2\*N+1) is stored in **window_length**. For CNN3, n_context_frames was 10 (window_length 21). Each of these windows overlaps with the next, e.g. the window for frame 50 includes frames 40-60, the window for frame 51 includes frames 41-61, and so on. For RNNs, n_context_frames was 0 (window_length 1), but a training **SEQUENCE_LENGTH** of 64 is specified; the training set is split into non-overlapping sequences of frames of this length (though the final model can accept sequences of any length for evaluation).

# In[ ]:


NSUM = NFORMANTS + NZEROS

#Creation of output directory name (checkpoint_dir)
subexp = "f" + str(NFORMANTS) + "z" + str(NZEROS)
if testing is True:
    checkpoint_dir = expname + "_tmp_" + subexp
else:
    checkpoint_dir = expname + "_" + subexp

#Specifying the input files and directories
if testing is True:
    subdir = "/srcflt_r480_ENV/vtspecs/"
else:
    subdir = "/srcflt_r480_ENV_db68/vtspecs/"   #Where the training files are
superdir = "../data/"  #Where all input files are, including filelists
suffix = "_spec.fea"   #The suffix of the input files
filelist = 'timit_ordered1.txt'   #The input file list

# Other variables:
n_context_frames = 10
window_length = n_context_frames * 2 + 1
BATCH_SIZE = 32
top_activation='sigmoid'   #The activation of the model output layer
floor = 0.001   #Floor value added to linear spectra before conversion to log domain


# In[ ]:


print("")
print(expname + " OUTPUT:")
print("Formants:", NFORMANTS)
print("Zeros:", NZEROS)
print("Number of context spectra on either side:", n_context_frames)
print("Total window length:", window_length)
print("Data list:", filelist)
print("Data:", superdir + "SPKR" + subdir + "FILENAME" + suffix)
print("Saved in:", checkpoint_dir)
print("")


# In[ ]:


#Load the filelists

with open(superdir + filelist) as f:
    allfileset = [i[:-1] for i in list(f)]
    datadir = superdir + "timit" + subdir

with open(superdir + 'VTRlist0.txt') as f:
    vtrfileset = [i[:-1] for i in list(f)]
    if testing is True:
        vtrfileset = vtrfileset[:10]


# ### Datasets -- log scale spectra:
# 
# The following code reads in and normalizes the training and validation data, which consist of a log-scale (dB) spectral envelope calculated from each frame of input, as described in our IS2021 paper. There is one input file for each wavefile. The file is in binary format, and starts with a 24-byte header. The first 8 bytes consist of two 4-byte integers specifying first the number of frames, and then the number of data points per frame (the spectral resolution), which for IS2021 was kept at a constant 257 points and stored in the variable **npoints**. Following the header are the spectra themselves, which are stored as float values.

# In[ ]:


# The following function is used to load data and, if needed, add N context frames (ncframes) to each end.
# Input include datadir (the common superdirectory for all input files), the filelist, and a suffix. 
# datadir and suffix are prepended and appended (respectively) to each file in the filelist, and may each
# be left as empty strings. The filelist may itself include its own subdirectories and suffixes. The filelist
# may be a single file (necessary for evaluation, as seen below). If filelist is empty, the function will
# load all files in datadir (in which case suffix should be left empty). Note that getdata() concatenates
# all input files into one single NumPy array, with nothing to indicate boundaries between input files.
# Initial and final context frames (copies of the first and last frames) are added to this entire structure.
# This is necessary because of how tf.keras.preprocessing.timeseries_dataset_from_array() works, which will
# be used to divide the array into input windows (see below).

def getdata(datadir, filelist=[], suffix=".wav", ncframes=0, verbose=1):
    import struct
    import time
    start_time = time.perf_counter()
    datalist = []
    if filelist == []:
        filelist = [name for name in os.listdir(datadir)]
    for f in filelist:
        with open(datadir + '/' + f + suffix, 'rb') as file:
            nspecs, speclen = struct.unpack('ii', file.read(8))
            file.seek(24)
            x = file.read(nspecs*speclen*4)
            datalist += list(struct.unpack(str(nspecs*speclen)+'f', x))

    dataset = np.array(datalist, dtype=np.float32).reshape(-1, speclen)
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
    return speclen, dataset

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
    npoints, val1 = getdata(datadir, allfileset[4140:4200], suffix, n_context_frames)
else:
    npoints, val1 = getdata(datadir, allfileset[4140:4620], suffix, n_context_frames)
print("Log validation data stats:")
getstats(val1)
print("")


# In[ ]:


print("Loading training data ....")
sys.stdout.flush()
if testing is True:
    len2, train1 = getdata(datadir, allfileset[:50], suffix, n_context_frames)
else:
    len2, train1 = getdata(datadir, allfileset[:4140], suffix, n_context_frames)
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


train2 = (train1 - trmean) / trstd
val2 = (val1 - trmean) / trstd

print("Normalized training data stats:")
getstats(train2)
sys.stdout.flush()


# The Keras function tf.keras.preprocessing.timeseries_dataset_from_array() is used to convert the training and validation sets from single NumPy arrays into tf.data.Datasets of overlapping **window_length** windows. Note that the data fed as input to the model are the normalized **window_length** windows, but the data used as the targets for evaluation are the original **un**-normalized target frames. Hence, the input data is 3-dimensional (batch_length * window_length * resolution) but the target data is only 2-dimensional (batch_length * resolution).

# In[11]:


batched_train_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
    train2[:len(train2)-n_context_frames,], train1[n_context_frames:,], sequence_length=window_length, batch_size=BATCH_SIZE)
batched_val_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
    val2[:len(val2)-n_context_frames], val1[n_context_frames:,], sequence_length=window_length, batch_size=BATCH_SIZE)

print("")
for batch_input, batch_target in batched_train_dset.take(1):
    print("Input shape:", batch_input.shape)
    print("Target shape:", batch_target.shape)
print("")
sys.stdout.flush()


# In[ ]:


del train1, train2
del val1, val2


# ### Definition of Loss function, etc.
# 
# The functions used to compute the loss are defined here. We tried to write the code so that it could handle variations in sampling rate (srate), frequency range (from 0 to maxfreq), number of formants (NFORMANTS), number of anti-formants (NZEROS), spectral resolution (npoints), and the activation type of the final model output layer (myactivation). For IS2021, these were all set constant across all experiments: 16K sampling rate, 0-8K frequency range, 6 formants, 1 zero, 257-point spectra, sigmoid activation.
# 
# The code here is a bit different here than in the LSTM models, due to the necessity of having to add another dimension to the data for sequences in the code for those models.

# The formant() function takes the frequency F and bandwidth B of each formant predicted by the model, and generates a corresponding formant: an array of spectrum levels h at each frequency bin f in the spectrum range at the given resolution (see Eqn. (1) of the paper). The vtfn() function weights these by their corresponding amplitude factors, and combines them (multiplying or dividing, corresponding to whether it's a pole or zero) to produce a linear-scale spectral envelope.

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
        sumspec = tf.reduce_sum(ax * specs, axis = 1)
        return sumspec
else:
    @tf.function
    def vtfn(freqs, bws, amps, zfreqs, zbws, npoints=257, srate=16000):
        ax = tf.expand_dims(amps, axis=-1)
        ax = 10.0 ** (ax / 20.0)   #convert log amplitudes to linear
        maxf = srate // 2
        fspecs = ax * formant(freqs, bws, NFORMANTS, npoints, maxf)
        sumspecs = tf.reduce_sum(fspecs, axis = 1, keepdims=True)
        zspecs = 1.0 / formant(zfreqs, zbws, NZEROS, npoints, maxf)
        allspecs = tf.concat([sumspecs, zspecs], axis = 1)
        prodspecs = tf.reduce_prod(allspecs, axis = 1)
        return prodspecs


# The rescale_params() function takes the output of the model, and rescales it to the expected scale for formant parameters (e.g. 0-8000 Hz for frequencies). The input scale depends on the output activation of the model; we experimented with linear, tanh, softsign, and ReLU, but found that sigmoid usually works best. Note that this function forces the output of the model to be in the order F1 F2 F3 ... B1 B2 B3 ... A1 A2 A3 .... Note also that this function is needed for evaluation (further below) and any future use of the model.

# In[ ]:


@tf.function
def rescale_params(params, act='linear'):
    freqs, bws, amps = tf.split(params, [NSUM, NSUM, NFORMANTS], axis=-1)
    if act == 'sigmoid':    #network produces values between 0 and 1
        freqs = freqs * 8000.0
        bws = (bws * 5000.0) + 20.0
        amps = (amps - 0.5) * 200.0
    elif act == 'softsign' or act == 'tanh':  #network produces values between -1 and 1
        freqs = (freqs + 1.0) * 4000.0
        bws = (bws * 2500.0) + 2520.0
        amps = amps * 100.0
    elif act == 'relu':   #network produces values of 0 or greater. Add 20.0 to avoid division by 0
        bws = bws + 20.0
        amps = amps - 100.0
    return freqs, bws, amps      


# Finally, the model loss is calculated with custom_loss(). The input model parameters are first rescaled with rescale_params(). The formants are split into poles and zeros, and sent to vtfn() to compute a linear-scale spectral envelope. The envelope is then converted to decibel scale, and the loss is calculated as the mean square difference between the generated envelope and the original envelope.

# In[ ]:


#Note that the floor is added to the log conversion here.
def get_custom_loss(myactivation='linear'):
    if NZEROS == 0:
        @tf.function
        def custom_loss(specs_input, params_pred):
            freqs, bws, amps = rescale_params(params_pred, act=myactivation)
            specs_pred = vtfn(freqs, bws, amps, npoints=specs_input.shape[-1], srate=16000)
            specs_pred = 20.0 * tf.math.log(floor + specs_pred) / tf.math.log(10.0)
            return(tf.math.reduce_mean(tf.square(specs_input - specs_pred))) #loss over whole batch

    else:  
        @tf.function
        def custom_loss(specs_input, params_pred):
            freqs, bws, amps = rescale_params(params_pred, act=myactivation)
            pfreqs, zfreqs = tf.split(freqs, [NFORMANTS, NZEROS], axis=-1)
            pbws, zbws = tf.split(bws, [NFORMANTS, NZEROS], axis=-1)
            specs_pred = vtfn(pfreqs, pbws, amps, zfreqs, zbws, npoints=specs_input.shape[-1], srate=16000)
            specs_pred = 20.0 * tf.math.log(floor + specs_pred) / tf.math.log(10.0)
            return(tf.math.reduce_mean(tf.square(specs_input - specs_pred))) #loss over whole batch
    
    return custom_loss


# ### Build and train model

# In[16]:


# With CNN models, we experimented with different window sizes. The following code was
# used to adjust the parameters of the max-pooling layers in the time (window length)
# dimension accordingly to handle different window sizes and make sure the output of
# the final pooling layer was of a manageable size. For IS2021, window_length was set to 21.

tp1 = tp2 = tp3 = tp4 = 1
if window_length - 2 > 2: tp3 = 3
if window_length - 4 > 2: tp2 = 3
if window_length - 6 > 2: tp1 = window_length - 6
print(tp1, tp2, tp3, tp4)


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Reshape([window_length, npoints, 1],
        input_shape=tf.constant((window_length, npoints))),
    tf.keras.layers.Conv2D(16, (1,3), activation='relu', strides=(1,1), padding='valid'),
    tf.keras.layers.MaxPool2D(pool_size=(tp1,2), strides=(1,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', strides=(1,2), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(tp2,2), strides=(1,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=(1,2), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(tp3,2), strides=(1,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(NFORMANTS*3 + NZEROS*2, activation=top_activation),
])


# In[18]:


model.summary()


# In[19]:


print("Input shape: ", batch_input.shape)
print("Output shape: ", model(batch_input).shape)
print("")
sys.stdout.flush()


# In[ ]:


myloss = get_custom_loss(top_activation)

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


#Code to restart training if it is interrupted; the last trained model is reloaded.
checkpoints = glob.glob(checkpoint_dir + "/*.index")
if len(checkpoints) > 0 and reload is True:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    model.load_weights(latest_checkpoint[:-6]) #remove .index suffix
    print("Reloading from", latest_checkpoint)
    sys.stdout.flush()
    last_epoch=int(latest_checkpoint.partition('-')[0][-3:])
else:
    last_epoch=0


# The trained model is saved after every epoch that produces a validation loss lower than that of any previous epoch. Models were trained until the best validation loss was not improved after 20 epochs (patience=20), or a maximum of 200 epochs.

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


# At this point, the TIMIT test data is loaded and tested. Note that all input data fed to the model must first be normalized using the mean and standard deviation of the original training data, so those values need to be recorded for evaluation (see also below) and any future model use.

# In[ ]:


if testing is True:
    len3, test1 = getdata(datadir, allfileset[4620:4680], suffix, n_context_frames, verbose=1)
else:
    len3, test1 = getdata(datadir, allfileset[4620:6300], suffix, n_context_frames, verbose=0)
test2 = (test1 - trmean) / trstd
batched_test_dset = tf.keras.preprocessing.timeseries_dataset_from_array(
    test2[:len(test2)-n_context_frames,], test1[n_context_frames:,], sequence_length=window_length, batch_size=BATCH_SIZE)
test_eval = model.evaluate(batched_test_dset, verbose=0)
print("Test loss (TIMIT):", test_eval[0])


# ## Generate TIMIT data
# 
# For evaluation, the models were run on the TIMIT recordings whose formants were measured for the VTR-TIMIT formant database. The output files are generated in a subdirectory /timit/ of the output directory (*checkpoint_dir*). The filename will be prefixed by *checkpoint_dir* and have the suffix ".abs". These files are text files, with one output line per input spectral envelope in the input file. The columns of the output file consist of the following in order: the filename, 3 placeholder columns (for the particular needs of our research), the time point in milliseconds, 7 more placeholder columns, and then the total number of resonances (poles plus zeros). This is followed by the parameters (frequency, bandwidth, and amplitude correction factor, in that order) of the poles, in order of increasing mean frequency, and then those of the zeros, in order of increasing mean absolute frequency.
# 
# Other notes:
# * For output interpretation, it's important to remember that the generated "amplitudes" are not actually final formant amplitudes, but rather weighting factors that are used to adjust the initial formant amplitudes generated by formant().
# * The following code changes the frequencies of the zeros to negative values, to distinguish them from the poles. Also, since the zeros don't have their own amplitude correction factors, a placeholder value of "0.0" is inserted (theoretically we should have used 1.0 instead, but this value is not used in any computations).
# * The output code below assumes a frame rate of once every 5 milliseconds, which is the rate we used for our input data. (However, the VTR TIMIT measurements were taken once every 10 milliseconds, so every other output frame was used for evaluation.)
# * Since there is nothing in the custom loss code above that distinguishes one formant from another (aside from poles versus zeros), and any of them can take frequency values between 0 and 8000, the model output neurons may generate the formants in any random order (although that order will be constant from one frame to the next; e.g. if neuron 3 generates F1 for one frame, it does so for all frames and files).  The code below reorders the formants by their mean frequencies over all frames.
# * For the CNN models, each input frame must be converted into a window, and each window fed to the model one at a time in a loop. For the RNN models, the frame sequence can be given to the model all at once because it is designed to read sequences.

# In[ ]:


print("Generating", len(vtrfileset), ".abs files on TIMIT data:")

cmd = ["mkdir", "-p", checkpoint_dir + "/timit"]
subprocess.run(cmd)
datadir = superdir + "timit" + subdir
for filename in vtrfileset:
        outname=checkpoint_dir + "/timit/" + checkpoint_dir + "_" + filename

        #Note that we feed the files one at a time to getdata()
        lenf, f1 = getdata(datadir, [filename], suffix, n_context_frames, verbose=0)
        #Again, the input data must be normalized by the training set statistics
        f2 = (f1 - trmean) / trstd
        batched_sample_set = tf.keras.preprocessing.timeseries_dataset_from_array(
            f2[:len(f2)-n_context_frames], f1[n_context_frames:,], sequence_length=window_length, batch_size=BATCH_SIZE)
    
        i=0
        tloss=0
        for example_input, example_target in batched_sample_set:
            # Generate predictions for each frame
            y = model.predict(example_input)     
            #tloss = tloss + myloss(example_target, y) * example_input.shape[0]  #batch size
            # Rescale and reorganize output
            f, b, a = rescale_params(y, act=top_activation)
            if (i==0):
                zf=f.numpy()
                za=a.numpy()
                zb=b.numpy()
            else:
                zf=np.vstack((zf, f.numpy()))
                za=np.vstack((za, a.numpy()))
                zb=np.vstack((zb, b.numpy()))
            i=i + example_input.shape[0]  #Add size of batch (different for last batch)

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
        
        print("FILE:", filename)
        sys.stdout.flush()

        #Re-sort parameters in the order F1 B1 A1, F2 B2 A2, etc. and write to output file
        p = [(i, i+NSUM, i+(NSUM*2)) for i in ord]
        p2=sum(p, ())
        zp=np.hstack((zf, zb, za))
        out1=zp[:,p2]
        ff = open(outname + ".abs", "w")
        for i in range(out1.shape[0]):
            ff.write("{} AA 1 1 {:.1f} 200.0 60 0 0 60 40 2 {}   ".format(filename, i*5.0, NSUM))
            out1[i,:].tofile(ff, sep=" ")
            ff.write(" \n")
        ff.close()


# In[ ]:


print("")
print("FINISHED script for", checkpoint_dir)

