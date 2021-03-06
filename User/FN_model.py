#!/usr/bin/env python
# coding: utf-8

# # FormantNet Model Functions

# Functions used for defining, training, or using FormantNet models.

# In[ ]:


import tensorflow as tf
import numpy as np
import sys
import os
import glob

from FN_data import getdata


# ### Rescaling Model Output
# The **rescale_params()** function takes the output of the model, which is typically on a scale of 0 to 1 or -1 to 1, and rescales it to the expected scale for formant parameters (e.g. 0-8000 Hz for frequencies). The minimum and maximum values allowed for each formant parameter type (frequency, bandwidth, and amplitude) can be adjusted by the user using the configuration file. The input scale depends on the output activation function of the model (TOP_ACTIVATION); we experimented with linear, tanh, softsign, and ReLU, but found that sigmoid usually works best.  
#   
# Notes: 
# * As defined by the use of the tf.split below, this function forces the output of the model to be in the order **F1 F2 F3 ... B1 B2 B3 ... A1 A2 A3 ....** (which may be subsequently rearranged by **track_files()**). 
# * If ReLU is used as the activation function, the formant maximum values defined by cfg.MAXFREQ, cfg.MAXBW, and cfg.MAXAMP will be ignored. If linear is used, both minima and maxima are ignored.
# * This function is needed for evaluation and any future use of the model (see again  **track_files()** below).

# In[ ]:


def get_rescale_fn(cfg):
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, cfg.NPARAMS], dtype=tf.float32),))
    def rescale_params(params):
        freqs, bws, amps = tf.split(params, [cfg.NSUM, cfg.NSUM, cfg.NFORMANTS], axis=-1)
        if cfg.TOP_ACTIVATION == 'sigmoid':    #network produces values between 0 and 1
            freqs = cfg.MINFREQ + (freqs * (cfg.MAXFREQ - cfg.MINFREQ))
            bws = cfg.MINBW + (bws * (cfg.MAXBW - cfg.MINBW))
            amps = cfg.MINAMP + (amps * (cfg.MAXAMP - cfg.MINAMP))
        elif cfg.TOP_ACTIVATION == 'softsign' or cfg.TOP_ACTIVATION == 'tanh':  #network produces values between -1 and 1
            freqs = cfg.MINFREQ + ((freqs + 1.0) * ((cfg.MAXFREQ - cfg.MINFREQ) / 2.0))
            bws = cfg.MINBW + ((bws + 1.0) * ((cfg.MAXBW - cfg.MINBW) / 2.0))
            amps = cfg.MINAMP + ((amps + 1.0) * ((cfg.MAXAMP - cfg.MINAMP) / 2.0))
        elif cfg.TOP_ACTIVATION == 'relu':   #network produces values of 0 or greater
            freqs = freqs + cfg.MINFREQ
            bws = bws + cfg.MINBW
            amps = amps + cfg.MINAMP
        return freqs, bws, amps
    
    return rescale_params


# ### Loss Function
# 
# The functions used to compute the loss are defined here. We tried to write the code so that it could handle variations in sampling rate (SAMPLERATE), frequency range (from 0 to MAX_ANALYSIS_FREQ), number of formants (NFORMANTS), number of anti-formants (NZEROS), spectral resolution (SPECTRUM_NPOINTS), and the activation type of the final model output layer (TOP_ACTIVATION). For the TIMIT experiments, these were all set constant across all experiments: 16K sampling rate, 0-8K frequency range, 6 formants, 1 zero, 257-point spectra, sigmoid activation.

# The **formant()** function takes the frequency **F** and bandwidth **B** of each formant predicted by the model, and generates a corresponding formant: an array of spectrum levels **h** at each frequency bin **f** in the spectrum range at the given resolution (see Eqn. (1) of the IS2021 paper). The **vtfn()** function weights these by their corresponding amplitude factors, and combines them (multiplying or dividing, corresponding to whether it's a pole or zero) to produce a linear-scale spectral envelope.

# In[ ]:


def get_vtfn_func(cfg):
    spec1 = tf.cast(np.linspace(0.0, cfg.MAX_ANALYSIS_FREQ, cfg.SPECTRUM_NPOINTS), dtype=tf.float32)

    @tf.function
    def formant(freq, bw, nres):
        fex = tf.expand_dims(freq, axis=-1)
        bex = tf.expand_dims(bw, axis=-1)
        bsq = bex**2 * 0.25
        anum = fex**2 + bsq    
        #spec1 = tf.cast(np.linspace(0.0, cfg.MAX_ANALYSIS_FREQ, cfg.SPECTRUM_NPOINTS), dtype=tf.float32)
        spec2 = tf.tile(spec1, [tf.size(freq)])
        spec = tf.reshape(spec2, [-1, nres, cfg.SPECTRUM_NPOINTS])
        negspec = (spec - fex)**2 + bsq
        posspec = (spec + fex)**2 + bsq
        formants = anum / tf.math.sqrt(negspec * posspec)
        return(formants)

    #Note that vtfn returns a LINEAR-scale spectrum
    if cfg.NZEROS == 0:
        @tf.function
        def vtfn(freqs, bws, amps):
            ax = tf.expand_dims(amps, axis=-1)
            ax = 10.0 ** (ax / 20.0)   #convert log amplitudes to linear
            specs = formant(freqs, bws, cfg.NFORMANTS)
            sumspec = tf.reduce_sum(ax * specs, axis = -2)
            return sumspec
    else:
        @tf.function
        def vtfn(freqs, bws, amps, zfreqs, zbws):
            ax = tf.expand_dims(amps, axis=-1)
            ax = 10.0 ** (ax / 20.0)   #convert log amplitudes to linear
            fspecs = ax * formant(freqs, bws, cfg.NFORMANTS)
            sumspecs = tf.reduce_sum(fspecs, axis = -2, keepdims=True)
            zspecs = 1.0 / formant(zfreqs, zbws, cfg.NZEROS)
            allspecs = tf.concat([sumspecs, zspecs], axis = -2)
            prodspecs = tf.reduce_prod(allspecs, axis = -2)
            return prodspecs
        
    return vtfn


# Finally, the model loss is calculated with **custom_loss()**. First, the batch and sequence dimensions are collapsed. Then the input model parameters are rescaled with **rescale_params()**. The formants are split into poles and zeros, and sent to **vtfn()** to compute a linear-scale spectral envelope. The envelope is then converted to decibel scale, and the spectral loss is calculated as the mean square difference between the generated envelope and the original envelope.

# If Delta-Frequency Loss (added loss equal to variation in predicted formant frequency over time) is being used, it is also calculated here, weighted by the **DIFFWEIGHT** parameter, and added to the spectral loss to get the final loss. (We also experimented with using delta-loss with bandwidths and amplitudes, but those experiments have been unsuccessful so far.)

# In[ ]:


def get_custom_loss(cfg, rescale_params, vtfn):
    if cfg.NZEROS == 0:
        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, cfg.SPECTRUM_NPOINTS], dtype=tf.float32),
                                     tf.TensorSpec(shape=[None, None, cfg.NPARAMS], dtype=tf.float32)))
        def custom_loss(specs_input, params_pred):
            specs_input = tf.reshape(specs_input, [-1, cfg.SPECTRUM_NPOINTS])
            params_pred = tf.reshape(params_pred, [-1, cfg.NPARAMS])
            freqs, bws, amps = rescale_params(params_pred)
            specs_pred = vtfn(freqs, bws, amps)
            specs_pred = 20.0 * tf.math.log(cfg.FLOOR + specs_pred) / tf.math.log(10.0)
            mse = tf.math.reduce_mean(tf.square(specs_input - specs_pred)) #loss over whole batch
            fdiff = tf.math.reduce_mean(tf.abs(freqs[1:] - freqs[:-1]))
            #bdiff = tf.math.reduce_mean(tf.abs(bws[1:] - bws[:-1]))
            #adiff = tf.math.reduce_mean(tf.abs(amps[1:] - amps[:-1]))
            #return (mse + cfg.DIFFWEIGHT * (fdiff + bdiff + 25.0 * adiff))
            return (mse + cfg.DIFFWEIGHT * fdiff)

    else:  
        @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, cfg.SPECTRUM_NPOINTS], dtype=tf.float32),
                                     tf.TensorSpec(shape=[None, None, cfg.NPARAMS], dtype=tf.float32)))
        def custom_loss(specs_input, params_pred):
            specs_input = tf.reshape(specs_input, [-1, cfg.SPECTRUM_NPOINTS])
            params_pred = tf.reshape(params_pred, [-1, cfg.NPARAMS])
            freqs, bws, amps = rescale_params(params_pred)
            pfreqs, zfreqs = tf.split(freqs, [cfg.NFORMANTS, cfg.NZEROS], axis=-1)
            pbws, zbws = tf.split(bws, [cfg.NFORMANTS, cfg.NZEROS], axis=-1)
            specs_pred = vtfn(pfreqs, pbws, amps, zfreqs, zbws)
            specs_pred = 20.0 * tf.math.log(cfg.FLOOR + specs_pred) / tf.math.log(10.0)
            mse = tf.math.reduce_mean(tf.square(specs_input - specs_pred)) #loss over whole batch
            fdiff = tf.math.reduce_mean(tf.abs(freqs[1:] - freqs[:-1]))
            #bdiff = tf.math.reduce_mean(tf.abs(bws[1:] - bws[:-1]))
            #adiff = tf.math.reduce_mean(tf.abs(amps[1:] - amps[:-1]))
            #return (mse + cfg.DIFFWEIGHT * (fdiff + bdiff + 25.0 * adiff))
            return (mse + cfg.DIFFWEIGHT * fdiff)
    
    return custom_loss


# ### Model definition
# 
# **define_model()** defines the model's architecture, loss function, and optimizer before compiling it, then prints summary statistics. In our TIMIT experiments, we produced the best results with just a single LSTM layer of 512 units followed by a single Dense layer, but the user may experiment with adding more layers of either type, and changing their sizes and activation functions, via the provided configuration file. The LEARNING_RATE used by the Adam optimizer can also be adjusted (default 0.0001).

# In[ ]:


def define_model(cfg):

    #Definition of model architecture:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(None, cfg.SPECTRUM_NPOINTS)))
    for i in range(cfg.LSTM_LAYERS):
        model.add(tf.keras.layers.LSTM(cfg.LSTM_UNITS, return_sequences=True))
    for i in range(cfg.DENSE_LAYERS - 1):
        model.add(tf.keras.layers.Dense(cfg.DENSE_UNITS, activation=cfg.DENSE_ACTIVATION))
    model.add(tf.keras.layers.Dense(cfg.NPARAMS, activation=cfg.TOP_ACTIVATION))

    #Here the model's loss function is defined before the model is compiled.
    rescale_params = get_rescale_fn(cfg)
    myvtfn = get_vtfn_func(cfg)
    myloss = get_custom_loss(cfg, rescale_params, myvtfn)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LEARNING_RATE),
        loss=myloss, metrics=[myloss]
    )

    #Summary statistics:
    print("\nModel input shape: ({}, {}, {})".format(cfg.BATCH_SIZE, cfg.SEQUENCE_LENGTH, cfg.SPECTRUM_NPOINTS))
    print("Model output shape: ({}, {}, {})".format(cfg.BATCH_SIZE, cfg.SEQUENCE_LENGTH, cfg.NPARAMS))
    model.summary()
    sys.stdout.flush()
    
    return model


# ### Model training

# **train_model()** does the model training. During training, the model weights are saved after every epoch that produces a validation loss lower than that of any previous epoch. Models are trained until the best validation loss has not improved after cfg.PATIENCE epochs, or a maximum of cfg.EPOCHS epochs. If there is no validation set, the training set loss is used for monitoring instead.

# If ALLOW_RETRAIN is True and there is at least one pre-existing model in the model directory, the newest saved model will be reloaded, and training will pick up where it left off with that model.

# The non-best models are deleted, unless DELETE_OLDER_MODELS is set to False.

# In[ ]:


def train_model(model, modeldir, cfg, train_dset, val_dset=None):

    checkpoints = glob.glob(modeldir + "/*.index")
    if len(checkpoints) > 0 and cfg.ALLOW_RETRAIN and not cfg.TESTRUN:
        latest = tf.train.latest_checkpoint(modeldir)
        model.load_weights(latest)
        print("\nReloading pre-existing model", latest, "for further training.")
        last_epoch = int(latest[-3:])
    else:
        last_epoch = 0

    if val_dset is None:
        mymonitor = 'custom_loss'
    else:
        mymonitor = 'val_custom_loss'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=modeldir + "/model.epoch{epoch:03d}",
        save_best_only=True, save_weights_only=True,
        monitor=mymonitor, mode='min')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=cfg.PATIENCE, monitor=mymonitor, mode='min')

    if cfg.TESTRUN:
        Verbosity = 1
    else:
        Verbosity = 2

    print("\nBegin training:")
    sys.stdout.flush()
    model.fit(train_dset, epochs=cfg.EPOCHS, initial_epoch=last_epoch, 
              validation_data=val_dset, verbose=Verbosity,
              callbacks=[model_checkpoint_callback, early_stopping_callback])
    sys.stdout.flush()

    if cfg.DELETE_OLDER_MODELS:
        print("\nDeleting non-best models....")
        latest = tf.train.latest_checkpoint(modeldir)
        for i in glob.glob(modeldir + "/model.epoch*"):
            if not i.startswith(latest):
                #print("Deleting", i)
                os.remove(i)


# ### Generating formant tracks on test files
# 
# The **track_files()** function runs a trained model on a list of wavefiles, generating formant measurements every cfg.FRAME_STRIDE_MSEC milliseconds (5 msec by default). These measurements are saved in text files, with one output line for every frame. The columns of the output file consist of the following in order: the filename, the time point in milliseconds, and then the total number of resonances (poles plus zeros). This is followed by the parameters (frequency, bandwidth, and amplitudes, in that order) of the poles, in order of increasing mean frequency, and then those of the zeros, in order of increasing mean absolute frequency. By default, the frequencies are listed first, then bandwidths, then amplitudes (i.e. F1 F2 F3 ... B1 B2 B3 ... etc.), but this can be switched to an interleaved ordering, i.e. F1 B1 A1, F2 B2 A2, etc., by setting the configuration parameter FREQUENCIES_FIRST to False.  
#   
# The names and locations of these output text files depend on whether an output directory name is provided. If so, Then all of the output text files are stored in this directory. To prevent name collisions, the output filename is derived from by converting all slashes in the input pathname to underscores -- but to shorten the names, any initial part of the input path that is common to all of the files is left out. As an illustration, the following example shows the corresponding output names for a list of 3 input files:
# 
# timit/test/dr1/faks0/sa1.wav -> outdir/dr1_faks0_sa1.txt  
# timit/test/dr1/faks0/sa2.wav -> outdir/dr1_faks0_sa2.txt  
# timit/test/dr2/fcmr0/sa1.wav -> outdir/dr2_fcmr0_sa1.txt  
# 
# On the other hand, if an output directory name is **not** provided, then each output file is written to the same directory as the corresponding input file, with the same name as the wavefile but a different extension (.txt).  
# 
# In either case, the file extension can be changed via the configuration file (OUT_EXT).
# 
# Other notes:
# * Since there is nothing in the custom loss code above that distinguishes one formant from another (aside from poles versus zeros), and any of them can take frequency values between MINFREQ and MAXFREQ, the model output neurons may generate the formants in any random order (although that order will be constant from one frame to the next; e.g. if neuron 3 generates F1 for one frame, it does so for all frames and files). track_files() reorders the formants by their mean frequencies over all frames.
# * track_files() changes the frequencies of the antiformants (zeros) to negative values, to distinguish them from the poles. Also, since the antiformants don't have their own amplitude correction factors, a placeholder value of "1.0" is inserted.
# * For output interpretation, it's important to remember that the initial "amplitudes" generated by the model are not actually final formant amplitudes, but rather weighting factors that are used to adjust the initial formant amplitudes generated by formant(). By default, track_files() attempts to convert these weighting factors to "real" formant amplitude estimates (but see README for more information).  This behavior can be changed to generate the original weighting factors by changing the configuration parameter REAL_AMPLITUDES to False.
# * Unlike the python scripts in IS2021/ and PaPE2021/, track_files() also performs the final binomial smoothing of the output parameters; the number of smoothing passes is controlled by BIN_SMOOTH_PASSES (default 10).

# In[ ]:


def track_files(testlist, model, trmean, trstd, cfg, outdir=None):
    
    import time
    start_time = time.perf_counter()

    rescale_params = get_rescale_fn(cfg)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

        #Determine how much of the directory path is common to all test files
        y = [i.split("/") for i in testlist]
        n_common_prefixes = 0
        b = 0
        for i in y[0]:
            for j in y[1:]:
                if (j[n_common_prefixes] != i):
                    b = 1
                    break
            if b == 1:
                break
            else:
                n_common_prefixes += 1

    print("\nGenerating", len(testlist), "output files:")
    for filename in testlist:

        #Determine output directories and filenames
        if filename.endswith(".wav"):
            shortname = filename.rsplit(".", 1)[0]
        else:
            shortname = filename
        if outdir is None:
            outname = shortname + '.' + cfg.OUT_EXT
        else:
            shortname = "_".join(shortname.split("/")[n_common_prefixes:])
            outname = outdir + "/" + shortname + '.' + cfg.OUT_EXT
        print(filename, " -> ", outname)
        sys.stdout.flush()

        #Note that we feed the files one at a time to getdata()
        lenf, f1 = getdata([filename], cfg, verbose=0)
        #Again, the input data must be normalized by the training set statistics
        f2 = (f1 - trmean) / trstd
        #Add a third (batch) dimension so that frame sequence can be read directly by model
        f2 = tf.expand_dims(f2, axis=0)
        #Generate formant predictions
        y = model.predict(f2)[0]
        # Rescale and reorganize output
        f, b, a = rescale_params(y)
        zf = f.numpy()
        za = a.numpy()
        zb = b.numpy()        

        # Convert antiformant frequencies to negative numbers, and 
        # insert placeholder "1.0" for antiformant amplitudes.
        # Then sort formants and antiformants separately by increasing frequency.
        if cfg.NZEROS > 0:
            fp, f0 = tf.split(zf, [cfg.NFORMANTS, cfg.NZEROS], axis=-1)
            f0 = f0 * -1.0
            zf = tf.concat([fp, f0], axis=-1)
            a0 = tf.ones([za.shape[0], cfg.NZEROS], dtype=tf.float32)
            za = tf.concat([za, a0], axis=-1)    
            ord1 = np.hstack((np.argsort(np.mean(fp, axis=0)), 
                              (np.flip(np.argsort(np.mean(f0, axis=0)) + cfg.NFORMANTS))))
        else:
            ord1 = np.argsort(np.mean(np.abs(zf),axis=0))

        # Convert amplitude adjustment factors to true estimates of amplitudes
        if cfg.REAL_AMPLITUDES:
            za2 = 10.0 ** (za / 20.0)
            zf2 = zf ** 2.0
            zb2 = (zb ** 2.0) / 4.0
            num = za2 * (zf2 + zb2)
            den = ((4.0 * zb2 * zf2) + (zb2 ** 2.0)) ** 0.5
            rat = num / den
            za = 20.0 * np.log10(cfg.FLOOR + rat)

        # Sort parameters in the order F1 F2 F3 ... B1 B2 B3 ... A1 A2 A3 ...
        if cfg.FREQUENCIES_FIRST:
            ord2 = [i for i in ord1] + [i+cfg.NSUM for i in ord1] + [i+cfg.NSUM*2 for i in ord1]
        # Otherwise output in order F1 B1 A1 F2 B2 A2 F3 B3 A3 ... 
        else:
            p = [(i, i+cfg.NSUM, i+(cfg.NSUM*2)) for i in ord1]
            ord2 = sum(p, ())

        # Do binomial smoothing of output, if any
        zp = np.hstack((zf, zb, za))
        for i in range(cfg.BIN_SMOOTH_PASSES):
            zp = np.vstack((0.75*zp[0] + 0.25*zp[1],
                            0.5*zp[1:-1,] + 0.25*zp[2:,] + 0.25*zp[:-2,],
                            0.75*zp[-1] + 0.25*zp[-2]))

        # Write to output file
        out1 = zp[:,ord2]
        ff = open(outname, "w")
        for i in range(out1.shape[0]):
            ff.write("{} {:.1f} {}   ".format(shortname, i * cfg.FRAME_STRIDE_MSEC, cfg.NSUM))
            out1[i,:].tofile(ff, sep=" ", format="%.2f")
            ff.write(" \n")
        ff.close()
        
    t = time.perf_counter() - start_time
    print("Total tracking time: {:.2f} min ({:.2f} sec/file)".format(t/60.0, t/len(testlist)))


# ## Code bebugging

# In[ ]:


if __name__ == "__main__":
    import FN_configuration
    cfg = FN_configuration.configuration()
    cfg.configure(None)
    model = define_model(cfg)

