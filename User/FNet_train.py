#!/usr/bin/env python
# coding: utf-8

# # FormantNet Training Script (user-friendly version)
# 
# This script is a user-friendly version of the code used in the IS2021 and PaPE2021 papers. It can be used as an executable script to train a FormantNet model, as described below. The trained model can then be run on test wavefiles to generate formant tracks with the companion script **FNet_track.py**. (Alternatively, one may use **FormantNet.py** to do both training and tracking at the same time.)
# 
# ### Prerequisites:
# These scripts require Python 3 and NumPy, as well as Tensorflow (2.3 or later) and all of its dependencies. They also require the supplied supporting code files (FN_configuration.py, FN_data.py, and FN_model.py). If possible, set it up to run on a GPU (or TPU -- tensor processing unit), which will be much faster than on a conventional CPU. For example, our experiments with the TIMIT dataset (including both training and tracking) would take at least 12 hours on a CPU, and sometimes multiple days, but would only take about 4 hours on our GPU (most of which was actually for data loading and evaluation -- your GPU setup may be even faster).
# 
# ### Use:
# For use on the command line, the syntax for this script is summarized as follows, where square brackets indicate optional elements:
# 
# **python3 FNet_train.py** *[**-h**] [**-c** config] [**-v** validlist] modeldir trainlist*
# 
# In other words, there are 2 required arguments: the name of a directory in which to store the trained model (*modeldir*), and a text file (*trainlist*) listing the training files (with full pathnames). With the flags, the user may also optionally specify a configuration file and a validation file list. The -h ("help") option will print a summary of this information to the screen. 
# 
# Note that the script will also print a bunch of information (including the full configuration, data statistics, model architecture summary, training progress, and per-epoch loss values) to the screen, which can be redirected to a file in the usual ways, e.g.:
# 
# **python3 FNet_all.py -v timit_valid.txt my_FNet_model timit_train.txt > FNet_traininglog.txt**
# 
# See the README file in this directory for more information about the use of this script, the output file format, and the various options available via the configuration file.

# ## Set-up

# In[ ]:


import tensorflow as tf
import sys
import argparse

import FN_configuration
import FN_data
import FN_model


# Set JUPYTER_WINDOW to True if you want to run this script as a notebook in e.g. Jupyter. Note that input files will have to be specified manually below. Also note below that cfg.TESTRUN will be set to True.  

# In[ ]:


#JUPYTER_WINDOW = True
JUPYTER_WINDOW = False


# In[ ]:


if JUPYTER_WINDOW:
    configfile = 'FNet_config0.txt'
    modeldir = 'ft2_model'
    trainlistfile = 'timit_train.txt'
    validlistfile = 'timit_valid.txt'
    #validlistfile = None

else:
    parser = argparse.ArgumentParser(description="Train a FormantNet model.")
    parser.add_argument("modeldir", help="Directory to save model files in")
    parser.add_argument("trainlist", help="List of training files")
    parser.add_argument("-v", "--validlist", help="List of validation files")
    parser.add_argument("-c", "--config", help="Configuration file")
    
    args = parser.parse_args()    
    configfile = args.config
    modeldir = args.modeldir
    trainlistfile = args.trainlist
    validlistfile = args.validlist


# In[ ]:


cfg = FN_configuration.configuration()
cfg.configure(configfile)

if JUPYTER_WINDOW:
    cfg.TESTRUN = True

with open(trainlistfile) as f:
    trainlist = [i[:-1] for i in list(f)]

if validlistfile is not None:
    with open(validlistfile) as f:
        validlist = [i[:-1] for i in list(f)]
else:
    validlist = None

if cfg.TESTRUN:
    modeldir = "tmp_" + modeldir
    trainlist = trainlist[:50]
    if validlist is not None:
        validlist = validlist[:25]


# In[ ]:


print("\nFORMANTNET script FNet_train.py: Train a neural-network model for formant tracking.")
print("\nSUMMARY OF FILES AND DIRECTORIES:")
print("Test Run:", cfg.TESTRUN)
print("\nTraining file list: {} ({} files)".format(trainlistfile, len(trainlist)))
if validlist is None:
    print("Validation file list: None")
else:    
    print("Validation file list: {} ({} files)".format(validlistfile, len(validlist)))
print("Model directory:", modeldir)
print("Configuration file:", configfile)
cfg.report_status()


# ## Load data

# In[ ]:


print("\nLoading training data ....")
sys.stdout.flush()
batched_train_dset, trmean, trstd = FN_data.get_batched_data(trainlist, cfg)


# In[ ]:


if validlist is not None:  
    print("\nLoading validation data ....")
    sys.stdout.flush()
    batched_val_dset, mn, sd = FN_data.get_batched_data(validlist, cfg, trmean, trstd)
else:
    batched_val_dset = None


# ## Define and compile model

# In[ ]:


model = FN_model.define_model(cfg)


# ## Train model

# In[ ]:


FN_model.train_model(model, modeldir, cfg, batched_train_dset, batched_val_dset)


# The training set statistics are saved to a text file in the model directory so they can be used to normalize future test data.

# In[ ]:


with open(modeldir + "/Normfile", "w") as f:
    f.write(str(trmean) + '\n')
    f.write(str(trstd) + '\n')
    
print("\nSaving mean and standard deviation of training data to model directory (" + modeldir + "/Normfile" + "):")
print("Mean:", trmean)
print("Standard Deviation:", trstd)


# ## Restore best model and calculate overall loss on training and validation sets

# In[ ]:


latest = tf.train.latest_checkpoint(modeldir)
model.load_weights(latest)
print("\nRestoring model", latest)
sys.stdout.flush()

train_eval = model.evaluate(batched_train_dset, verbose=0)
print("Training loss:", train_eval[0])
sys.stdout.flush()

if validlist is not None:
    val_eval = model.evaluate(batched_val_dset, verbose=0)
    print("Validation loss:", val_eval[0])


# In[ ]:


print("\nFINISHED training script for", modeldir)

