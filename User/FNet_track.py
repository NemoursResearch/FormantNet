#!/usr/bin/env python
# coding: utf-8

# # FormantNet Tracking Script (user-friendly version)
# 
# This script is a user-friendly version of the code used in the IS2021 and PaPE2021 papers. It can be used as an executable script to run a FormantNet model (that has been trained by either **FormantNet.py** or **FNet_train.py**) on test wavefiles to generate predicted formant tracks.  (Alternatively, one may use **FormantNet.py** to do both training and tracking at the same time.)
# 
# ### Prerequisites:
# These scripts require Python 3 and NumPy, as well as Tensorflow (2.3 or later) and all of its dependencies. They also require the supplied supporting code files (FN_configuration.py, FN_data.py, and FN_model.py). If possible, set it up to run on a GPU (or TPU -- tensor processing unit), which will be much faster than on a conventional CPU. For example, our experiments with the TIMIT dataset (including both training and tracking) would take at least 12 hours on a CPU, and sometimes multiple days, but would only take about 4 hours on our GPU (most of which was actually for data loading and evaluation -- your GPU setup may be even faster).
# 
# ### Use:
# For use on the command line, the syntax for this script is summarized as follows, where square brackets indicate optional elements:
# 
# **python3 FNet_track.py** *[**-h**] [**-c** config] [**-o** outdir] modeldir testlist*
# 
# In other words, there are 2 required arguments: the name of the directory in which the trained model can be found (*modeldir*), and a text file (*testlist*) listing the files to be tracked (with full pathnames). With the flags, the user may also optionally specify a configuration file and an output directory in which to store the text files that will hold the predicted formant tracks. If no output directory is specified, then each output file is written to the same directory as the input wavefile. The -h ("help") option will print a summary of this information to the screen. 
# 
# Note that the script will also print a bunch of information (including the full configuration, data statistics, model architecture summary, and evaluation progress) to the screen, which can be redirected to a file in the usual ways, e.g.:
# 
# **python3 FNet_track.py -o my_FNet_results my_FNet_model timit_test.txt > FNet_trackinglog.txt**
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
    testlistfile = 'timit_test.txt'
    #testlistfile = 'VTRlist3.txt'
    outdir = 'ft2_out'
    #outdir = None

else:
    parser = argparse.ArgumentParser(description="Run a trained FormantNet model on a test set.")
    parser.add_argument("modeldir", help="Directory holding saved model")
    parser.add_argument("testlist", help="List of test files")
    parser.add_argument("-o", "--outdir", help="Directory to save test output files in [Default: input dir(s)]")
    parser.add_argument("-c", "--config", help="Configuration file")
    
    args = parser.parse_args()    
    configfile = args.config
    modeldir = args.modeldir
    outdir = args.outdir
    testlistfile = args.testlist


# In[ ]:


cfg = FN_configuration.configuration()
cfg.configure(configfile)

if JUPYTER_WINDOW:
    cfg.TESTRUN = True

with open(testlistfile) as f:
    testlist = [i[:-1] for i in list(f)]

if cfg.TESTRUN:
    modeldir = "tmp_" + modeldir
    testlist = testlist[:10]
    if outdir is not None:
        outdir = "tmp_" + outdir


# In[ ]:


print("\nFORMANTNET script FNet_track.py: Test a formant-tracking neural-network model on new wavefiles.")
print("\nSUMMARY OF FILES AND DIRECTORIES:")
print("Test Run:", cfg.TESTRUN)
print("\nTest file list: {} ({} files)".format(testlistfile, len(testlist)))
print("Model directory:", modeldir)
if outdir is None:
    print("Output directory: None (output tracking files redirected to wavefile directory/ies)")
else:
    print("Output directory:", outdir)
print("Configuration file:", configfile)
cfg.report_status()


# ## Define and compile model

# In[ ]:


model = FN_model.define_model(cfg)


# ## Load saved model weights and calculate overall loss on test set

# In[ ]:


latest = tf.train.latest_checkpoint(modeldir)
model.load_weights(latest)
print("\nRestoring model", latest)


# Prior to testing, we need to normalize the test data with the mean and standard deviation of the training set. These stats are saved in a file called "Normfile" in the model directory.

# In[ ]:


with open(modeldir + "/Normfile", "r") as f:
    trmean = float(f.readline())
    trstd = float(f.readline())

print("\nReloading mean and standard deviation of training data from model directory (" + modeldir + "/Normfile" + "):")
print("Mean:", trmean)
print("Standard Deviation:", trstd)


# At this point, the test data is loaded into a tf.data.Dataset, the data statistics are computed, and the model loss is calculated with respect to the test data.  Note that the test data is loaded again, as individual files, during tracking below, so the test data is effectively loaded twice. Thus, if you have a large dataset and slow machine, and are not interested in the statistics or loss value, it may make sense to skip this step by setting cfg.GET_TEST_LOSS to false.

# In[ ]:


if cfg.GET_TEST_LOSS:
    print("\nLoading test data ....")
    sys.stdout.flush()
    batched_test_dset, mn, sd = FN_data.get_batched_data(testlist, cfg, trmean, trstd)

    test_eval = model.evaluate(batched_test_dset, verbose=0)
    print("\nTest loss:", test_eval[0])
    sys.stdout.flush()


# ## Generate formant tracks on test data

# In[ ]:


FN_model.track_files(testlist, model, trmean, trstd, cfg, outdir)


# In[ ]:


print("\nFINISHED tracking script for files in", testlistfile)

