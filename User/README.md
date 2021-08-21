# FormantNet Training and Tracking Scripts: user-friendly versions

This directory holds user-friendly versions of the code used in the IS2021 and PaPE2021 papers. Three Python scripts that can be executed on the command-line are provided. **FormantNet.py** can be used to both train a FormantNet model and run the model to generate formant tracks, as described below. Alternatively, one may separately use **FNet_train.py** to train a model, and **FNet_track.py** to run a trained model on new wavefiles. These are described in more detail below.


## Prerequisites:
These scripts require Python 3 and NumPy, as well as Tensorflow (2.3 or later) and all of its dependencies. They also require the supplied supporting code files: **FN_configuration.py**, **FN_data.py**, and **FN_model.py**. If possible, set up the scripts to run on a GPU (or TPU -- tensor processing unit), which will be much faster than on a conventional CPU. For example, our experiments with the TIMIT dataset (including both training and tracking) would take at least 12 hours on a CPU, and sometimes multiple days, but would only take about 4 hours on our GPU (most of which was actually for data loading and evaluation -- your GPU setup may be even faster).

The files in the **notebooks/** directory are simply Notebook versions of the same Python files, and are not needed to execute the Python scripts.


## Quick-Start:
**FNet_train.py** trains a FormantNet model on a corpus of wavefiles. **FNet_track.py** takes a trained FormantNet model and runs it on a list of wavefiles to generate formant tracks. Alternatively, **FormantNet.py** does both training and tracking in the same script (it can't be used to do just one). Note that **FNet_track.py** can use models trained by either **FormantNet.py** or **FNet_train.py**.

Each script has a set of required files as well as optional files that can be specified on the command line with flags, as well as further options that can be specified in a configuration file. The command-line syntax of any of these scripts can be displayed by using the -h ("help") flag, e.g.: 
**> python3 FormantNet.py -h**

Their syntax is summarized as follows:

**python3 FNet_train.py** _[**-c** config] [**-v** validlist] modeldir trainlist_
**python3 FNet_track.py** _[**-c** config] [**-o** outdir] modeldir testlist_

**python3 FormantNet.py** _[**-c** config] [**-v** validlist] [**-t** testlist] [**-o** outdir] modeldir trainlist_

In other words, each script requires the name of a model directory (the location of either the model to be trained, or the already-trained model) and a list of either the training or test wavefiles. Flags can be used to specify further options. For the training scripts, one can provide a validation filelist with the **-v** option (see below). With FormantNet.py, the test filelist is optional and can be specified by the **-t** option; if it is not provided, then the trained model is run on the training and validation files. The filelists should provide the complete path to the input wavefiles; some example filelists are provided in the **filelists/** directory. Like the **PaPE2021** scripts (but unlike the **IS2021** scripts), these scripts take raw wavefiles as input.

For the tracking scripts, one can use the **-o** option to specify an output directory for the tracking output files (which can be the same as the model directory if desired); if an output directory is not provided, the output files are written to the input wavefile directories (see below).

Finally, the **-c** option can be used to provide a configuration file, which can be used to specify more options. An example file (**FNet_config0.txt**) is provided. Without a configuration file, the scripts use the default option values, which are listed in FNet\_config0.txt. The default options generate frequencies, bandwidths, and amplitudes for 6 formants and 1 antiformant between 0 and 8000 Hz. These are generally speaking the options used to get the best results on the TIMIT dataset as described in the **IS2021** and **PaPE2021** directories, and should be adequate for most purposes, but some options should be considered carefully; see below for details.

Note that the scripts will also print a bunch of information (including the full configuration, data statistics, model architecture summary, training progress, per-epoch loss values, and/or evaluation progress) to the terminal, which can be redirected to a file in the usual ways, e.g.:

**python3 FormantNet.py -t timit_test.txt my_FNet_model timit_train.txt > FNet_log.txt**


## Output:

### Model:
A trained model will be stored in the specified model directory as 4 files: one named **checkpoint**; one named **Normfile** (which stores the mean and standard deviation of the training set, and is needed to normalize future test data); And two called **model.epoch**_NNN_ **.index** and **model.epoch**_NNN_**.data-0000-of-0001**, where _NNN_ is a numeral indicating the final epoch at which the model was saved during training (see below). If the configuration variable SAVE\_OLDER_MODELS is set to False, earlier versions of the model may be saved in the model directory as well. (These won't interfere with future model use; the scripts always use the latest model version.)

### Formant Track files:

**File Names and Locations:**
The formant measurements produced by FormantNet.py and FNet\_track.py are saved in plain-text files, one per input wavefile. The names and locations of these files depend on whether an output directory name is provided. If so, then all of the output files are stored in this directory. To prevent name collisions, the output filename is derived from by converting all slashes in the input pathname to underscores -- but to shorten the names, any initial part of the input path that is common to all of the files is left out. As an illustration, the following example shows the corresponding output names for a list of 3 input files:

timit/test/dr1/faks0/sa1.wav -> outdir/dr1\_faks0_sa1.txt  
timit/test/dr1/faks0/sa2.wav -> outdir/dr1\_faks0_sa2.txt  
timit/test/dr2/fcmr0/sa1.wav -> outdir/dr2\_fcmr0_sa1.txt  

On the other hand, if an output directory name is **not** provided, then each output file is written to the same directory as the corresponding input file, with the same name as the wavefile but a different extension (.txt).

In either case, the file extension ("txt" by default) can be changed with the configuration parameter OUT_EXT.

**File Format:**
The output files will consist of one output line for every frame (analysis window). The frame rate is 5 msec by default, i.e. formant measurements are produced every 5 msec (the first at 0 msec, the second at 5 msec, etc.); this can be altered by the configuration parameter FRAME\_STRIDE_MSEC.

The columns of the output file consist of the following in order: the filename, the time point in milliseconds, and then the total number of predicted resonances (formants plus antiformants). This is followed by the parameters (frequency, bandwidth, and amplitudes, in that order) of the formants, in order of increasing mean frequency, and then those of the antiformants, in order of increasing mean absolute frequency. By default, all frequencies are listed first, then all bandwidths, then all amplitudes (i.e. F1 F2 F3 ... B1 B2 B3 ... etc.), but this can be switched to an interleaved ordering, i.e. F1 B1 A1, F2 B2 A2, etc., by setting the configuration parameter FREQUENCIES_FIRST to False.

Other notes:
* The frequencies of the antiformants (zeros) are written as negative values, to distinguish them from the poles. Also, note that the FormantNet models do not estimate antiformant amplitude weight factors, so a placeholder value of "1.0" is generated for the antiformant amplitude columns (which may be converted to "real" amplitudes as described below).
* For output interpretation, it's important to remember that the initial "amplitudes" generated by the model are not actually final formant amplitudes, but rather weighting factors that are used to adjust the initial formant amplitudes determined by the formant's frequency and bandwidth. By default, the scripts attempt to convert these weighting factors to a "real" formant amplitude estimate. Note that this should be considered a lower bound estimate of the amplitude at the formant frequency, since it won't include contributions to the energy by the neighboring formants. This behavior can be changed to generate the original weighting factors by changing the configuration parameter REAL_AMPLITUDES to False.
* Unlike the python scripts in IS2021/ and PaPE2021/, these scripts also perform the final binomial smoothing of the output parameters; the number of smoothing passes is controlled by the configuration parameter BIN\_SMOOTH_PASSES (default 10).


## Use of Validation Files:

The conventional wisdom regarding the training of machine learning models (such as neural networks) is to withhold a portion of the available data from the training set, and use it to monitor the progress of the training. The reason is that hyperparameters such as the optimum number of _epochs_ (training iterations) is not known in advance; over-training can produce a model that performs very well on the training data but does not generalize to other data. So we try to determine when to stop training by evaluating the model against an independent validation set after every epoch.

In our scripts, the model's "loss" (error) is measured against the validation set after every epoch. Each time the loss value reaches a new minimum, the current version of the model is saved in the model directory. Training will halt if a new minimum is not reached after 20 epochs by default; this value is called the PATIENCE and can be changed in the configuration file. There is also a configuration variable for the maximum number of epochs (EPOCHS, default 200), after which training will be stopped regardless.

The validation set should be perhaps about 10% of the available training data, and separated from the training data. Its distribution should be representative of the corpus as a whole in terms of any possibly relevant variables, e.g. speaker gender, age, and dialect. The provided filelists **timit_train.txt** and **timit_valid.txt** are the datasets used in our TIMIT experiments and give an example of how to divide a dataset into training and validation sets.

Per conventional wisdom, we recommend the use of a validation set. However, in practice, the LSTM models we trained on the TIMIT dataset never reached a minimum validation loss before the maximum EPOCHS value (200) was reached; and when we raised the maximum to reach the validation minimum (about 350 epochs), the resulting models did not perform as well as the models stopped after 200 epochs. (This may be because the loss value, computed as a difference between input and predicted spectral envelopes, may not perfectly correlate with the final evaluation metric, which was predicted versus hand-measured formant frequencies.) So arguably, in our case the validation set was not needed. In addition, a user with a small dataset may want to use all available data for training. Hence, we provide the option not to use a validation set. In this case, the loss measured after every epoch will be the training loss rather than a validation loss. In practice, this loss is unlikely to reach a minimum, and training will stop after EPOCHS epochs.


## Configuration Files:

The optional configuration file allows the user to further control aspects of model architecture, training, and use. The provided file **FNet_config0.txt** is an example configuration file that also lists the default values for every configuration variable. The format of the file is simple; it starts with the required line "**[DEFAULT]**", and is followed by lines of the format "_VARIABLE_ = _VALUE_". These can all be preceded or followed by optional comment lines that are marked by a line-initial **#** sign. Any of the listed variables may be omitted from the file; the scripts will fill in the default value. Any listed values that are not used by a particular script will be ignored.

You can use the same configuration file for all of the scripts, and in fact this is recommended, for most of the configuration variables must be identical between the training and tracking steps. Hence it's a good idea to save the configuration file used in training a particular model, if you want to use that model later for tracking.

As mentioned above, the provided default values were used in our experiments that provided the best results on the TIMIT dataset (see the **IS2021** and **PaPE2021** directories), and we expect most of them to perform well under most conditions. But there are certain variables that the user should consider carefully, which we'll discuss first below.  The rest are provided for users who are more experienced in working with neural network models and wish to experiment further.


### Important parameters:

**SAMPLERATE:** The sampling rate of the input wavefiles is assumed to be 16 KHz (16000.0 Hz) by default; this value must be changed to the sampling rate of your wavefiles (which must all have the same sample rate).

**MAX_ANALYSIS_FREQ:** This value determines the upper limit of the frequency range to be modeled -- the range in which the requested formants and antiformants will be fitted (the lower limit is fixed at 0 Hz). This value is set to half the SAMPLERATE by default, and must be equal to or less than this amount. If your SAMPLERATE is very high, you may want to reduce MAX\_ANALYSIS_FREQ to about 8000 Hz. If you're only interested in the first few formants, you may also try setting this value to a lower value, e.g. 4000 or 5000 Hz, and setting NFORMANTS to 3, for example. However, in our TIMIT experiments, we found better fits for the first 3 formants when we modeled the entire 0-8000 Hz frequency range and requested 6 formants.

**NFORMANTS and NZEROS:** These control the number of formants and antiformants, respectively, to be modeled in the frequency range determined by MAX\_ANALYSIS_FREQ, and should be considered carefully. NFORMANTS should be the average number of formants you would expect to find in that frequency range, given your corpus population. For the TIMIT corpus, which consisted of a mixture of adult men and women, we found that setting NFORMANTS to 6 gave the best performance (at least for F1-F3 frequency estimates). You might choose a higher number if for instance your population is all men, or a lower number for all children. We also found better performance by including one antiformant.

Note that the total number of parameters estimated by the neural network (its "output size") will equal NFORMANTS * 3 + NZEROS * 2, since we estimate frequencies, bandwidths, and amplitude weight factors for the formants, but only frequencies and bandwidths for the antiformants; the latter's amplitude weight factors are fixed at 1.0.

**PREEMPH:** This controls the amount of pre-emphasis used when extracting the analysis windows. When pre-emphasis is used, a value between 0.90 and 1.0 is typical; the default is 0.98. However, the user may want to consider not using pre-emphasis at all (setting it to 0). In our recent TIMIT experiments, we produced the lowest error rates on all segments -- including consonants -- when using pre-emphasis, but got the lowest error rates on vowels -- including much lower F1 errors -- when **not** using pre-emphasis (see **PaPE2021/PaPE_presentation.pdf**, slides 19-20). So this is a parameter that may be worth experimenting with.

**DIFFWEIGHT:** A FormantNet model's loss is calculated primarily from the difference between the input spectral envelope and the model envelope calculated from the predicted formant features. However, in recent experiments, we experimented with adding an additional loss component calculated from the sum of the changes in predicted formant frequencies between each pair of adjacent frames; this has the effect of smoothing out the frequency tracks. The additional loss value is multiplied by DIFFWEIGHT before being added to the total loss. Values between 0.05 and 0.15 tended to work best, but this is another parameter whose best value varied between experiments; we found the best consonant results with 0.15 (the default value) but the best vowel results with 0.10 (see again **PaPE2021/PaPE_presentation.pdf**, slides 19-20). Again, the user may want to experiment with this.

**FRAME_STRIDE_MSEC:** As described above, this parameter determines the spacing between analysis windows (e.g. every 5.0 msec by default) and thus how many formant measurements are produced per file. But also note that this also controls how many analysis windows are extracted and used during training as well. So, for example, reducing this to 2.5 msec would in effect double the amount of training instances from the same training set (doubling the training time), while increasing it to 10.0 msec would halve the training material.

**WINDOW_LENGTH_MSEC:** This parameter (default 32.0 msec) determines the analysis window width (amount of signal used to predict formant values). Note that this value along with the MAX\_ANALYSIS\_FREQ determines the resulting size of the spectral envelope (number of frequency bins), and hence the model's "input size" (size = WINDOW\_LENGTH\_MSEC * self.MAX\_ANALYSIS_FREQ / 1000.0 + 1). Increasing it will increase the model's size, possibly increasing its performance but also substantially increasing training and testing times.

**TESTRUN:** Setting this to True causes the script to run in Test Mode: Only 50 training files and 25 validation files are loaded for training, only 3 epochs of training are run, and only 10 files are tested. The entire script should only take a few minutes. In addition, the prefix "tmp_" is prepended to the given model and output directories (to avoid writing the test output to the final directories). So doing an initial run with TESTRUN = True is a great way to quickly test your setup, making sure all the required software is installed, input and output files are readable and writeable, and so forth, before running the script for real.


### Other parameters:

**MINFRQ, MAXFREQ, MINBW, MAXBW, MINAMP, MAXAMP:** These parameters determine the minimum and maximum formant measurements that the model is allowed to predict (but see **TOP_ACTIVATION** below). The default values should be fine for most uses (as long as MAXFREQ is not too far from MAX\_ANALYSIS\_FREQ). MINBW must be greater than 0 to avoid division-by-zero errors. Note that MINAMP and MAXAMP (-100.0 and +100.0 by default) are actually limits not for the final formant amplitudes themselves, but for the _amplitude weighting factors_ (expressed in dB) predicted by the model, which are multiplied to the initial formants calculated from the frequencies and bandwidths (see Eqn. 1 in the **IS2021** manuscript).

**SMOOTH_LINEAR:** The initial spectral envelope calculated from the Fourier transform of the input signal is smoothed before being given to the model. We experimented with performing the smoothing step either before (SMOOTH_LINEAR = True) or after (False) the envelope is converted from the linear scale to the dB scale. We generally found better performance by smoothing on the linear scale.

**ENV_SMOOTH_PASSES:** The number of passes of smoothing performed in the aforementioned spectral envelope smoothing. The default is 6. This is not to be confused with BIN\_SMOOTH_PASSES (see below).

**FLOOR:** A value (default 0.001) added to the linear spectral envelope before being converted to the dB (log) scale. This prevents the resulting dB-scale envelopes from having large negative outliers, which allows us to keep MINAMP at a reasonable value.

**SEQUENCE_LENGTH:** The FormantNet models used in these scripts are RNN models, meaning they analyze the input as sequences of analysis windows, in which the model's analysis of input windows can be influenced by the preceding windows in the sequence. This value controls the length (number of analysis windows) of the sequences used in training, and ideally should be similar in length to the test wavefiles; but on the other hand, training is more efficient with shorter sequences. The default value of 64 seems like a good compromise. (This doesn't restrict the test sequences, which can be any length and are analyzed as a single sequence per file.)

**SEQUENCE_STRIDE:** This parameter determines how far apart the sequences are extracted from the training data. By default it is set equal to SEQUENCE_LENGTH, which means that each training sequence begins right where the last one ended. Using a smaller SEQUENCE\_STRIDE would produce overlapping training sequences, which theoretically would produce a larger, more varied training set (in terms of start and end points) from the same set of wavefiles. In our experiments, however, we did not find an improvement in performance from reducing this parameter.

**BATCH_SIZE:** DNN training sets are divided into small batches of training instance (sequences in this case). Conventional wisdom suggests this should be a small power of 2; the default is 8.

**LSTM_LAYERS and DENSE_LAYERS:** The DNN models allowed by these scripts consist of 1 or more LSTM layers followed by 1 or more Dense layers. The number of DENSE_LAYERS includes the final ("output") layer. We experimented with different numbers, but found no improvement in using more than one of each.

**LSTM_UNITS and DENSE_UNITS:** We also allow the user to experiment with the size (number of units) of the LSTM layer and the non-final Dense layers. These parameters (default 512) are restricted to be single integers (i.e. all LSTM layers must be the same size, and likewise all non-final Dense layers). Note that the final ("output") dense layer has a fixed size determined by the number of output parameters (NFORMANTS * 3 + NZEROS * 2).

**DENSE_ACTIVATION:** The activation function of the non-final Dense layers, if any. "relu", the default, is a common choice.

**TOP_ACTIVATION:** The activation function of the final ("output") Dense layer. This determines the initial scale of the model's output, e.g. the choice of sigmoid means that the model's initial output values are all between 0 and 1. These are later rescaled by a support function to the ranged determined by MINFREQ, MAXFREQ, etc., but note that the use of an activation function with no limit on one end (e.g. relu) or both ends (e.g. linear) will cause the limits defined by MINFREQ, MAXFREQ, etc. to be ignored on the unlimited end(s).

**LEARNING_RATE:** The learning rate parameter of the Adam model training optimizer function (default value 0.0001). This parameter is provided for experienced users of DNNs to play with.

**EPOCHS and PATIENCE:** As explained above in the Validation Data section, these parameters control, respectively, the absolute maximum number of training epochs (default 200), and the number of epochs the algorithm will wait for a new minimum loss value (default 20).

**ALLOW_RETRAIN:** If this parameter is set to True (the default), a training script will first check the model directory for any pre-existing models; if it finds any, it will load the latest model, and resume training that model from where it last stopped. For example, if EPOCHS is set to 200, then a pre-existing model trained for only 100 epochs (with the name model.epoch100.*) will be trained another 100 epochs. Hence, this option is particularly useful in cases in which the model training was interrupted for any reason before completion, saving time. This option could also be used to do more training on an already completely-trained model; however, this is unlikely to result in a substantially different model unless the training parameters are changed (a larger EPOCHS or PATIENCE, or a different LEARNING_RATE) or the training set is much different than before.

**DELETE_OLDER_MODELS:** As described above, the training algorithm initially saves a version of the model at each epoch that reaches a new loss minimum. When DELETE\_OLDER_MODELS is set to its default value of True, all but the final model is deleted at the end of training. Users may set this to False if they are interested in experimenting with the earlier models. (If multiple models are present, the current scripts always load the newest model; one must either remove the newer models or change the scripts to change this behavior.)

**GET_TEST_LOSS:** The scripts will calculate a few summary statistics for the training and validation sets, and will calculate the overall loss value of the model against those data sets. But it will only do the same for the test dataset if GET\_TEST\_LOSS is set to True. This is set to False by default, because it requires a redundant reloading of the entire test set (independent of the per-file loading done during formant tracking), which may take a substantial amount of time depending on your setup (e.g. about 2 additional hours for 5000 files in a recent run). If you're interested in these statistics, however, set this to True.

**OUT_EXT:** Extension (suffix) of the output formant track files; set to "txt" by default. You may wish to change this if you want to write the output files back into the input directory (as described in "Output" above) but already have .txt files there.

**BIN_SMOOTH_PASSES:** The initial formant parameters (frequencies, bandwidths, and amplitudes) predicted by the model can be smoothed via the binary smoothing algorithm; i.e. for each smoothing pass, the new value is a weighted average of the old value and the values of its neighbors (preceding and following analysis windows). The Python scripts in **IS2021** and **PaPE2021** did not provide this directly; it was instead performed by a post-tracking evaluation script. The tracking scripts in this directory do it directly. The default value of 10 provided the best results overall (as compared to the hand labels), including consonants as well as vowels; if your sole focus is vowels, a lower number may provide better results.

**FREQUENCIES_FIRST:** As described in the "Output" section above, the default order of parameters in the output files is frequencies first (F1 F2 ...), followed by bandwidths (B1 B2 ...) and amplitudes (A1 A2 ...). If this parameter is set to False, then the output parameters are sorted first by formant, i.e. F1 B1 A1 F2 B2 A2 ....

**REAL_AMPLITUDES:** As described in the "Output" section above, the "amplitudes" generated directly by the model are not actually final formant amplitudes, but rather weighting factors that are multiplied with the initial formant energy calculated from its frequency and bandwidth (see Eqn. 1 of the **IS2021** paper). By default, with REAL\_AMPLITUDES set to True, the scripts attempt to convert these weights to a "real" formant amplitude estimate, by plugging "f=F" into Eqn. 1. Note that this should be considered a lower bound estimate of the amplitude at the formant frequency, since it won't include contributions to the energy by the neighboring formants. This behavior can be changed to generate the original weighting factors by changing this configuration parameter to False.
