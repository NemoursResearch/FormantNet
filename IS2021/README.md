# IS2021
Software and data supporting Lilley &amp; Bunnell (2021) InterSpeech paper.

In the interest of having accurate and complete scientific records, these files are completely identical to the ones used for the research reported in that manuscript, except that (1) additional comments have been added, and (2) code used only to evaluate the models on data other than TIMIT data has been removed.

## Model training:

The Python code used to train the models, and run them on the evaluation data, are provided as both Jupyter notebook files (\*.ipynb) and executable Python scripts (\*.py). TensorFlow 2.3 or above is required to run them. Each file contains the code for one particular model described in the manuscript:

* **CNN3**: A model with 3 convolutional layers.
* **LSTM1**: A model with 1 LSTM layer.
* **BLSTM1**: A model with 1 bidirectional LSTM layer.
* **LSTM3**: A model with 3 LSTM layers.

Note that the code makes some assumptions based on the circumstances of our computational setup at the time; one would need to edit the code accordingly for their own setup. You may also notice differences in code between the four models. Some of these differences are due to the necessary differences between the 4 experiments, of course, while other differences are irrelevant to the training and evaluation, and are simply due to the evolution of the code over time, e.g. to make the code more readable and generalizable. We intend to provide a more uniform and user-friendly version of the code for general use soon.

### Execution:

Each script is run with two or three command-line parameters (usually specifying the desired number of poles and zeros); see documentation within each script for details. Output consisting of data statistics, model specifications, and script progress reports, including training and validation loss, can be saved to an output file, e.g.:

> CNN3.py 6 1 > CNN3.f6z1.out

The script may take hours to days to run. (In our case, running the script on machines using 24 parallel cores per job, these experiments ran between half a day and 2 days.)

### Input:

The script looks for the two provided files, timit_ordered1.txt and VTRlist0.txt, which are file lists of the TIMIT dataset (Garofolo et al. 1993) and the VTR Formants database (Deng et al. 2006) respectively. The script will then look for the input data files listed in these lists in the locations specified in the scripts. Due to the license restrictions on TIMIT, we cannot provide our input files here; see our manuscript for details of their construction. The files consist of a log-scale (dB) spectral envelope calculated from each frame of input. There is one input file for each wavefile. The file is in binary format, and starts with a 24-byte header. The first 8 bytes consists of two 4-byte integers specifying first the number of frames, and then number of data points per frame (the spectral resolution), which for IS2021 was kept at a constant 257 points. Following the header are the spectra themselves, which are stored as float values.


### Output:

The output models and evaluation files will be saved in a directory with a name like mvt24\_f6z1/ (where the first part, e.g. "mvt24", is a unique designation for the script, and the second part, e.g. "f6z1" indicates the selected command-line parameters, e.g. 6 formants and 1 zero). The model files (stored in tensorflow model weight format, consisting of a data file and an index file) are stored directly in this directory. A subdirectory, e.g. mvt24_f6z1/timit/, will hold the output formant track files, one for each input file.

The output formant track files are stored as text files with the suffix .abs. The following is the first few lines from an example output file (with values rounded to 1 place after the decimal point, for brevity):

> test\_dr1\_felc0\_si1386 AA 1 1 0.0 200.0 60 0 0 60 40 2 7   578.0 346.4 -20.5 1081.1 235.4 -27.2 2435.6 760.9 -30.0 3267.7 732.1 -28.2 4501.1 1276.0 -31.2 6511.0 2621.8 -42.5 -3026.4 4877.2 0.0  
> test\_dr1\_felc0\_si1386 AA 1 1 5.0 200.0 60 0 0 60 40 2 7   683.5 229.3 -19.6 953.2 269.2 -19.7 2644.0 851.6 -26.1 3158.2 383.5 -32.2 4267.3 556.6 -35.8 6737.9 2716.3 -41.8 -1850.0 4717.65 0.0  
> test\_dr1\_felc0\_si1386 AA 1 1 10.0 200.0 60 0 0 60 40 2 7   718.8 237.9 -18.1 946.5 333.1 -20.0 2728.7 566.0 -29.6 3144.7 285.4 -36.7 4302.6 539.9 -38.1 7174.0 1398.4 -48.1 -1329.9 4069.2 0.0  

This format was designed for the specific interests of our laboratory; the first 12 columns are placeholders for parameters of interest of us to be filled in later, with the exception of column 1 (the input filename), and column 4, the frame's time-stamp. Note that our input files were generated at a 5-msec frame rate, and so the time stamps are filled in assuming a 5-msec frame rate (this can be easily changed in the script).

Column 13 indicates the total number of resonances (7 here: 6 poles and 1 zero). Following column 13 are the actual predicted formant estimates, in the order F1 B1 A1, F2 B2 A2, etc. -- first the poles, in order of increasing mean frequency, and then zeros, in order of increasing absolute mean frequency. Note that the zeros are given a negative frequency to distinguish them from the poles, and the meaningless placeholder value "0.0" inserted for their amplitudes (which are not estimated by the model).

Note that the formant track parameters generated by these Python scripts are not smoothed; the binomial smoothing described in our manuscript was done by the evaluation scripts, as described below.


## Evaluation scripts

Further scripts have been provided, which (1) compile all the estimated frequencies in the .abs files into a single file, and (2) run statistical analyses of the frequencies against the VTR Formant database. Again, these files are identical to those used for the IS 2021 analysis, except with the addition of extra commentary. See the individual scripts for details.

* **get_report.sh**: This shell script collects the formant frequencies generated in all the individual .abs files and outputs them to a single "database" (text) file, with a name like timdb.mvt13.f6z1.txt, with one line per selected frame. Note that since our input and output files were generated at a 5-msec frame rate, while the VTR Formants database reports frequency measurements only every 10 msec, this script actually only collects the frequencies of every other frame (starting at 0 msec, then 10, etc.). The output database file is used as the input for the following 3 scripts. This script also runs timitscript.R on the database file.

* **timitscript.R**: This R script generates a statistical report on the error of the formant estimates generated by the model, in comparison to the VTR-TIMIT F1-F3 reference values. The statistics are mostly MAEs (mean absolute errors) and RMSEs (root mean square errors) of various subsets of the data and are meant to be comparable those found in the literature on VTR-TIMIT, e.g. Dissen et al. (2019) and Schiel & Zitzelberger (2018). This script does no smoothing.

* **timitscript2.R**: This R script is identical to timitscript.R, except that it also runs binomial smoothing on the input data (how much smoothing is determined by command-line parameters), and generates a report on the smoothed data. The statistics it calculates are identical to those of timitscript.R.

* **timitscript3.R**: This R script generates a different set of statistics, consisting of MAEs, mean percentage errors (MPEs), and "formant detection rates" (FDRs) as per Gowda et al. (2017), and is meant to be compared to their results. Like timitscript2.R, it generates reports for both unsmoothed and smoothed data.

### Required files

The scripts above require the following text files to run, which we generated from the data in the TIMIT and VTR Formants databases. Due to the license restrictions on those datasets and works derived from them, we cannot provide these files, but we can describe their content. The order of the files listed in each of these must be the same across all four files.

* **nframes2.txt**: This text file is a list of the number of frames of each file with formant estimates in the VTR dataset, followed by the filename, e.g.:

> 552 test_dr1_felc0_si1386  
> 340 test_dr1_felc0_si2016  
> 418 test_dr1_felc0_si756  

This is used by **get_report.sh** to compile the data in the specified order and make sure that the number of frames of data extracted by that script is the same as that found in the VTR data--for a grand total of 160,511 frames.

* **textdb2.txt**: A 160511-line text file holding the hand-corrected formant measurements of the VTR Formants database, in the following format:

> test_dr1_felc0_si1386 test dr1 f felc0 si si1386    0  655.835 1489.377 2556.268 3603.549  220.735  322.101  397.020  460.449  
> test_dr1_felc0_si1386 test dr1 f felc0 si si1386   10  765.140 1513.626 2589.981 3608.018  211.584  357.977  356.811  453.889  
> test_dr1_felc0_si1386 test dr1 f felc0 si si1386   20  749.220 1517.242 2615.817 3658.496  204.688  335.453  373.721  447.311  

The columns indicate the filename; subset (test or train); dialect region; gender; speaker; TIMIT sentence type; sentence ID; and time stamp, followed by F1-F4 and B1-B4.

* **labels3.txt**: A 160511-line text file holding phonological information for each frame, derived from the TIMIT phone labels, in the following format:

> test_dr1_felc0_si1386 130 sil vl sil  
> test_dr1_felc0_si1386 140 q vl stop  
> test_dr1_felc0_si1386 150 q vl stop  
> test_dr1_felc0_si1386 160 ih vd vow  

The columns indicate the filename; the time stamp; the TIMIT phonetic label; whether the label indicates a phone that is phonologically voiced (vd) or voiceless (vl); and the broad phonetic category of the phone (one of sil[ence], vow[el], semi[vowel], nasal, fric[ative], aff[ricate], or stop).

* **pitchlabs.txt**: A 160511-line text file holding information on whether each frame is voiced or voiceless; our data was calculated with the Praat pitch tracker, using the default settings, as per Schiel & Zitzelberger (2018). The format is as follows:

> test_dr1_felc0_si1386 130 vl 0  
> test_dr1_felc0_si1386 140 vd 259.41  
> test_dr1_felc0_si1386 150 vd 247.19  

The columns indicate the filename; the time stamp; the voicing status of the frame; and the pitch estimate (not used by the scripts above).


## References:

* J. Lilley and H. T. Bunnell, "Unsupervised training of a DNN-based formant tracker," manuscript to appear in *Interspeech 2021*.
* Y. Dissen, J. Goldberger, and J. Keshet, “Formant estimation and tracking: A deep learning approach,” *Journal of the Acoustical Society of America*, vol. 145, no. 2, pp. 642-653, Feb. 2019.
* L. Deng, X. Cui, R. Pruvenok, Y. Chen, S. Momen, and A. Alwan, “A database of vocal tract resonance trajectories for research in speech processing,” in *Proceedings of ICASSP 2006—IEEE International Conference on Acoustics, Speech and Signal Processing*, vol. 1, pp. I-I, 2006.  Database downloaded from http://www.seas.ucla.edu/spapl/VTRFormants.html
* J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, D. S. Pallett, N. L. Dahlgren, and V. Zue, *The DARPA TIMIT acoustic-phonetic continuous speech corpus*. Linguistic Data Consortium, 1993.
* D. Gowda, M. Airaksinen, and P. Alku, “Quasi-closed phase forward-backward linear prediction analysis of speech for accurate formant detection and estimation,” *Journal of the Acoustical Society of America*, vol. 142, no. 3, pp. 1542-1553, 2017. Doi: 10.1121/1.5001512
* F. Schiel and T. Zitzelberger, “Evaluation of automatic formant trackers,” in *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan, May 7-12, 2018, pp. 2843-2848.
* P. Boersma and D. Weenink, “Praat, a system for doing phonetics by computer,” *Glot International*, vol. 5, no. 9/10, pp. 341-345, 2002.

