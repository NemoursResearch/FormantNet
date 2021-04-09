#!/bin/bash

#This script takes two command-line variables, an experiment label and sub-experiment label, that together combine to form the output directory resulting from a FormantTracking experiment. E.g. the command line:

#get_report.sh mvt19 f6z1

# looks for output files in the directory mvt19_f6z1/timit/.

#Output:
# 1) Will generate a text file timdb.$exp.$sub.txt, with one line for every frame of every file in the VTR Formants database. Each line consists of columns for the filename, time point in msec, and then the frequencies of each formant (pole and zero) estimated by the model.

## NOTE: Our input files for the experiments submitted to IS 2021 had a 5-msec frame rate, and so the model output files generated formant estimates every 5 msec. However, the VTR Formants database only has formant estimates once every 10 msec. So the script below only compiles the model outputs of every other frame.
## NOTE ALSO: Probably due to different window lengths in the analysis, the number of frames per file is slightly different between the VTR Formants database and our models. The script below uses nframes2.txt, which reports the number of frames per file as per VTR Formants, to collect the same number of frames from the model output files.

# 2) The timdb.$exp.$sub.txt is used as input to the R script timitscript.R, which generates a new file, timreport.$exp.$sub.txt, with a comparison of the formant estimates to the VTR-TIMIT reference values. See that script for more details.

#Prerequisites: This script will look for nframes2.txt (in a sister directory ../data), timitscript.R (in the directory above the CWD), and the prerequisite files of timitscript.R -- see that file for details.  See the README file for details about nframes2.txt.

exp=$1
sub=$2
if [[ -e timreport.$exp.$sub.txt ]]; then
    echo "timreport.$exp.$sub.txt already exists; skipping."
elif [[ ! -e ${exp}_$sub/timit/${exp}_${sub}_test_dr1_felc0_si1386.abs ]]; then
    echo "Input file ${exp}_$sub/timit/${exp}_${sub}_test_dr1_felc0_si1386.abs does not exist; skipping."
else
    echo $exp $sub;
    cat ../data/nframes2.txt | while read nfr name; do
	awk '$5~/0.0/ {printf("%s %4.0f", $1, $5); for (i=14; i<=NF; i=i+3) printf(" %8.3f", $i); printf("\n")}' ${exp}_$sub/timit/${exp}_${sub}_$name.abs | head -$nfr
    done > timdb.$exp.$sub.txt
    Rscript ../timitscript.R timdb.$exp.$sub.txt | sed "s:\[1\] ::" > timreport.$exp.$sub.txt
fi

