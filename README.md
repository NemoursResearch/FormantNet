# FormantNet
Software and data supporting research on formant tracking via neural networks done at Nemours.

The subdirectory **User** has user-friendly Python scripts executable on the command-line, with various options available either on the command line or via an optional configuration file. If you're interested in training and running formant-tracking models on your own data, these are the files you should use.

The subdirectory **IS2021** holds files specifically used for the training and evaluation of the models reported by Lilley and Bunnell in the manuscript accepted to InterSpeech 2021.

The subdirectory **PaPE2021** holds similar files, used for the training of the models Lilley and Bunnell reported at the Phonetics and Phonology in Europe (PaPE) 2021 conference. These models are a more advanced version of the IS2021 models, producing better results.

In the interest of having accurate and complete scientific records, those files in **IS2021** and **PaPE2021** are completely identical to the ones used for research, except that (1) additional comments have been added, and (2) code used only to evaluate the models on data other than TIMIT data has been removed. Owing to their particular history and evolution, those files are not particularly user-friendly. We recommend the scripts in **User** for your own projects instead.

The **R** directory holds some R scripts we used in the process of research and development.

See the README files in each directory for more details. 

