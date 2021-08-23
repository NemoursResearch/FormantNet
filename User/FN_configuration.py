#!/usr/bin/env python
# coding: utf-8

# # FormantNet Configuration Code

# This code is used to parse the configuration file, if one exists, and save the global variables used by FormantNet into one object, referred to as **cfg** in the other scripts and passed around from function to function.

# In[ ]:


import configparser

class configuration(object):

    def __init__(self):

        self.TESTRUN = False
        self.NFORMANTS =  6
        self.NZEROS = 1
        self.DIFFWEIGHT = 0.15
        self.SAMPLERATE = 16000.0
        self.MAX_ANALYSIS_FREQ = self.SAMPLERATE / 2.0
        self.MAXFREQ = 8000.0
        self.MINFREQ = 0.0
        self.MAXBW = 5000.0
        self.MINBW = 20.0
        self.MAXAMP = 100.0
        self.MINAMP = -100.0
        self.WINDOW_LENGTH_MSEC = 32.0
        self.FRAME_STRIDE_MSEC = 5.0
        self.PREEMPH = 0.98
        self.SMOOTH_LINEAR = True
        self.ENV_SMOOTH_PASSES = 6
        self.FLOOR = 0.001
        self.SEQUENCE_LENGTH = 64
        self.SEQUENCE_STRIDE = self.SEQUENCE_LENGTH
        self.BATCH_SIZE = 32
        self.LSTM_LAYERS = 1
        self.DENSE_LAYERS = 1
        self.LSTM_UNITS = 512
        self.DENSE_UNITS = 512
        self.DENSE_ACTIVATION = 'relu'
        self.TOP_ACTIVATION = 'sigmoid'
        self.LEARNING_RATE = 0.0001
        self.ALLOW_RETRAIN = True
        self.EPOCHS = 200
        self.PATIENCE = 20
        self.DELETE_OLDER_MODELS = True
        self.GET_TEST_LOSS = False
        self.OUT_EXT = 'txt'
        self.REAL_AMPLITUDES = True
        self.FREQUENCIES_FIRST = True
        self.BIN_SMOOTH_PASSES = 10
        
        
    def configure(self, configFile=None):
        
        if configFile is not None:
            config = configparser.ConfigParser()
            config.read(configFile)

            self.TESTRUN = config['DEFAULT'].getboolean('TESTRUN', self.TESTRUN)
            self.NFORMANTS = config['DEFAULT'].getint('NFORMANTS', self.NFORMANTS)
            self.NZEROS = config['DEFAULT'].getint('NZEROS', self.NZEROS)
            self.DIFFWEIGHT = config['DEFAULT'].getfloat('DIFFWEIGHT', self.DIFFWEIGHT)
            self.SAMPLERATE = config['DEFAULT'].getfloat('SAMPLERATE', self.SAMPLERATE)
            self.MAX_ANALYSIS_FREQ = config['DEFAULT'].getfloat('MAX_ANALYSIS_FREQ', self.SAMPLERATE / 2.0)
            self.MAXFREQ = config['DEFAULT'].getfloat('MAXFREQ', self.MAXFREQ)
            self.MINFREQ = config['DEFAULT'].getfloat('MINFREQ', self.MINFREQ)
            self.MAXBW = config['DEFAULT'].getfloat('MAXBW', self.MAXBW)
            self.MINBW = config['DEFAULT'].getfloat('MINBW', self.MINBW)
            self.MAXAMP = config['DEFAULT'].getfloat('MAXAMP', self.MAXAMP)
            self.MINAMP = config['DEFAULT'].getfloat('MINAMP', self.MINAMP)
            self.WINDOW_LENGTH_MSEC = config['DEFAULT'].getfloat('WINDOW_LENGTH_MSEC', self.WINDOW_LENGTH_MSEC)
            self.FRAME_STRIDE_MSEC = config['DEFAULT'].getfloat('FRAME_STRIDE_MSEC', self.FRAME_STRIDE_MSEC)
            self.PREEMPH = config['DEFAULT'].getfloat('PREEMPH', self.PREEMPH)
            self.SMOOTH_LINEAR = config['DEFAULT'].getboolean('SMOOTH_LINEAR', self.SMOOTH_LINEAR)
            self.ENV_SMOOTH_PASSES = config['DEFAULT'].getint('ENV_SMOOTH_PASSES', self.ENV_SMOOTH_PASSES)
            self.FLOOR = config['DEFAULT'].getfloat('FLOOR', self.FLOOR)
            self.SEQUENCE_LENGTH = config['DEFAULT'].getint('SEQUENCE_LENGTH', self.SEQUENCE_LENGTH)
            self.SEQUENCE_STRIDE = config['DEFAULT'].getint('SEQUENCE_STRIDE', self.SEQUENCE_LENGTH)
            self.BATCH_SIZE = config['DEFAULT'].getint('BATCH_SIZE', self.BATCH_SIZE)
            self.LSTM_LAYERS = config['DEFAULT'].getint('LSTM_LAYERS', self.LSTM_LAYERS)
            self.DENSE_LAYERS = config['DEFAULT'].getint('DENSE_LAYERS', self.DENSE_LAYERS)
            self.LSTM_UNITS = config['DEFAULT'].getint('LSTM_UNITS', self.LSTM_UNITS)
            self.DENSE_UNITS = config['DEFAULT'].getint('DENSE_UNITS', self.DENSE_UNITS)
            self.DENSE_ACTIVATION = config['DEFAULT'].get('DENSE_ACTIVATION', self.DENSE_ACTIVATION)
            self.TOP_ACTIVATION = config['DEFAULT'].get('TOP_ACTIVATION', self.TOP_ACTIVATION)
            self.LEARNING_RATE = config['DEFAULT'].getfloat('LEARNING_RATE', self.LEARNING_RATE)
            self.ALLOW_RETRAIN = config['DEFAULT'].getboolean('ALLOW_RETRAIN', self.ALLOW_RETRAIN)
            self.EPOCHS = config['DEFAULT'].getint('EPOCHS', self.EPOCHS)
            self.PATIENCE = config['DEFAULT'].getint('PATIENCE', self.PATIENCE)
            self.DELETE_OLDER_MODELS = config['DEFAULT'].getboolean('DELETE_OLDER_MODELS', self.DELETE_OLDER_MODELS)
            self.GET_TEST_LOSS = config['DEFAULT'].getboolean('GET_TEST_LOSS', self.GET_TEST_LOSS)
            self.OUT_EXT = config['DEFAULT'].get('OUT_EXT', self.OUT_EXT)
            self.REAL_AMPLITUDES = config['DEFAULT'].getboolean('REAL_AMPLITUDES', self.REAL_AMPLITUDES)
            self.FREQUENCIES_FIRST = config['DEFAULT'].getboolean('FREQUENCIES_FIRST', self.FREQUENCIES_FIRST)
            self.BIN_SMOOTH_PASSES = config['DEFAULT'].getint('BIN_SMOOTH_PASSES', self.BIN_SMOOTH_PASSES)
        
        if self.MAX_ANALYSIS_FREQ > self.SAMPLERATE / 2.0:
            print("MAX_ANALYSIS_FREQ value", self.MAX_ANALYSIS_FREQ, 
                  "is too high; it must be less than or equal to half the SAMPLERATE.")
            self.MAX_ANALYSIS_FREQ = self.SAMPLERATE / 2.0
            print("Reset MAX_ANALYSIS_FREQ to", self.MAX_ANALYSIS_FREQ)
            
        if self.MINBW <= 0:
            print("MINBW value", self.MINBW, "is too low; it must be greater than 0.")
            self.MINBW = 1.0
            print("Reset self.MINBW to 1.0")
            
        if self.TESTRUN:
            self.EPOCHS = 3

        # Width of signal analysis window in samples
        self.WINDOW_LENGTH_SAMPLES = int(self.WINDOW_LENGTH_MSEC * self.SAMPLERATE / 1000.0)
        # Number of samples between analysis window start points
        self.FRAME_STRIDE_SAMPLES = int(self.FRAME_STRIDE_MSEC * self.SAMPLERATE / 1000.0)

        # Spectral resolution: Number of points per input spectrum
        self.SPECTRUM_NPOINTS = int(self.WINDOW_LENGTH_MSEC * self.MAX_ANALYSIS_FREQ / 1000.0) + 1

        self.NSUM = self.NFORMANTS + self.NZEROS   # Total number of resonances (poles + zeros)
        self.NPARAMS = self.NFORMANTS*3 + self.NZEROS*2   # Total number of model output features


    def report_status(self):
        
        print("\n\nSUMMARY OF CONFIGURATION SETTINGS:")
        print("Test Run:", self.TESTRUN)

        print("\n# Formants (poles) to be modeled:", self.NFORMANTS)
        print("# Antiformants (zeros) to be modeled:", self.NZEROS)
        print("Total # of model output parameters:", self.NPARAMS)

        print("\nDelta-frequency weight:", self.DIFFWEIGHT)
        print("Wavefile sampling rate:", self.SAMPLERATE, "Hz")
        print("Frequency analysis range: 0 -", self.MAX_ANALYSIS_FREQ, "Hz")

        print("\nLower and upper limits on formant parameter predictions:")
        print("Frequencies:", self.MINFREQ, "-", self.MAXFREQ, "Hz")
        print("Bandwidths:", self.MINBW, "-", self.MAXBW, "Hz")
        print("Amplitude correction factors:", self.MINAMP, "-", self.MAXAMP, "dB")

        print("\nAnalysis window length:", self.WINDOW_LENGTH_MSEC, 
              "msec (" + str(self.WINDOW_LENGTH_SAMPLES), "samples)")
        print("Spectral resolution (model input size):", self.SPECTRUM_NPOINTS, "bins")
        print("Analysis window spacing: Once every", self.FRAME_STRIDE_MSEC, 
              "msec (" + str(self.FRAME_STRIDE_SAMPLES), "samples)")
        print("Pre-emphasis factor:", self.PREEMPH)
        print("Perform smoothing on linear-scale envelopes (rather than dB-scale):", self.SMOOTH_LINEAR)
        print("# of envelope smoothing passes:", self.ENV_SMOOTH_PASSES)
        print("Floor value added to linear envelopes before conversion to dB:", self.FLOOR)

        print("\nTraining sequence length:", self.SEQUENCE_LENGTH, "frames")
        print("Training sequence spacing: every", self.SEQUENCE_STRIDE, "frames")
        print("Batch size (sequences per training batch):", self.BATCH_SIZE)

        print("\n# of LSTM layers:", self.LSTM_LAYERS)
        print("LSTM layer size:", self.LSTM_UNITS, "units")
        print("# of Dense layers:", self.DENSE_LAYERS, "(including output layer)")
        if self.DENSE_LAYERS > 1:
            print("Dense hidden layer size:", self.DENSE_UNITS, "units")
            print("Activation function of dense hidden layers:", self.DENSE_ACTIVATION)
        print("Activation function of output layer:", self.TOP_ACTIVATION)

        print("\nAllow retraining of pre-existing model?:", self.ALLOW_RETRAIN)
        print("Optimizer learning rate:", self.LEARNING_RATE)
        print("Maximum # of training epochs:", self.EPOCHS)
        print("Convergence patience:", self.PATIENCE, "epochs")
        print("Delete older models after training?:", self.DELETE_OLDER_MODELS)

        print("\nCalculate test set stats and loss?:", self.GET_TEST_LOSS)
        print("Output file extension: ." + self.OUT_EXT)
        print("Output real (predicted) amplitudes?:", self.REAL_AMPLITUDES)
        print("Output in frequencies-first order?:", self.FREQUENCIES_FIRST)
        print("# of binomial smoothing passes on output:", self.BIN_SMOOTH_PASSES)
        print("\n")


# In[ ]:


if __name__ == "__main__":
    cfg = configuration()
    cfg.configure(None)
    cfg.report_status()

