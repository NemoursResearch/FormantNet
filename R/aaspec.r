fv = 10*(0:500)

axspec = formant(750, 100, 500, 5000)
axspec = axspec * formant(1250, 100, 500, 5000)
axspec = axspec * formant(2750, 100, 500, 5000)
axspec = axspec * formant(3250, 100, 500, 5000)
axspec = axspec * formant(4750, 100, 500, 5000)
axspec = axspec * hpc2(5, fv, 500)

plot(fv, 20*log10(axspec), type='l', ylab="Log Magnitude (dB)", xlab="Frequency (Hz)", ylim=c(-20,40),
		 main="F1 Frequency & Spectrum Amplitude")
