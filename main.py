import autostep as asp
from scipy.io import wavfile
import numpy as np

nbCol = 13
nbRow = 13
Fs = 96000
tSim = 30
x = np.zeros((nbRow*nbCol, tSim*Fs))
i = 0
out = np.zeros(tSim*Fs+192000-1)

for xx in np.arange(nbCol):
    for yy in np.arange(nbRow):
        Fs, h = wavfile.read('Omni/x{:02d}y{:02d}.wav'.format(xx, yy))

        x[i, :] = np.random.normal(0, 1, size=tSim*Fs)
        out += np.convolve(h, x[i, :])
        i += 1

h_as = asp.autostep(x, out, nbCol*nbRow, 192000)
