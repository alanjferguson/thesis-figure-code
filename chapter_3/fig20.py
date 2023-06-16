import pandas as pd

import numpy as np

from scipy import signal

import matplotlib as mpl
import matplotlib.pyplot as plt

import ajf_plts

import code

plt.style.use(['ajf.mplstyle'])

np.random.seed(0xC0FFEE)

data = pd.read_csv("./Voltage.txt", sep="\t", names=["x", "y", "z"], skiprows=7)

T_s = 0.000605

data["t"] = np.arange(len(data)) * T_s

data = data.set_index("t")

chan_sens = 2.0

data[["x", "y", "z"]] /= chan_sens

data = data.apply(lambda s: signal.detrend(s), axis=0)

fc = 1.0

b, a = signal.iirfilter(
    N=4, Wn=fc, btype="lowpass", ftype="butter", analog=False, fs=1.0 / T_s, output="ba"
)

data = data.apply(lambda s: signal.lfilter(b, a, s), axis=0)

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

(data.x*10e3).plot(ax=ax1)

ax1.set_xlim(619, 633)

ax1.set_xlabel("Time / s")
ax1.set_ylabel("Rotation / rad")

fig.tight_layout()
fig.savefig("output/fig20c.pdf")

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

(data.x*10e3).plot(ax=ax1)

ax1.set_xlim(735, 765)

ax1.set_xlabel("Time / s")
ax1.set_ylabel("Rotation / rad")

fig.tight_layout()
fig.savefig("output/fig20d.pdf")
