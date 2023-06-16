from scipy import signal

import numpy as np

import ajf_plts

import matplotlib.pyplot as plt

import pickle

import BeamModel as bm
from Simulation import perform_static_sim

with open("noisy_sig.pkl", "rb") as f:
    time, lhs_noisy = pickle.load(f)

beam = bm.Beam()
E25 = 2.601e10
beam.E = E25

P_full = np.array([68670, 107910, 71613, 77499, 73575])
S_full = np.array([0, 3.9000, 9.7500, 11.0500, 12.3500])
vel = 15  # metre per second

steps, disp = perform_static_sim(beam.Kg, P_full, S_full)

lhs_static = disp[bm.LHS_ROT_IDX] * 1e3  # convert to gx10^-3

# # Detrending
mean = np.mean(lhs_noisy[500:])
lhs_noisy = lhs_noisy - mean

# Filtering

# Dynamics Filter
# 4th order butterworth, fc= 1Hz (roughly f1/4)
fc = 1

b, a = signal.iirfilter(
    N=4, Wn=fc, btype="lowpass", ftype="butter", analog=False, fs=1000, output="ba"
)
denoise_filt = signal.lfilter(b, a, lhs_noisy)[5000:-5000]
denoise_filt -= denoise_filt[0]
time = np.arange(len(denoise_filt)) / 1000

# denoise_filt = np.zeros_like(dynamic_filt)


# Setup figure for plotting
plt.style.use(["./ajf_plts/base.mplstyle"])

fig, ax1 = plt.subplots(figsize=ajf_plts.single_wide_figsize, ncols=1, nrows=1)
# ax1, ax2 = axes.ravel()

# TIME SERIES
# l1, = ax1.plot(dynamicTime,
#          dynamicDisp,
#          ls = next(ajf_plts.line_sty_cyc),
#          color=next(ajf_plts.color_cyc_cb))

staticTime = (steps / vel).squeeze()
col = "C0"
(l2,) = ax1.plot(staticTime, lhs_static, color=col)
max_i = np.argmax(lhs_static)
ax1.axhline(lhs_static[max_i], linewidth=0.8, ls="dotted", color=col, zorder=-100)
ax1.axvline(staticTime[max_i], linewidth=0.8, ls="dotted", color=col, zorder=-100)

col = "C1"
(l3,) = ax1.plot(time, denoise_filt, color=col)
max_i = np.argmax(denoise_filt)
ax1.axhline(denoise_filt[max_i], linewidth=0.8, ls="dotted", color=col, zorder=-200)
ax1.axvline(time[max_i], linewidth=0.8, ls="dotted", color=col, zorder=-200)

ax1.set_xlabel("Time / s")
ax1.set_ylabel(r"Amplitude / $g\times10^{-3}$")

plt.tight_layout()

ax1.legend(
    [l2, l3],
    ["Static Model", "Denoise Filter, $a_f$"],
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
)

ajf_plts.save_fig(fig)
