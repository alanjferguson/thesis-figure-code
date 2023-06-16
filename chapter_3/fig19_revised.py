import pandas as pd
import numpy as np

from scipy import signal

import matplotlib.pyplot as plt

import ajf_plts

import BeamModel as bm
from Simulation import perform_dynamic_sim

import pickle

T_s = 0.000605

plt.style.use(['./ajf_plts/base.mplstyle'])

np.random.seed(0xC0FFEE)


#############################################################################
# Simulate 2 trucks for first subplot
###############################################################################

beam = bm.Beam()
E25 = 2.601e10
beam.E = E25

# perform_dynamic_sim doesn't allow varying velocity so we construct this signal
# in 3 parts and join together

P_empty = np.array([61803, 50031, 17658, 23544, 21582])
S_empty = np.array([0, 3.8000, 9.1500, 10.3500, 11.6000])
P_full = np.array([68670, 107910, 71613, 77499, 73575]) * 1.081
S_full = np.array([0, 3.9000, 9.7500, 11.0500, 12.3500])

############################################################
# VEHICLE 1
############################################################
vel = 28  # metres per second
t1, disp_d1 = perform_dynamic_sim(beam.Kg, P_empty, S_empty, vel, time_step=T_s)[:2]

############################################################
# VEHICLE 2
############################################################
vel = 30  # metres per second
t2, disp_d2 = perform_dynamic_sim(beam.Kg, P_full, S_full, vel, time_step=T_s)[:2]

############################################################
# VEHICLE 2
############################################################
start_offset = 619
v1_offset = 620.5
v2_offset = 629
end_offset = 633

dyn_time = np.concatenate(
    ([start_offset], t1 + v1_offset, t2 + v2_offset, [end_offset])
)

dyn_disp = np.concatenate(
    (
        [disp_d1[bm.LHS_ROT_IDX, 0]],
        disp_d1[bm.LHS_ROT_IDX, :],
        disp_d2[bm.LHS_ROT_IDX, :],
        [disp_d2[bm.LHS_ROT_IDX, -1]],
    )
)

time = np.arange(start_offset, end_offset, T_s)

dyn_disp = np.interp(time, dyn_time, dyn_disp) * 1e3

################################################################################
# PAD + ADD NOISE
################################################################################

with open("./noise_fft.pkl", "rb") as f:
    noise_fft = pickle.load(f)

noise_fft = np.array(noise_fft, dtype="complex")

def get_noise_sample(n_samples):
    Np = (len(noise_fft) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    noise_fft[1:(Np + 1)] *= phases
    noise_fft[-1:(-1 - Np):-1] = np.conj(noise_fft[1:(Np + 1)])
    return np.fft.ifft(noise_fft).real[:n_samples]


PAD_LEN = int(20 / T_s)


def add_noise_and_pad(s):
    y = np.zeros(len(s) + 2 * PAD_LEN)
    y[PAD_LEN:-PAD_LEN] = s
    y += get_noise_sample(len(y))
    return y


lhs_noisy = add_noise_and_pad(dyn_disp)


############################################################
# DETREND
############################################################

lhs_noisy -= np.mean(lhs_noisy[PAD_LEN:PAD_LEN+1000])

############################################################
# FILTER AND UNPAD
############################################################


def unpad(s):
    return s[PAD_LEN:-PAD_LEN]


def lp_filter(s):
    FC = 1.0
    ORDER = 4
    b, a = signal.iirfilter(
        N=ORDER,
        Wn=FC,
        btype="lowpass",
        ftype="butter",
        analog=False,
        fs=1 / T_s,
        output="ba",
    )
    return signal.lfilter(b, a, s)


lhs_noisy = signal.detrend(lhs_noisy)
lhs_filt = unpad(lp_filter(lhs_noisy))

lhs_filt -= lhs_filt[0]  # makes sure starts at same level as static model

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

data.x *= 1e3

###############################################################################
# Extract signals to plot
###############################################################################
x_min = 619
x_max = 633

# t_vals = data.index.values[
#     np.where(np.logical_and(x_min <= data.index.values, data.index.values <= x_max))
# ]
# 
# data = data.loc[t_vals]

data.x -= data.x.iloc[0]

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

data.x.plot(ax=ax1, label="Measured")

ax1.set_xlim(619, 633)
# ax1.set_ylim(-0.6e-4, 2.6e-4)

ax1.set_xlabel("Time / s")
ax1.set_ylabel("Rotation / rad")

fig.tight_layout()
ajf_plts.savefig(fig, 'figLineBridgeRot619-633.pdf')

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

data.x.plot(ax=ax1)

ax1.set_xlim(735, 765)

ax1.set_xlabel("Time / s")
ax1.set_ylabel("Rotation / rad")

fig.tight_layout()
ajf_plts.savefig(fig, 'figLineBridgeRot735-765.pdf')
