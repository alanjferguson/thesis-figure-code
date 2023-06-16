from scipy import signal

import numpy as np

import matplotlib.pyplot as plt

import pickle

import ajf_plts

import BeamModel as bm
from Simulation import perform_dynamic_sim
from Simulation import perform_static_sim

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

###############################################################################
# Simulate train of 3 vehicles
###############################################################################

F_sim = 1000
T_sim = 1.0 / F_sim

beam = bm.Beam()
E25 = 2.601e10
beam.E = E25

# perform_dynamic_sim can't vary velocity so we construct this signal
# in 3 parts and join together

P_empty = np.array([61803, 50031, 17658, 23544, 21582])
S_empty = np.array([0, 3.8000, 9.1500, 10.3500, 11.6000])
P_full = np.array([68670, 107910, 71613, 77499, 73575])
S_full = np.array([0, 3.9000, 9.7500, 11.0500, 12.3500])

############################################################
# VEHICLE 1
############################################################
vel = 25  # metres per second
s1, disp_s1 = perform_static_sim(beam.Kg, P_full, S_full)
st1 = s1 / vel
t1, disp_d1 = perform_dynamic_sim(beam.Kg, P_full, S_full, vel, time_step=T_sim)[:2]

############################################################
# VEHICLE 2
############################################################
vel = 20  # metres per second
s2, disp_s2 = perform_static_sim(beam.Kg, P_empty, S_empty)
st2 = s2 / vel
t2, disp_d2 = perform_dynamic_sim(beam.Kg, P_empty, S_empty, vel, time_step=T_sim)[:2]

############################################################
# VEHICLE 3
############################################################
vel = 22  # metres per second
s3, disp_s3 = perform_static_sim(beam.Kg, 1.25 * P_empty, S_full)
st3 = s3 / vel
t3, disp_d3 = perform_dynamic_sim(
    beam.Kg, 1.25 * P_empty, S_full, vel, time_step=T_sim
)[:2]

############################################################
# PIECE TOGETHER TIME VECTORS
############################################################
v1_offset = 0.5
v2_offset = 4
v3_offset = 9
end_offset = 12

dyn_time = np.concatenate(
    ([0], t1 + v1_offset, t2 + v2_offset, t3 + v3_offset, [end_offset])
)

stat_time = np.concatenate(
    ([0], st1 + v1_offset, st2 + v2_offset, st3 + v3_offset, [end_offset])
)

############################################################
# PIECE TOGETHER LHS ROT VECTORS
############################################################

dyn_disp = np.concatenate(
    (
        [disp_d1[bm.LHS_ROT_IDX, 0]],
        disp_d1[bm.LHS_ROT_IDX, :],
        disp_d2[bm.LHS_ROT_IDX, :],
        disp_d3[bm.LHS_ROT_IDX, :],
        [disp_d3[bm.LHS_ROT_IDX, -1]],
    )
)

stat_disp = np.concatenate(
    (
        [disp_s1[bm.LHS_ROT_IDX, 0]],
        disp_s1[bm.LHS_ROT_IDX, :],
        disp_s2[bm.LHS_ROT_IDX, :],
        disp_s3[bm.LHS_ROT_IDX, :],
        [disp_s3[bm.LHS_ROT_IDX, -1]],
    )
)

############################################################
# RESAMPLE TO FILL IN GAPS BETWEEN VEHICLES
############################################################

time = np.arange(0, end_offset, T_sim)

dyn_disp = np.interp(time, dyn_time, dyn_disp) * 1e3  # convert to gx10^-3
stat_disp = np.interp(time, stat_time, stat_disp) * 1e3  # convert to gx10^-3

###############################################################################
# PAD + ADD NOISE
###############################################################################

with open("./noise_fft.pkl", "rb") as f:
    noise_fft = pickle.load(f)

noise_fft = np.array(noise_fft, dtype="complex")


def get_noise_sample(n_samples):
    Np = (len(noise_fft) - 1) // 2
    phases = rng.random(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    noise_fft[1 : Np + 1] *= phases
    noise_fft[-1 : -1 - Np : -1] = np.conj(noise_fft[1 : Np + 1])
    return np.fft.ifft(noise_fft).real[:n_samples]


PAD_LEN = 10 * F_sim


def add_noise_and_pad(s):
    y = np.zeros(len(s) + 2 * PAD_LEN)
    y[PAD_LEN:-PAD_LEN] = s
    n = get_noise_sample(len(y))
    y += n
    return y, n


lhs_noisy, noise = add_noise_and_pad(dyn_disp)


############################################################
# DETREND
############################################################

lhs_noisy -= np.mean(lhs_noisy[PAD_LEN:-PAD_LEN])
noise -= np.mean(noise[PAD_LEN:-PAD_LEN])

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
        fs=F_sim,
        output="ba",
    )
    return signal.lfilter(b, a, s)


lhs_filt = unpad(lp_filter(lhs_noisy))

###############################################################################
# PROMINENCES
###############################################################################

window_len = ((34 + 13) / 20) * F_sim * 1.1

# STATIC
stat_peaks, _ = signal.find_peaks(stat_disp, distance=window_len, wlen=window_len)
stat_prominences, stat_l_bases, stat_r_bases = signal.peak_prominences(
    stat_disp, stat_peaks, wlen=window_len
)
stat_bases = np.minimum(stat_disp[stat_l_bases], stat_disp[stat_r_bases])

# DYNAMIC
dyn_peaks, _ = signal.find_peaks(lhs_filt, distance=window_len, wlen=window_len)
dyn_prominences, dyn_l_bases, dyn_r_bases = signal.peak_prominences(
    lhs_filt, dyn_peaks, wlen=window_len
)
dyn_bases = np.minimum(lhs_filt[dyn_l_bases], lhs_filt[dyn_r_bases])

###############################################################################
# PLOTTING
###############################################################################

# Setup figure for plotting
plt.style.use(["./ajf_plts/base.mplstyle"])
figsize = (ajf_plts.text_width_inches, 1.5 * ajf_plts.fig_height_inches)
fig, ax1 = plt.subplots(figsize=figsize, ncols=1, nrows=1)

# TIME SERIES
(l2,) = ax1.plot(time, stat_disp)

for t, p, b in zip(time[stat_peaks], stat_disp[stat_peaks], stat_bases):
    ax1.annotate(
        text="",
        xy=(t, p),
        xytext=(t, b),
        zorder=200,
        arrowprops=dict(ec="C0", fc="C0", arrowstyle="<|-|>", shrinkA=0, shrinkB=0),
    )

ax1.hlines(
    y=stat_disp[stat_peaks]-stat_prominences,
    xmin=time[stat_peaks] - 0.5*window_len/F_sim,
    xmax=time[stat_peaks] + 0.05*window_len/F_sim,
    zorder=100,
    lw=0.8,
    ls="dotted",
)
ax1.hlines(
    y=stat_disp[stat_peaks],
    xmin=time[stat_peaks] - 0.15*window_len/F_sim,
    xmax=time[stat_peaks] + 0.15*window_len/F_sim,
    zorder=100,
    lw=0.8,
    ls="dotted",
)

(l4,) = ax1.plot(time, (lhs_filt), c="C1")

for t, p, b in zip(time[dyn_peaks], lhs_filt[dyn_peaks], lhs_filt[dyn_peaks]-dyn_prominences):
    ax1.annotate(
        text="",
        xy=(t, p),
        xytext=(t, b),
        zorder=200,
        arrowprops=dict(ec="C1", fc="C1", arrowstyle="<|-|>", shrinkA=0, shrinkB=0),
    )

ax1.hlines(
    y=lhs_filt[dyn_peaks]-dyn_prominences,
    xmin=time[dyn_peaks] - 0.5*window_len/F_sim,
    xmax=time[dyn_peaks] + 0.05*window_len/F_sim,
    zorder=100,
    lw=0.8,
    ls="dotted",
    color="C1",
)
ax1.hlines(
    y=lhs_filt[dyn_peaks],
    xmin=time[dyn_peaks] - 0.15*window_len/F_sim,
    xmax=time[dyn_peaks] + 0.15*window_len/F_sim,
    zorder=100,
    lw=0.8,
    ls="dotted",
    color="C1",
)
# OTHER BITS

ax1.set_xlabel("Time / s")
ax1.set_ylabel(r"Amplitude / $g\times10^{-3}$")

# ax1.set_ylim([-0.75e-4, 2e-4])

plt.tight_layout()

ax1.legend([l2, l4], ["Static Model", "Denoise Filter, $a_f$"], loc="best")

ajf_plts.save_fig(fig)

# get max values and errors
with open("output/tab01.txt", "w") as f:
    f.write("Static prominences\n")
    f.write(str(stat_prominences))
    f.write("\nFiltered prominences\n")
    f.write(str(dyn_prominences))
    f.write("\nPercent Error\n")
    f.write(str(100 * (dyn_prominences - stat_prominences) / stat_prominences))
    f.write("\n")
