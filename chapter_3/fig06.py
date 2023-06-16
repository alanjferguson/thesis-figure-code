from Simulation import perform_dynamic_sim
import BeamModel as bm

import numpy as np

import matplotlib.pyplot as plt

import pickle

import ajf_plts

from scipy import signal

from scipy.fftpack import fft, fftfreq

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

# Setup Figures early
plt.style.use(["./ajf_plts/base.mplstyle",
               "./ajf_plts/legend_frame.mplstyle"])

figsize = (ajf_plts.text_width_inches, 3.0 * ajf_plts.fig_height_inches)
fig, axes = plt.subplots(figsize=figsize, ncols=2, nrows=3)
axes = axes.ravel()

###############################################################################
# Noise Data
###############################################################################
with open("noise_data.pkl", "rb") as f:
    t_vals, z_noise = pickle.load(f)

T_sample = np.mean(np.diff(t_vals))
f_sample = 1 / T_sample
N_samples = len(t_vals)

fe_T = 0.001
fe_F = 1 / fe_T

t_vals = signal.decimate(t_vals, int(f_sample / fe_F))
z_noise = signal.decimate(z_noise, int(f_sample / fe_F))
z_noise *= 1e3  # convert to gx10^-3

T_sample = np.mean(np.diff(t_vals))
f_sample = 1 / T_sample
N_samples = len(t_vals)

start_idx = 100000
plot_len = int(3 * f_sample)
end_idx = start_idx + plot_len

axes[0].plot(t_vals[:plot_len] - t_vals[0], z_noise[start_idx:end_idx])

axes[0].set_xlabel(r"Time / \unit{\second}")
axes[0].set_ylabel(r"Amplitude / $g\times10^{-3}$")

z_fft = fft(z_noise)
x_fft = fftfreq(N_samples, T_sample)[: N_samples // 2]

axes[1].loglog(x_fft, 2 / N_samples * np.abs(z_fft)[0 : N_samples // 2])
axes[1].set_xlabel(r"Frequency / \unit{\hertz}")
axes[1].set_ylabel(r"Amplitude / $g\times10^{-3}$")


def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = rng.random(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


###############################################################################
# Simulated Noise
###############################################################################

sim_noise = fftnoise(z_fft)

axes[2].plot(t_vals[:plot_len] - t_vals[0], sim_noise[start_idx:end_idx], c="C1")
axes[2].set_xlabel(r"Time / \unit{\second}")
axes[2].set_ylabel(r"Amplitude / $g\times10^{-3}$")

sim_fft = fft(sim_noise)
axes[3].loglog(x_fft, 2 / N_samples * np.abs(sim_fft)[0 : N_samples // 2], c="C1")
axes[3].set_xlabel(r"Frequency / \unit{\hertz}")
axes[3].set_ylabel(r"Amplitude / $g\times10^{-3}$")


################################################################################
# Noisy Signal
################################################################################

beam = bm.Beam()

beam.E = bm.get_E_val_from_temp(25)

P_full = np.array([68670, 107910, 71613, 77499, 73575])
S_full = np.array([0, 3.9000, 9.7500, 11.0500, 12.3500])
vel = 15  # metre per second

time, disp = perform_dynamic_sim(beam.Kg, P_full, S_full, vel, time_step=0.001)[:2]

lhs_fe = disp[bm.LHS_ROT_IDX] * 1e3  # convert to gx10^-3
pad_len = 5000
lhs_fe = np.pad(lhs_fe, pad_width=pad_len)
lhs_noisy = sim_noise[: len(lhs_fe)]
lhs_noisy += lhs_fe
time = np.arange(len(lhs_noisy)) / 1000

ax = plt.subplot(3, 2, (5, 6))
ax.plot(
    time[: -2 * pad_len],
    lhs_fe[pad_len:-pad_len] + np.mean(lhs_noisy),
    ls="dashed",
    c="k",
    linewidth=2,
    label="FE Model",
    zorder=10,
)
ax.plot(
    time[: -2 * pad_len],
    lhs_noisy[pad_len:-pad_len],
    ls="solid",
    c="C3",
    label="FE+Sensor Model",
)
ax.set_xlabel(r"Time / \unit{\second}")
ax.set_ylabel(r"Amplitude / $g\times10^{-3}$")
ax.legend(loc="lower center")

ax.get_shared_y_axes().join(axes[0], ax)
ax.get_shared_y_axes().join(axes[2], ax)

ajf_plts.caption_axes(axes[[0, 1, 2, 3]])
ajf_plts.caption_axes([ax], start_val=4)

# fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=2.0)
fig.tight_layout()
ajf_plts.save_fig(fig)

with open("noise_fft.pkl", "wb") as f:
    pickle.dump(z_fft, f)

with open("noisy_sig.pkl", "wb") as f:
    pickle.dump([time, lhs_noisy], f)
