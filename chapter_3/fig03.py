import numpy as np

import ajf_plts

import matplotlib.pyplot as plt

import BeamModel as bm

from Simulation import perform_dynamic_sim

beam = bm.Beam()

P_ax = np.array([300 * 1000])  # Newtons
S_ax = np.array([0])  # metres lag
vel = 10  # metres per second

deltas = [0.0, 0.1, 0.2]

x_d = bm.BEAM_LENGTH / 3.0

# plot data
plt.style.use(["./ajf_plts/base.mplstyle"])

fig, axes = plt.subplots(
    figsize=ajf_plts.double_wide_figsize,
    ncols=2,
    nrows=1,
    sharex=True,
    sharey=False
)
ax1, ax2 = axes.ravel()

for d in deltas:
    beam.inflict_damage_at_x(x_d, d)

    time, disp, _, _ = perform_dynamic_sim(beam.Kg, P_ax, S_ax, vel)

    msd = disp[bm.MID_DIS_IDX, :]
    lhs = disp[bm.LHS_ROT_IDX, :]

    ax1.plot(time, msd*1e3, label=rf"$\delta={str(d)}$")
    ax2.plot(time, lhs*1e3, label=rf"$\delta={str(d)}$")

ax1.set_ylabel(r"Displacement / \unit{\milli\meter}")
ax2.set_ylabel(r"Rotation / \unit{\milli\radian}")

for ax in axes.ravel():
    ax.set_xlabel(r"Time / \unit{\second}")
    ax.legend(loc="best")

ajf_plts.caption_axes(axes)

fig.tight_layout()
ajf_plts.save_fig(fig)
