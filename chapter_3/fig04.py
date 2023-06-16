import numpy as np

import matplotlib.pyplot as plt

import ajf_plts

import BeamModel as bm

from Simulation import perform_dynamic_sim

beam = bm.Beam()

beam.E = bm.get_E_val_from_temp(25)

P_empty = np.array([61803, 50031, 17658, 23544, 21582])
S_empty = np.array([0, 3.8000, 9.1500, 10.3500, 11.6000])

P_full = np.array([68670, 107910, 71613, 77499, 73575])
S_full = np.array([0, 3.9000, 9.7500, 11.0500, 12.3500])

vel = 15  # metre per second

# Set Figure and Axes
plt.style.use(["./ajf_plts/base.mplstyle",
               "./ajf_plts/legend_frame.mplstyle"])

figsize = (ajf_plts.text_width_inches, ajf_plts.fig_height_inches)
fig, axes = plt.subplots(
    figsize=figsize,
    ncols=3,
    nrows=1,
    sharex=True,
    sharey=True,
    constrained_layout=False,
)
ax1, ax2, ax3 = axes.ravel()

# Plot Fig 4a

t_empty, disp_empty = perform_dynamic_sim(beam.Kg, P_empty, S_empty, vel)[:2]
t_full, disp_full = perform_dynamic_sim(beam.Kg, P_full, S_full, vel)[:2]

for time, disp, l in zip(
    [t_empty, t_full], [disp_empty, disp_full], ["Unladen", "Laden"]
):

    lhs = disp[bm.LHS_ROT_IDX, :] * 1e3  # convert to mrad

    ax1.plot(time, lhs, label=l)

# Plot Fig 4b

for v in [15, 20, 25]:
    time, disp = perform_dynamic_sim(beam.Kg, P_full, S_full, v)[:2]

    lhs = disp[bm.LHS_ROT_IDX, :] * 1e3  # convert to mrad

    ax2.plot(
        time,
        lhs,
        label=str(v) + r" \unit{\meter\per\second}",
    )


# Plot Fig 4c

for temp in [0, 25]:
    beam.E = bm.get_E_val_from_temp(temp)

    time, disp = perform_dynamic_sim(beam.Kg, P_full, S_full, 15)[:2]

    lhs = disp[bm.LHS_ROT_IDX, :] * 1e3  # convert to mrad

    ax3.plot(time, lhs, label=str(temp) + r"\unit{\degreeCelsius}")

ax1.set_ylabel(r"Rotation / \unit{\milli\radian}")

for ax in axes:
    ax.set_xlabel(r"Time / \unit{\second}")

ajf_plts.caption_axes(axes)

plt.tight_layout()

for ax in axes:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.48))

ajf_plts.save_fig(fig)
