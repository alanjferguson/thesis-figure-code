import numpy as np

import ajf_plts

import matplotlib.pyplot as plt

plt.style.use(["./ajf_plts/base.mplstyle",
               "./ajf_plts/legend_frame.mplstyle"])


figsize = (ajf_plts.text_width_inches, ajf_plts.fig_height_inches)
fig, ax1 = plt.subplots(figsize=figsize, ncols=1, nrows=1)

baseline_vals = np.array([0.0, 0.0])
baseline_years = np.array([0, 0.9])

healthy_vals = np.array([0.0, 0.0])
healthy_years = np.array([1.5, 11])

delta_vals = np.array(
    [0.01, 0.01, np.NaN, 0.025, 0.025, np.NaN, 0.05, 0.05, np.NaN, 0.1, 0.1]
)
delta_periods = np.array([11, 10, 0, 0, 10, 0, 0, 10, 0, 0, 10]).cumsum()

(h1,) = ax1.plot(baseline_years, baseline_vals, lw=3, ls="-", c="C0")
(h2,) = ax1.plot(healthy_years, healthy_vals, lw=3, ls="--", c="C2")
(h3,) = ax1.plot(delta_periods, delta_vals, lw=3, ls=":", c="C1")

ax1.minorticks_off()

ax1.set_xticks([0, 1, 11, 21, 31, 41, 51])
ax1.set_yticks([0.0, 0.01, 0.025, 0.05, 0.1])

ax1.set_xlabel("Time / years")
ax1.set_ylabel(r"$\delta$")

ax1.grid()

ax1.legend(
    [h1, h2, h3],
    [r"Baseline (1 year)", r"Healthy (10 years)", r"Damaged (10 years ea.)"],
    loc="upper left",
    bbox_to_anchor=(0.075, 0.98)
)

fig.tight_layout()
ajf_plts.save_fig(fig)
