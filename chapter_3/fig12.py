import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import ajf_plts

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

WEIGHT_LABELS = ["W1", "W2", "W3", "W4", "W5", "W6"]
SPACING_LABELS = ["S0", "S1", "S2", "S3", "S4", "S5"]

# read in wim data
df = pd.read_feather("./wim_data.feather")

# take sample of 50k vehicles otherwise this is far too slow
df = df.groupby("Axles").sample(10000, replace=True)

# group by axle count
groups = df.sort_values("Axles", ascending=False).groupby("Axles", sort=False)

# setup axes and figure
plt.style.use(["./ajf_plts/base.mplstyle"])


figsize = (ajf_plts.text_width_inches, 2.0 * ajf_plts.fig_height_inches)
fig, axes = plt.subplots(figsize=figsize,
                         ncols=1,
                         nrows=2)
ax1, ax2 = axes.ravel()

for name, group in groups:
    group = group.loc[(group[WEIGHT_LABELS] < 150e3).all(axis=1)]
    group = group.loc[(group[SPACING_LABELS].fillna(0.0) < 16).all(axis=1)]

    weights = group[WEIGHT_LABELS].stack(dropna=False).values / 1e3
    weights[weights == 0.0] = np.NaN
    weights = weights[~np.isnan(weights)]
    spacings = group[SPACING_LABELS].stack(dropna=False).values
    spacings = spacings[~np.isnan(spacings)]

    # Dither locations apart from 0 to help plot clarity
    mask = spacings > 0.0
    spacings[mask] += rng.normal(size=len(spacings))[mask] * 0.025

    ax1.scatter(
        spacings,
        weights,
        ec='none',
        s=0.5,
        alpha=0.2,
        label=name,
        rasterized=True,
    )

    n_bins = 130
    counts, bin_edges = np.histogram(
        spacings, weights=weights, bins=n_bins, density=True
    )
    cdf = np.cumsum(counts)
    cdf /= cdf[-1]
    cdf = np.concatenate(([cdf[0]], cdf))

    bin_edges[0] = 0.0

    ax2.plot(bin_edges, cdf, label=name)

ax1.set_xlabel("Axle Location / m")
ax2.set_xlabel("Axle Location / m")

ax1.set_ylabel(r"Axle Weight / \unit{\kilo\newton}")
ax2.set_ylabel("Load Proportion")

ax1.set_xlim([-0.5, 14])
ax2.set_xlim([-0.5, 14])
ax2.set_ylim([-0.05, 1.05])

ajf_plts.caption_axes(axes.ravel())
plt.tight_layout()

for ax in [ax1, ax2]:
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        reversed(handles),
        reversed(labels),
        title="Axles",
        loc="center left",
        bbox_to_anchor=(1.025, 0.5)
    )

for h in ax1.get_legend().legendHandles:
    h._sizes = [30.0]
    h.set_alpha(1)

fig.tight_layout()
ajf_plts.save_fig(fig)
