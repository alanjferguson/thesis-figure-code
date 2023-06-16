import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import ajf_plts

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

# read in wim data
df = pd.read_feather("./wim_data.feather")

# setup axes and figure
plt.style.use(["./ajf_plts/base.mplstyle"])

figsize = (ajf_plts.text_width_inches, 3.5*ajf_plts.fig_height_inches)
fig, axes = plt.subplots(figsize=figsize, nrows=3, ncols=3, sharey=True)


def get_weights_spacings(df):
    WEIGHT_LABELS = ["W1", "W2", "W3", "W4", "W5", "W6"]
    SPACING_LABELS = ["S0", "S1", "S2", "S3", "S4", "S5"]

    weights = df[WEIGHT_LABELS].stack(dropna=False).values / 1e3
    weights[weights == 0.0] = np.NaN
    weights = weights[~np.isnan(weights)]
    spacings = df[SPACING_LABELS].stack(dropna=False).values
    spacings = spacings[~np.isnan(spacings)]

    return weights, spacings


N_VEHICLES = [10, 100, 1000]

N_SAMPLES = 4

MIN_AXLES = [5, 6, 2]
MAX_AXLES = [5, 6, 6]

ROW_LABELS = ["$C = 5$", "$C = 6$", r"$2\leq C \leq 6$"]
COL_LABELS = ["$N={0:d}$".format(n) for n in N_VEHICLES]

for (row, col), ax in np.ndenumerate(axes):
    axins = ax.inset_axes([0.5, 0.05, 0.45, 0.4])

    N = N_VEHICLES[col]
    min_axles = MIN_AXLES[row]
    max_axles = MAX_AXLES[row]

    for i in range(N_SAMPLES):
        sample = df.loc[df.Axles.between(min_axles, max_axles)] \
                   .sample(n=N, random_state=rng)
        weights, spacings = get_weights_spacings(sample)

        n_bins = 180

        counts, bin_edges = np.histogram(
            spacings, weights=weights, bins=n_bins, density=True
        )
        cdf = np.cumsum(counts)
        cdf /= cdf[-1]
        cdf = np.concatenate(([cdf[0]], cdf))

        bin_edges[0] = 0.0

        ax.plot(bin_edges, cdf, label="$S_{" + str(i + 1) + "}$", lw=0.9)

        axins.plot(bin_edges, cdf, label="$S_{" + str(i + 1) + "}$", lw=0.9)

    ax.set_xlim([-0.5, 16])
    ax.set_ylim([-0.05, 1.05])

    x_lims = [9.5, 11]
    y_lims = [0.7, 0.78]

    axins.set_xlim(x_lims)
    axins.set_ylim(y_lims)
    axins.set_xlabel("")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(False)

    ax.indicate_inset_zoom(axins, label=None)

    ax.set_xlabel("Axle Position / m")

    if col == 0:
        ax.set_ylabel("Load Proportion")

    ax.set_title("{0:s}, {1:s}".format(ROW_LABELS[row], COL_LABELS[col]))


ajf_plts.caption_axes(axes.ravel())

plt.tight_layout()

for ax in axes[:, -1]:
    ax.legend(
        title="Sample",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
    )

fig.tight_layout()
ajf_plts.save_fig(fig)
