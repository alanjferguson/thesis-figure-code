import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import ajf_plts

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

# read in temperature data
df = pd.read_feather("./data/CEDA_temp_data.feather")

# convert to 0 based years and months
df.year -= df.year.min()
df.month -= df.month.min()

# get sequentially numbered months
df["seq_month"] = df.year * 12 + df.month

# setup axes and figure
plt.style.use(["./ajf_plts/base.mplstyle"])

figsize = (ajf_plts.text_width_inches, 3.5*ajf_plts.fig_height_inches)
fig, axes = plt.subplots(figsize=figsize, nrows=3, ncols=3, sharey=True)

PERIODS = [6, 12, 24]
SAMPLE_SIZES = [100, 1000, 10000]
N_SAMPLES = 4

for (row, col), ax in np.ndenumerate(axes):
    N = SAMPLE_SIZES[col]
    M = PERIODS[row]

    if M > 6:
        axins = ax.inset_axes([0.6, 0.05, 0.35, 0.4])

    for i in range(N_SAMPLES):
        offset = rng.integers(df.seq_month.min(), df.seq_month.max() - M)
        months = offset + np.arange(M)
        temps = df.temp.loc[df.seq_month.isin(
            months)].sample(N, replace=True, random_state=rng).values

        n_bins = 180

        counts, bin_edges = np.histogram(temps, bins=n_bins, density=True)
        cdf = np.cumsum(counts)
        cdf /= cdf[-1]
        cdf = np.concatenate(([cdf[0]], cdf))

        ax.plot(bin_edges, cdf, label="$S_{" + str(i + 1) + "}$", lw=0.9)

        if M > 6:
            axins.plot(
                bin_edges, cdf, label="$S_{" + str(i + 1) + "}$", lw=0.9)

    ax.set_xlim([-8, 28])
    ax.set_ylim([-0.05, 1.05])

    if M > 6:
        x_lims = [12, 18]
        y_lims = [0.8, 0.95]

        axins.set_xlim(x_lims)
        axins.set_ylim(y_lims)
        axins.set_xlabel("")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.grid(False)

        ax.indicate_inset_zoom(axins, label=None)

    ax.set_xlabel(r"Temperature / \unit{\degreeCelsius}")
    if col == 0:
        ax.set_ylabel("Cumulative Probability")
    ax.set_title(rf"$M=\num{{{M:d}}}, N=\num{{{N:d}}}$")

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
