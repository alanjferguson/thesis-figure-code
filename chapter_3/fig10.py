import pandas as pd
import numpy as np

import ajf_plts

import matplotlib.pyplot as plt

plt.style.use(["./ajf_plts/base.mplstyle",
               "./ajf_plts/lines_markers.mplstyle"])


df = pd.read_parquet("./data/fig10.parquet")

figsize = (ajf_plts.text_width_inches, 2.0*ajf_plts.fig_height_inches)
fig, axes = plt.subplots(figsize=figsize, ncols=2, nrows=2, sharey="row")
ax1, ax2, ax3, ax4 = axes.ravel()

ax3.get_shared_x_axes().join(ax3, ax4)

df.max_l *= 1e3  # convert to mrad

groups = df.groupby("delta")

bins = []
cdfs = []

bins_c = []
cdfs_c = []

sub_inds = np.arange(0, 200)
all_inds = np.arange(0, 2000)

# HACK to make hatching lines thinner
plt.rcParams['hatch.linewidth'] = 0.5
hatch_props = dict(facecolor="none", hatch="\\" * 8,
                   ec=(0, 0, 0, 0.8),
                   linewidth=0.0,
                   zorder=-100)

for name, group in groups:
    ax1.plot(sub_inds, group.max_l.iloc[:200],
             markersize=1, ls='', label=r"$\delta="+str(name)+"$")

    ax2.plot(all_inds, group.max_l[-2000:], markersize=1,
             ls='', label=r"$\delta="+str(name)+"$")

    sub_inds += 200
    all_inds += 2000

    n_bins = 50
    counts, bin_edges = np.histogram(
        group.max_l.iloc[:200], bins=n_bins, density=True)
    cdf = np.cumsum(counts)

    cdf /= cdf[-1]

    bins_c.append(bin_edges)
    cdfs_c.append(cdf)
    if len(cdfs_c) == 3:
        x1 = bins_c[0][1:]
        x2 = bins_c[2][1:]
        y1 = cdfs_c[0]
        y2 = cdfs_c[2]
        xfill = np.sort(np.concatenate([x1, x2]))
        y1fill = np.interp(xfill, x1, y1)
        y2fill = np.interp(xfill, x2, y2)
        ax3.fill_between(xfill, y1fill, y2fill, facecolor=(0, 0, 0, 0.1))
        x1 = bins_c[0][1:]
        x2 = bins_c[1][1:]
        y1 = cdfs_c[0]
        y2 = cdfs_c[1]
        xfill = np.sort(np.concatenate([x1, x2]))
        y1fill = np.interp(xfill, x1, y1)
        y2fill = np.interp(xfill, x2, y2)
        ax3.fill_between(
            xfill,
            y1fill,
            y2fill,
            **hatch_props
        )

    ax3.plot(bin_edges[1:], cdf, marker='')

    n_bins = 50
    counts, bin_edges = np.histogram(
        group.max_l[-2000:], bins=n_bins, density=True)
    cdf = np.cumsum(counts)
    cdf /= cdf[-1]

    ax4.plot(bin_edges[1:], cdf, marker='', label=r"$\delta="+str(name)+"$")

    bins.append(bin_edges)
    cdfs.append(cdf)
    if len(cdfs) == 3:
        x1 = bins[0][1:]
        x2 = bins[2][1:]
        y1 = cdfs[0]
        y2 = cdfs[2]
        xfill = np.sort(np.concatenate([x1, x2]))
        y1fill = np.interp(xfill, x1, y1)
        y2fill = np.interp(xfill, x2, y2)
        ax4.fill_between(
            xfill, y1fill, y2fill, interpolate=True, facecolor=(0, 0, 0, 0.1)
        )
        x1 = bins[0][1:]
        x2 = bins[1][1:]
        y1 = cdfs[0]
        y2 = cdfs[1]
        xfill = np.sort(np.concatenate([x1, x2]))
        y1fill = np.interp(xfill, x1, y1)
        y2fill = np.interp(xfill, x2, y2)
        ax4.fill_between(
            xfill,
            y1fill,
            y2fill,
            interpolate=True,
            **hatch_props)


for ax in [ax1, ax2]:
    ax.set_xlabel("Index")
    ax.ticklabel_format(axis='x', style='plain')

ax1.set_ylabel(r"Rotation / \unit{\milli\radian}")

ax2.tick_params(labelleft=False)
ax4.tick_params(labelleft=False)

ax3.set_ylabel("Cumulative Probability")

ax3.set_xlabel(r"Rotation / \unit{\milli\radian}")
ax4.set_xlabel(r"Rotation / \unit{\milli\radian}")

ajf_plts.caption_axes(axes.ravel())

leg_1 = ax2.legend(loc="center left",
                   bbox_to_anchor=(1.05, 0.5))
ax4.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

for lh in leg_1.legendHandles:
    lh._markersize = 4

fig.tight_layout()
ajf_plts.save_fig(fig)
