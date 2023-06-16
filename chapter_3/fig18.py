import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance

import os

import ajf_plts

plt.style.use(['./ajf_plts/base.mplstyle',
               './ajf_plts/markers.mplstyle',
               './ajf_plts/legend_frame.mplstyle'])

# Set Figure and Axes
fig, axes = plt.subplots(figsize=(ajf_plts.text_width_inches,
                                  1.2 * ajf_plts.fig_height_inches),
                         ncols=2, nrows=1,
                         sharex=True, sharey=True)


def read_emd_file(file):
    df = pd.read_feather(file)
    df['year'] = df.index.values + 1969
    return df


############################################################
# DAMAGE LOCATION FOR ALL Vehicles
############################################################
df = read_emd_file('./emds_all.feather')
df['x_dam'] = 0.5 + 0.75 * np.where(df.lvl.values >= df.rvr.values,
                                    np.cbrt(df.rvr.values / df.lvl.values) - 1,
                                    1-np.cbrt(df.lvl.values / df.rvr.values))

for name, group in df.groupby('delta'):
    axes[0].plot(group.year.values,
                 group.x_dam.values,
                 ls="",
                 label=name)

############################################################
# DAMAGE LOCATION FOR 6 AXLE Vehicles
############################################################
df = read_emd_file('./emds_6.feather')
df['x_dam'] = 0.5 + 0.75 * np.where(df.lvl.values >= df.rvr.values,
                                    np.cbrt(df.rvr.values / df.lvl.values) - 1,
                                    1-np.cbrt(df.lvl.values / df.rvr.values))

for name, group in df.groupby('delta'):
    axes[1].plot(group.year.values,
                 group.x_dam.values,
                 ls="",
                 label=name)

for ax in axes.ravel():
    ax.set_xlim([1965, 2025])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Year')
    ax.ticklabel_format(axis='x', style='plain')
    # annotate the true value of x_dam
    ax.axhline(y=0.333, ls='--', lw=1.25, zorder=-100, color='black')

axes[0].set_ylabel(r"$\frac{\hat{x}_{d}}{L}$")

axes[1].legend(loc="center left",
               bbox_to_anchor=(1.05, 0.5),
               title=r'$\delta$',
               markerscale=1.5)

ajf_plts.caption_axes(axes.ravel())

fig.tight_layout()

ajf_plts.save_fig(fig)
