import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from scipy.stats import wasserstein_distance

import os

import ajf_plts

plt.style.use(["./ajf_plts/base.mplstyle",
               "./ajf_plts/legend_frame.mplstyle"])

mpl.rcParams['lines.linewidth'] = 0.4

res_dir = './51y100kv3/'


def get_l_r_cdfs(df):
    min_val = 0.5e-1
    step_val = 1e-2
    max_val = 4.0e-1

    new_index = pd.Index(np.arange(min_val, max_val, step_val),
                         name='vals')

    cdf_l = df.groupby('max_l')['max_l'] \
              .agg('count').cumsum() \
              .reindex(new_index, method='pad', tolerance=step_val)
    cdf_l[min_val] = 0
    cdf_l = cdf_l.interpolate(method='pad')/cdf_l.max()

    cdf_r = df.groupby('max_r')['max_r'] \
        .agg('count').cumsum() \
        .reindex(new_index, method='pad', tolerance=step_val)
    cdf_r[min_val] = 0
    cdf_r = cdf_r.interpolate(method='pad')/cdf_r.max()

    return cdf_l, cdf_r


def get_emd(x, y):
    return wasserstein_distance(x.index.values,
                                y.index.values,
                                x.values,
                                y.values)


# Set Figure and Axes
figsize = (ajf_plts.text_width_inches, 1.2 * ajf_plts.fig_height_inches)
fig, axes = plt.subplots(figsize=figsize,
                         ncols=2,
                         nrows=1,
                         sharex=True,
                         sharey=True)
ax1, ax2 = axes.ravel()

# plot baseline year CDFs for LHS
delta_cols = {0.0: 'C0',  # blue
              0.01: 'C1',  # green
              0.025: 'C2',  # yellow
              0.05: 'C3',  # red
              0.1: 'C4'}  # violet

BASELINE_LEN = 12
AXLES = [2, 3, 4, 5, 6]

baseline_months = np.arange(BASELINE_LEN)
baseline_df = []
for m in baseline_months:
    baseline_df.append(pd.read_feather(os.path.join(
        res_dir, 'res_month_'+str(m)+'.feather')))
baseline_df = pd.concat(baseline_df)
baseline_df = baseline_df.loc[baseline_df.Axles.isin(AXLES), :]
baseline_df.max_l *= 1e3  # convert to mrad
baseline_df.max_r *= 1e3  # convert to mrad
baseline_l, baseline_r = get_l_r_cdfs(baseline_df)

axins1 = ax1.inset_axes([0.45, 0.05, 0.5, 0.6])
axins2 = ax2.inset_axes([0.45, 0.05, 0.5, 0.6])

for y in range(1, 51):
    months = (y*12) + np.arange(12)
    df = []
    for m in months:
        df.append(pd.read_feather(os.path.join(
            res_dir, 'res_month_'+str(m)+'.feather')))
    df = pd.concat(df)
    df.max_l *= 1e3  # convert to mrad
    df.max_r *= 1e3  # convert to mrad
    df = df.loc[df.Axles.isin(AXLES), :]
    delta = df.delta.max()

    l, r = get_l_r_cdfs(df)

    l.plot(ax=ax1,
           c=delta_cols[delta], ls='-', zorder=-1*y)
    l.plot(ax=axins1,
           c=delta_cols[delta], ls='-', zorder=-1*y)
    r.plot(ax=ax2,
           c=delta_cols[delta], ls='-', zorder=-1*y)
    r.plot(ax=axins2,
           c=delta_cols[delta], ls='-', zorder=-1*y)

baseline_l.plot(lw=1.0, ls='--',
                c='k',
                ax=ax1)
baseline_l.plot(lw=1.2, ls='--',
                c='k',
                ax=axins1)

baseline_r.plot(lw=1.0, ls='--',
                c='k',
                ax=ax2)
baseline_r.plot(lw=1.2, ls='--',
                c='k',
                ax=axins2)

ax1.set_ylabel('Cumulative Probability')

for ax in [ax1, ax2]:
    ax.set_xlabel(r'Rotation / \unit{\milli\radian}')


x_lims = [2.25e-1, 2.40e-1]
y_lims = [0.9, 0.91]

for ax in [axins1, axins2]:
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.grid(False)

ax1.indicate_inset_zoom(axins1)
ax2.indicate_inset_zoom(axins2)

ajf_plts.caption_axes([ax1, ax2])

custom_lines = [Line2D([0], [0], lw=1, color='k', ls='--'),
                Line2D([0], [0], lw=1, color=delta_cols[0.0]),
                Line2D([0], [0], lw=1, color=delta_cols[0.01]),
                Line2D([0], [0], lw=1, color=delta_cols[0.025]),
                Line2D([0], [0], lw=1, color=delta_cols[0.05]),
                Line2D([0], [0], lw=1, color=delta_cols[0.1])]
leg_labels = [r'Baseline ($\delta=0.0$)',
              r'$\delta=0.000$',
              r'$\delta=0.010$',
              r'$\delta=0.025$',
              r'$\delta=0.050$',
              r'$\delta=0.100$']

fig.tight_layout()

plt.legend(custom_lines,
           leg_labels,
           title='Condition Level',
           ncol=3,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.23),
           bbox_transform=fig.transFigure)

ajf_plts.save_fig(fig)
