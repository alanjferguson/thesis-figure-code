#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import ajf_plts

plt.style.use(["./ajf_plts/base.mplstyle",
               "./ajf_plts/lines_markers.mplstyle"])


def scatter_hist(x, y, label, ax, ax_hist):
    # no labels
    ax_hist.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.plot(x, y*1e6, ls="", label=label)
    ax.set_xlabel('Year')
    ax.set_xlim([1965, 2025])

    # histogram
    ax_hist.hist(y*1e6, 
                 label=label,
                 alpha=0.5,
                 orientation='horizontal',
                 density=False)

    ax.ticklabel_format(axis='x', style='plain')

    ax_hist.set_xlabel('Count')
    ax_hist.set_xlim([0,6])


fig = plt.figure(figsize=(ajf_plts.text_width_inches, 4.0*ajf_plts.fig_height_inches))

gs = gridspec.GridSpec(ncols=5,
                       nrows=3,
                       left=0.07,
                       right=0.98,
                       top=0.95,
                       bottom=0.1,
                       wspace=0.1,
                       hspace=0.35,
                       width_ratios=[0.36, 0.12, 0.09, 0.36, 0.12])

ax_scatter1 = plt.subplot(gs[0, 0])
ax_hist1 = plt.subplot(gs[0, 1],
                       sharey=ax_scatter1)

ax_scatter2 = plt.subplot(gs[0,3],
                       sharex=ax_scatter1,
                       sharey=ax_scatter1)
ax_hist2 = plt.subplot(gs[0, 4],
                       sharex=ax_hist1,
                       sharey=ax_scatter1)

ax_scatter3 = plt.subplot(gs[1,0])
ax_hist3 = plt.subplot(gs[1, 1],
                       sharey=ax_scatter3)

ax_scatter4 = plt.subplot(gs[1,3],
                       sharex=ax_scatter3,
                       sharey=ax_scatter3)
ax_hist4 = plt.subplot(gs[1, 4],
                       sharex=ax_hist3,
                       sharey=ax_scatter3)

ax_scatter5 = plt.subplot(gs[2,0])
ax_hist5 = plt.subplot(gs[2, 1],
                       sharey=ax_scatter5)

ax_scatter6 = plt.subplot(gs[2,3],
                       sharex=ax_scatter5,
                       sharey=ax_scatter5)
ax_hist6 = plt.subplot(gs[2, 4],
                       sharex=ax_hist5,
                       sharey=ax_scatter5)


def read_emd_file(file):
    df = pd.read_feather(file)
    df['year'] = df.index.values + 1969
    return df

# use previously defined funciton

############################################################
# ALL VEHICLES
############################################################


df = read_emd_file('./emds_all.feather')

for name, group in df.groupby('delta'):
    scatter_hist(group.year.values,
                 group.lvl.values,
                 name,
                 ax_scatter1,
                 ax_hist1)
    scatter_hist(group.year.values,
                 group.rvr.values,
                 name,
                 ax_scatter2,
                 ax_hist2)

############################################################
# 5 axle VEHICLES
############################################################

df = read_emd_file('./emds_5.feather')

for name, group in df.groupby('delta'):
    scatter_hist(group.year.values,
                 group.lvl.values,
                 name,
                 ax_scatter3,
                 ax_hist3)
    scatter_hist(group.year.values,
                 group.rvr.values,
                 name,
                 ax_scatter4,
                 ax_hist4)

############################################################
# 6 axle VEHICLES
############################################################
df = read_emd_file('./emds_6.feather')

for name, group in df.groupby('delta'):
    scatter_hist(group.year.values, 
                 group.lvl.values,
                 name,
                 ax_scatter5,
                 ax_hist5)
    scatter_hist(group.year.values, 
                 group.rvr.values,
                 name,
                 ax_scatter6,
                 ax_hist6)

for ax in [ax_scatter1, ax_scatter3, ax_scatter5]:
    ax.set_ylabel(r'$\mathrm{DI_{LL}}$ / \unit{\micro\radian}')

for ax in [ax_scatter2, ax_scatter4, ax_scatter6]:
    ax.set_ylabel(r'$\mathrm{DI_{RR}}$ / \unit{\micro\radian}')


for ax in [ax_scatter1, ax_scatter2, ax_scatter3, ax_scatter4, ax_scatter5, ax_scatter6]:
    ax.legend(loc='upper left', title=r'$\delta$', markerscale=1.5)

ajf_plts.caption_axes([ax_scatter1,
                       ax_scatter2,
                       ax_scatter3,
                       ax_scatter4,
                       ax_scatter5,
                       ax_scatter6])

ajf_plts.save_fig(fig)
