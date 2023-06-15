import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import string

import os
import sys
import inspect

from cycler import cycler

from numpy.random import Generator, PCG64

def get_seed(seed_string):
    """
    + Returns a seed number formed by concatenating the ASCII values of
      the passed in string.
    + Intended to be used by passing in __file__ so each plot gets a different
      different RNG that is reproducible

    Parameters
    ----------
    seed_string : str
        String to create RNG seed from.

    Returns
    -------
    seed_num
        Seed number.
    """
    return int("".join([str(ord(c)) for c in seed_string])) % (2**32 - 1)

def get_rng(seed_string):
    """
    Returns a seeded random number generator.
    + Seeds RNG with the number formed by concatenating the ASCII values of
      the passed in string.
    + Intended to be used by passing in __file__ so each plot gets a different
      different RNG that is reproducible

    Parameters
    ----------
    seed_string : str
        String to create RNG seed from.

    Returns
    -------
    rng
        numpy random number generator object.
    """
    return Generator(PCG64(get_seed(seed_string)))

def get_random_state(seed_string):
    """
    Returns a seeded random state.
    + Seeds random state with the number formed by concatenating the ASCII values of
      the passed in string.
    + Intended to be used by passing in __file__ so each plot gets a different
      different random state that is reproducible

    Parameters
    ----------
    seed_string : str
        String to create RNG seed from.

    Returns
    -------
    random_state 
        numpy random state object.
    """
    return np.random.RandomState(seed=get_seed(seed_string))

# Update these values if Thesis LaTeX margins / stock size changes
text_height_inches = 597.5 / 72  # converts points to inches
text_width_inches = 418.25 / 72  # converts points to inches

# The ratio is  taken from SciencePlots IEEE mplstyle (3.3" wide by 2.5" high)
# Assumes target figure width is half the text width
# fig_height_inches = (text_width_inches / 2.0) * (2.5 / 3.3)
fig_height_inches = (text_width_inches / 2.0) * (4.0 / 5.0)

single_wide_figsize = (text_width_inches / 2.0, fig_height_inches)
double_wide_figsize = (text_width_inches, fig_height_inches)


def caption_axes(axes, start_val=0):
    for n, ax in enumerate(axes, start=start_val):
        orig_x_label = ax.get_xlabel()
        # handle case where axis doesn't have an xlabel
        if len(orig_x_label):
            orig_x_label += r"\\"
        ax.set_xlabel(
            r"\begin{center}"
            + orig_x_label
            + r"\bf{("
            + string.ascii_lowercase[n]
            + r")}\end{center}"
        )


def save_fig(fig, out_filename=None):
    if out_filename is None:
        caller_path = inspect.getfile(sys._getframe(1))
        caller_base = os.path.splitext(os.path.basename(caller_path))[0]
        out_filename = f"{caller_base}.pdf"
    fig.savefig(f"./output/{out_filename}", dpi=1200)


def annotate_train_test_split(ax,
                              split_x_val,
                              label_x_offset,
                              label_on=True,
                              ls="--",
                              c='k',
                              lw='2.0',
                              fontsize=14.0,
                              fontweight='bold'
                              ):
    ax.axvline(split_x_val, ls=ls, c=c, lw=lw)
    if label_on:
        annot_y_val =  ax.transData.inverted().transform( (0.0, ax.transAxes.transform((0.0, 0.01))[1]) )[1]
        ax.annotate(
            "TRAINING",
            xy=(split_x_val - label_x_offset, annot_y_val),
            ha="right",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
        )
        ax.annotate(
            "TESTING",
            xy=(split_x_val + label_x_offset, annot_y_val),
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
        )

def caption_subplots(axes,
                 start_val=0,
                 h_pos=0.5,
                 v_pos=-0.2,
                 ha='center',
                 va='top',
                 label_format=r"$\mathbf{{({})}}$",
                 labels=string.ascii_lowercase,
                 kwargs=dict()):
    flat_axes = (axes.ravel() if isinstance(axes, np.ndarray) else axes)
    for n, (ax, l) in enumerate(zip(flat_axes, labels[start_val:]), start=start_val):
        ax.text(h_pos, v_pos,
                str.format(label_format, l),
                transform=ax.transAxes,
                va=va,
                ha=ha,
                **kwargs)