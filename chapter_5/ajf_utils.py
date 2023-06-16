import ajf_plts
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import r2_score, explained_variance_score

AXLE_NUMS = [np.arange(2, 7), np.arange(5, 6), np.arange(6, 7)]
AXLE_NAMES = ["All", "5ax", "6ax"]


def plot_steps_stairs(emds, locs, years, delta_vals):
    plt.style.use(
        [
            "./ajf_plts/base.mplstyle",
            "./ajf_plts/legend_frame.mplstyle",
            "./ajf_plts/lines_markers.mplstyle",
        ]
    )

    year_delta_df = pd.DataFrame(
        years,
        index=delta_vals[
            (np.arange(1, years.max() + 1) - 1) // (years.max() // len(delta_vals))
        ],
        columns=["year"],
    )
    figsize = (ajf_plts.text_width_inches, len(AXLE_NUMS) * ajf_plts.fig_height_inches)
    fig, axes = plt.subplots(figsize=figsize, ncols=len(locs), nrows=len(AXLE_NUMS))
    for i, axles in enumerate(AXLE_NAMES):
        for j, pos in enumerate(locs):
            for delta, group in emds.groupby("delta"):
                axes[i, j].plot(
                    group.loc[delta].loc[
                        year_delta_df.loc[delta].values.flatten(), f"{pos}_{axles}"
                    ]
                    * 1e6,
                    ls="",
                    label=f'{delta:.3f}',
                )
    for a in axes.ravel():
        _ = a.set_xlabel("Year")
    for a in axes[:, 0]:
        _ = a.set_ylabel(r"$\mathrm{DI_{LL}}$ / \unit{\micro\radian}")
    for a in axes[:, -1]:
        _ = a.set_ylabel(r"$\mathrm{DI_{RR}}$ / \unit{\micro\radian}")
        _ = a.legend(
            loc="upper left",
            title="$\delta$",
            ncol=2,
            handletextpad=0.0,
            columnspacing=0.0,
        )
    for row in axes[0:, :]:
        for a in row:
            _ = a.set_ylim(row[0].get_ylim())
    ajf_plts.caption_axes(axes.ravel())
    fig.tight_layout(h_pad=1.0)
    return fig


def plot_scatter_all(emds, DAM_LOCS):
    plt.style.use(
        [
            "./ajf_plts/base.mplstyle",
            "./ajf_plts/legend_frame.mplstyle",
            "./ajf_plts/lines_markers.mplstyle",
        ]
    )

    figsize = (ajf_plts.text_width_inches, len(AXLE_NUMS) * ajf_plts.fig_height_inches)
    fig, axes = plt.subplots(figsize=figsize, ncols=len(DAM_LOCS), nrows=len(AXLE_NUMS))

    for i, axles in enumerate(AXLE_NAMES):
        for j, pos in enumerate(DAM_LOCS):
            for delta, group in emds.groupby("delta"):
                axes[i, j].plot(
                    group.loc[:, f"{pos}_{axles}"].values * 1e6, ls="", label=f'{delta:.3f}'
                )

    for a in axes.ravel():
        _ = a.set_xlabel("Year")

    for a in axes[:, 0]:
        _ = a.set_ylabel(r"$\mathrm{DI_{LL}}$ / \unit{\micro\radian}")

    for a in axes[:, -1]:
        _ = a.set_ylabel(r"$\mathrm{DI_{RR}}$ / \unit{\micro\radian}")
        _ = a.legend(
            loc="upper center",
            title="$\delta$",
            ncol=3,
            handletextpad=0.0,
            columnspacing=0.0,
        )

    for row in axes[0:, :]:
        for a in row:
            _ = a.set_ylim(row[0].get_ylim())

    ajf_plts.caption_axes(axes.ravel())
    fig.tight_layout(h_pad=1.0)
    return fig


# Uses Freedman-Diaconis rule to size bins of histogram
# Ref: D. Freedman, P. Diaconis, "On the histogram as a density estimator: L_2 theory", Probability Theory and Related Fields. 57 (4): 453–476, December 1981
def get_FD_n_bins(vals):
    h = 2.0 * stats.iqr(vals) * np.power(len(vals), -1.0 / 3.0)
    return int(np.ptp(vals) // h)


# Uses Freedman-Diaconis rule to size bins of histogram
# Ref: D. Freedman, P. Diaconis, "On the histogram as a density estimator: L_2 theory", Probability Theory and Related Fields. 57 (4): 453–476, December 1981
def get_rv_hist(vals):
    h = 2.0 * stats.iqr(vals) * np.power(len(vals), -1.0 / 3.0)
    n_bins = int(np.ptp(vals) // h)
    return stats.rv_histogram(np.histogram(vals, bins=n_bins))


def get_pi_via_kde(data, ALPHA=0.05):
    kde = sm.nonparametric.KDEUnivariate(data).fit(kernel="gau")
    cdf = np.linspace(0, 1, len(kde.density))
    return kde.icdf[
        [
            (np.abs(cdf - ALPHA / 2.0)).argmin(),
            (np.abs(cdf - (1.0 - ALPHA / 2.0))).argmin(),
        ]
    ]


from sklearn.preprocessing import PolynomialFeatures


def plot_pi(emds, DAM_LOCS, CONF=0.95, PI_LWR_FIT_ORD=None):
    plt.style.use(
        [
            "./ajf_plts/base.mplstyle",
            "./ajf_plts/legend_frame.mplstyle",
            "./ajf_plts/lines_markers.mplstyle",
        ]
    )

    delta_vals = emds.delta.unique()

    alpha = 1.0 - CONF

    figsize = (ajf_plts.text_width_inches, len(AXLE_NUMS) * ajf_plts.fig_height_inches)
    fig, axes = plt.subplots(figsize=figsize, ncols=len(DAM_LOCS), nrows=len(AXLE_NUMS))

    for i, axles in enumerate(AXLE_NAMES):
        for j, pos in enumerate(DAM_LOCS):
            lv = f"{pos}_{axles}"
            x_pos = 0
            deltas = []
            pi_uppers = []
            pi_lowers = []
            for delta, group in emds.groupby("delta"):
                x_pos += 1
                x_pos = delta
                pi = get_pi_via_kde(group.loc[:, lv])
                if delta == 0.0:
                    axes[i, j].axhline(pi[1], ls="--", c="k")
                    pi_base = pi
                med = np.median(group.loc[:, lv])
                axes[i, j].plot(
                    x_pos,
                    med,
                    c="r" if pi_base[1] <= pi[0] else "k",
                    ls="",
                    marker="o",
                    label="_Median",
                )
                axes[i, j].vlines(
                    x_pos,
                    pi[0],
                    pi[1],
                    colors="r" if pi_base[1] <= pi[0] else "k",
                    label="_95\%PI",
                )
                if pi_base[1] < pi[0]:
                    deltas.append(delta)
                    pi_lowers.append(pi[0])
                    pi_uppers.append(pi[1])
            if PI_LWR_FIT_ORD:
                ### fit to lower bound of PIs
                x = np.array(deltas)
                y = np.array(pi_lowers)
                coeff = np.polyfit(x, y, PI_LWR_FIT_ORD)
                f = np.poly1d(coeff)
                xn = np.linspace(delta_vals.min(), delta_vals.max(), 1000)
                yn = f(xn)
                axes[i,j].plot(xn, yn, c='r', ls=':', marker='')
                min_d = x.min()
                min_est = xn[np.argmin(np.abs(yn-pi_base[1]))]
                print(f'{lv}: adj.R2:{1.0-(1.0-r2_score(y,f(x)))*(len(x)-1)/(len(x)-PI_LWR_FIT_ORD):.3f} exp.var.:{explained_variance_score(y,f(x)):.3f} min d:{min_d-0.02:.3f} min est:{min_est:.3f}')

    for a in axes.ravel():
        a.set_xlabel("Damage Severity, $\delta$")
        # a.set_xticks(np.arange(1, len(delta_vals) + 1))
        # a.set_xticklabels(delta_vals)
        # a.tick_params(axis="x", which="minor", bottom=False, top=False)

    for a in axes[:, 0]:
        _ = a.set_ylabel(r"$\mathrm{DI_{LL}}$ / \unit{\micro\radian}")

    for a in axes[:, -1]:
        _ = a.set_ylabel(r"$\mathrm{DI_{RR}}$ / \unit{\micro\radian}")

    handles = []
    # Median
    handles.append(mpl.lines.Line2D([], [], color="k", marker="o", linestyle="None"))
    handles.append(
        mpl.lines.Line2D(
            [],
            [],
            color="k",
            marker="|",
            linestyle="None",
            markersize=10,
            markeredgewidth=1.5,
            label="_95\%PI",
        )
    )
    handles.append(mpl.lines.Line2D([], [], color="r", marker="s", linestyle="None", markersize=5.0))
    for a in axes.ravel():
        a.legend(
            loc="upper left",
            handles=handles,
            labels=["Median", "95\% PI", 'Disjoint'],
            handletextpad=0.2,
        )

    for row in axes[0:, :]:
        ylim = np.array(row[0].get_ylim())
        ylim[0] = -0.05 * ylim[1]
        for a in row:
            _ = a.set_ylim(ylim)

    ajf_plts.caption_axes(axes.ravel())
    fig.tight_layout(h_pad=1.0)
    return fig


def format_pred_int_table(emds, DAM_LOCS, CONF=0.95):
    delta_vals = emds.reset_index().delta.unique()
    loc_veh_pairs = [f"{l}_{a}" for l in DAM_LOCS for a in AXLE_NAMES]
    pi_tab = ""
    pi_tab += r"\begin{tabular}{l" + len(loc_veh_pairs) * "c" + "}\n"
    pi_tab += r"""
    \toprule
    {    } & \multicolumn{3}{c}{$\DI{LL}$ (\unit{\micro\radian})} & \multicolumn{3}{c}{$\DI{RR}$ (\unit{\micro\radian})}\\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    $\delta$ & All &  5-axle & 6-axle &  All & 5-axle &  6-axle\\
    \midrule
    """
    for d1 in delta_vals:
        pi_tab += f"{d1:.2f}"
        for lv in loc_veh_pairs:
            pi_base = get_pi_via_kde(emds.loc[0.0, lv])
            pi = get_pi_via_kde(emds.loc[d1, lv])
            pi_sep = pi_base[1] < pi[0]
            pi_tab += (
                " & " + (r"\bfseries" if pi_sep else "") + f"({pi[0]:.3}, {pi[1]:.3})"
            )
        pi_tab += r"\\" + "\n"

    pi_tab += r"\bottomrule\end{tabular}"
    return pi_tab


def print_pred_int_table(emds, DAM_LOCS, CONF=0.95):
    print(format_pred_int_table(emds, DAM_LOCS, delta_vals, CONF))
