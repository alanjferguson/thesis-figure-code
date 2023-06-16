import numpy as np
import pandas as pd

from BeamModel import Beam
from Simulation import perform_static_sim

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import ajf_plts

import code

delta = [0.0, 0.1, 0.2, 0.1, 0.2]
x_d = np.floor([34 / 3, 34 / 3, 34 / 3, 34 / 2, 34 / 2])

P = np.array([1e3 / 9.818])
S = np.array([0])

beam = Beam()

steps = []
L = []
R = []

for d, x in zip(delta, x_d):
    beam.reset_damage()
    beam.inflict_damage(x, d)

    s, disp = perform_static_sim(beam.Kg, P, S)

    steps.append(s)
    L.append(disp[1])
    R.append(np.abs(disp[-1]))

df = pd.DataFrame({"delta": delta, "x_d": x_d, "steps": steps[:], "L": L[:], "R": R[:]})

df["condition_d"] = df.delta.astype("str")
df["condition_xd"] = df.x_d.map({x_d[1]: "L/3", x_d[3]: "L/2"})
df["condition"] = "$\delta=" + df.condition_d + "$@$" + df.condition_xd + "$"
df.loc[df.delta == 0.0, "condition"] = "$\delta=" + df.delta.astype("str") + "$"
df = df.drop(columns=["condition_d", "condition_xd"])

df = df.set_index(["delta", "x_d", "condition"]).apply(pd.Series.explode).reset_index()

fig, axes = plt.subplots(
    figsize=(7.5, 12), ncols=2, nrows=4, sharex="row", sharey="row"
)
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.ravel()

groups = df.groupby("condition", sort=False)

colors = ajf_plts.get_color_cyc_cb()
linestyles = ajf_plts.get_line_sty_cyc()

for name, group in groups:
    col = next(colors)
    ls = next(linestyles)
    ax1.plot(group.steps.values / 34.0, group.L.values, label=name, c=col, ls=ls)
    ax2.plot(group.steps.values / 34.0, group.R.values, label=name, c=col, ls=ls)

ax2.legend(title="Condition")

ax1.set_ylabel("Rotation / $\\textrm{rad}\cdot\\textrm{kN}^{-1}$", usetex=True)

ax1.set_xlabel("Position")
ax2.set_xlabel("Position")


delta_vals = np.linspace(0, 0.2, 50)
pos_vals = np.arange(0, 34)

max_l = np.zeros((len(pos_vals), len(delta_vals)))
max_r = np.zeros((len(pos_vals), len(delta_vals)))

for i, p in enumerate(pos_vals):
    for j, d in enumerate(delta_vals):
        beam.reset_damage()
        beam.inflict_damage(p, d)

        s, disp = perform_static_sim(beam.Kg, P, S)

        max_l[i, j] = np.max(disp[1])
        max_r[i, j] = np.max(np.abs(disp[-1]))

pv, dv = np.meshgrid(pos_vals / 34.0, delta_vals)

CS = ax3.contourf(pv, dv, max_l.T)
fig.colorbar(CS, ax=ax3)

CS = ax4.contourf(pv, dv, max_r.T)
fig.colorbar(CS, ax=ax4)

ax3.set_xlim([0, 1.0])
ax4.set_xlim([0, 1.0])

ax3.set_xlabel("Damage Position")
ax4.set_xlabel("Damage Position")

ax3.set_ylabel("Damage Severity ($\delta$)")

max_lvl = max_l.T - max_l[:, 0]
max_rvr = max_r.T - max_l[:, 0]

CS = ax5.contourf(pv, dv, max_lvl)
fig.colorbar(CS, ax=ax5)

CS = ax6.contourf(pv, dv, max_rvr)
fig.colorbar(CS, ax=ax6)

ax5.set_xlim([0, 1.0])
ax6.set_xlim([0, 1.0])

ax5.set_xlabel("Damage Position")
ax6.set_xlabel("Damage Position")

ax5.set_ylabel("Damage Severity ($\delta$)")

max_lvr = np.abs(max_l.T - max_r.T)

CS = ax7.contourf(pv, dv, max_lvr)
fig.colorbar(CS, ax=ax7)

max_loc = np.cbrt(max_rvr / max_lvl)

CS = ax8.contourf(pv, dv, max_loc)
fig.colorbar(CS, ax=ax8)

ax7.set_xlim([0, 1.0])
ax8.set_xlim([0, 1.0])

ax7.set_xlabel("Damage Position")
ax8.set_xlabel("Damage Position")

ax7.set_ylabel("Damage Severity ($\delta$)")

ajf_plts.caption_axes(axes.ravel(), v_pos=-0.20)
fig.tight_layout()
plt.show()
