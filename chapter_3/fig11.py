import pandas as pd

import matplotlib.pyplot as plt

import ajf_plts

plt.style.use(["./ajf_plts/base.mplstyle"])

# read in speed data
df = pd.read_feather("./speed_data.feather")
df.x *= 3.6  # convert to km/h
df = df.set_index("x")

# set 0 point
df.iloc[0, :] = 0.0
df = df.sort_index()
df /= df.sum()

fig, ax = plt.subplots(figsize=ajf_plts.single_wide_figsize)

for i, axles in enumerate(["2", "3", "4", "5", "6"]):
    ax.plot(df.index, df[axles],
            label=axles,
            zorder=10-i)

ax.legend(title="Axles")

ax.set_xlabel(r"Velocity / \unit{\kilo\meter\per\hour}")
ax.set_ylabel("Probability")

fig.tight_layout()
ajf_plts.save_fig(fig)
