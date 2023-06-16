import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance

import os

res_dir = "./51y100kv3/"

axle_sets = {"5": [5], "6": [6], "all": [2, 3, 4, 5, 6]}


def get_df_by_months(months, axles):
    df = []
    for m in months:
        df.append(
            pd.read_feather(os.path.join(res_dir, "res_month_" + str(m) + ".feather"))
        )
    df = pd.concat(df)
    return df.loc[df.Axles.isin(axles)]


for name, axles in axle_sets.items():
    baseline = get_df_by_months(np.arange(12), axles)

    deltas = np.zeros(51)
    lvl = np.empty(51) * np.NaN
    rvr = np.empty(51) * np.NaN
    lvr = np.empty(51) * np.NaN

    for y in range(0, 51):
        df = get_df_by_months((y * 12) + np.arange(12),
                              axles)

        deltas[y] = df.delta.max()

        lvr[y] = wasserstein_distance(df.max_l.values, df.max_r.values)

        if y >= 1:
            lvl[y] = wasserstein_distance(baseline.max_l.values, df.max_l.values)
            rvr[y] = wasserstein_distance(baseline.max_r.values, df.max_r.values)

    results = pd.DataFrame({"delta": deltas, "lvl": lvl, "rvr": rvr, "lvr": lvr})

    results.to_feather(f"emds_{name}.feather")
