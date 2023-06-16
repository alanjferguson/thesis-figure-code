import pandas as pd
import numpy as np

import ajf_plts
import code

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import wasserstein_distance

N_VEHICLES = 1000

ITERATIONS = 100

delta1_vals = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1]
xd1_vals = [0.0, 0.0, 0.0, 34 / 2.0, 0.0, 0.0, 34 / 3.0]

delta2_vals = [0.0, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2]
xd2_vals = [0.0, 34 / 2.0, 34 / 2.0, 34 / 2.0, 34 / 3.0, 34 / 3.0, 34 / 3.0]

# get results values to bootstap from
res_df = pd.read_parquet("./res_all.parquet").set_index(["delta", "x"])

emd_L = np.zeros((len(delta1_vals), ITERATIONS))
emd_R = np.zeros((len(delta1_vals), ITERATIONS))

for i, (d1, xd1, d2, xd2) in enumerate(
    zip(delta1_vals, xd1_vals, delta2_vals, xd2_vals)
):

    for j in range(ITERATIONS):
        L1 = res_df.loc[d1].loc[xd1].sample(N_VEHICLES, replace=False)["max_l"].values
        L2 = res_df.loc[d2].loc[xd2].sample(N_VEHICLES, replace=False)["max_l"].values

        R1 = res_df.loc[d1].loc[xd1].sample(N_VEHICLES, replace=False)["max_r"].values
        R2 = res_df.loc[d2].loc[xd2].sample(N_VEHICLES, replace=False)["max_r"].values

        emd_L[i, j] = wasserstein_distance(L1, L2)
        emd_R[i, j] = wasserstein_distance(R1, R2)

print(emd_L.mean(axis=1))
print(emd_L.std(axis=1))

print(emd_R.mean(axis=1))
print(emd_R.std(axis=1))

code.interact(local=locals())
