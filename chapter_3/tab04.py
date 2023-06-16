import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance

import ajf_plts

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

# read in temperature data
df = pd.read_feather("./data/CEDA_temp_data.feather")

# convert to 0 based years and months
df.year -= df.year.min()
df.month -= df.month.min()

# get sequentially numbered months
df["seq_month"] = df.year * 12 + df.month

PERIODS = [1, 6, 12, 24]
N_SAMPLES = [100, 1000, 10000]
N_ITERS = 1000

MIN_MONTH = df.seq_month.min()
MAX_MONTH = df.seq_month.max()


def get_sample(M, N):
    offset = rng.integers(MIN_MONTH,
                          MAX_MONTH - M)
    months = offset + np.arange(M)
    temps = df.temp.loc[df.seq_month.isin(months)]\
                   .sample(N, replace=True, random_state=rng)
    return temps.values


means = np.zeros((len(PERIODS), len(N_SAMPLES)))
stds = np.zeros_like(means)

for i, M in enumerate(PERIODS):
    for j, N in enumerate(N_SAMPLES):
        emds = np.zeros(N_ITERS)
        for k in range(N_ITERS):
            s1 = get_sample(M, N)
            s2 = get_sample(M, N)

            emds[k] = wasserstein_distance(s1, s2)
        means[i, j] = np.mean(emds)
        stds[i, j] = np.std(emds)

result_str = ""
for i, M in enumerate(PERIODS):
    result_str += f"{M:2d} "
    for j, _ in enumerate(N_SAMPLES):
        result_str += f"{means[i,j]:03.3f}({stds[i,j]:03.3f}) "
    result_str += "\n"

with open("./output/tab04.txt", "w") as f:
    f.write(result_str)
