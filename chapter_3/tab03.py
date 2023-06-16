import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance

import code

# read in wim data
df = pd.read_feather("./wim_data.feather").sample(100000)


def get_weights_spacings(df):
    WEIGHT_LABELS = ["W1", "W2", "W3", "W4", "W5", "W6"]
    SPACING_LABELS = ["S0", "S1", "S2", "S3", "S4", "S5"]

    weights = df[WEIGHT_LABELS].dropna().values.flatten()
    weights = weights[weights > 0.0]

    spacings = df[SPACING_LABELS].values.flatten()
    spacings = spacings[~np.isnan(spacings)]

    return weights, spacings


N_VEHICLES = [10, 100, 1000]

N_ITERS = 100

MIN_AXLES = [2, 5, 6]
MAX_AXLES = [6, 5, 6]

means = np.zeros((len(N_VEHICLES), len(MIN_AXLES)))
stds = np.zeros((len(N_VEHICLES), len(MIN_AXLES)))

for i, N in enumerate(N_VEHICLES):
    for j, (min_ax, max_ax) in enumerate(zip(MIN_AXLES, MAX_AXLES)):
        emds = np.zeros(N_ITERS)

        mask = df.Axles.between(min_ax, max_ax)

        for k in range(N_ITERS):
            print(i, j, k)
            w1, s1 = get_weights_spacings(df.loc[mask].sample(N, replace=True))
            w2, s2 = get_weights_spacings(df.loc[mask].sample(N, replace=True))
            emds[k] = wasserstein_distance(s1, s2, w1, w2)

        means[i, j] = np.mean(emds)
        stds[i, j] = np.std(emds)

print(means)

print(stds)
