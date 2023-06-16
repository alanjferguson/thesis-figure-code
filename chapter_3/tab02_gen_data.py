import pandas as pd
import numpy as np

import ajf_plts
import code

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from BeamModel import Beam
from Simulation import perform_static_sim

from scipy.stats import wasserstein_distance

from numba import njit
from numba import objmode

# Fixed temperature T = 25 degrees C
# Fixed velocity, v = 15 m/s

SPACING_LABELS = ["S0", "S1", "S2", "S3", "S4"]
WEIGHT_LABELS = ["W1", "W2", "W3", "W4", "W5"]


wim_df = pd.read_feather("./wim_data.feather")

# Filter to leave only 5 axle vehicles
wim_df = wim_df.loc[wim_df.Axles.isin([5])]

# Round to 1 dp
wim_df = wim_df.round(1)

# Filter to leave only the most common axle plan
ax_plan = wim_df.groupby(SPACING_LABELS).agg("count").W1.idxmax()
wim_df = wim_df.loc[(wim_df[SPACING_LABELS] == ax_plan).all(axis=1)]

WEIGHTS = wim_df[WEIGHT_LABELS].values
SPACINGS = wim_df[SPACING_LABELS].values

N_WIM = WEIGHTS.shape[0]


N_VEHICLES = 1000
DELTA = 0.1

ITERATIONS = 100

# delta1_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
# xd1_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 11]
# delta2_vals = [0.0, 0.1, 0.2, 0.1, 0.2, 0.2]
# xd2_vals = [0.0, 17, 17, 11, 11, 11]
delta1_vals = [0.1]
xd1_vals = [17]
delta2_vals = [0.2]
xd2_vals = [17]


@njit(cache=True, parallel=False, fastmath=True)
def get_wim_sample(n):
    all_idxs = np.arange(N_WIM)

    idx = np.random.choice(all_idxs, size=n, replace=True)

    w = WEIGHTS[idx, :]
    s = SPACINGS[idx, :]
    return w, s


@njit(cache=True, parallel=False, fastmath=True)
def process_one_beam(Kg, n, WEIGHTS, SPACINGS):
    all_idxs = np.arange(N_WIM)

    idx = np.random.choice(all_idxs, size=n, replace=True)

    weights_arr = WEIGHTS[idx, :]
    spacings_arr = SPACINGS[idx, :]

    results_L = np.zeros(len(weights_arr))
    results_R = np.zeros(len(weights_arr))
    for row in range(len(weights_arr)):
        steps, disp = perform_static_sim(Kg, weights_arr[row, :], spacings_arr[row, :])
        results_L[row] = np.max(np.abs(disp[1]))
        results_R[row] = np.max(np.abs(disp[-1]))
    return results_L, results_R


@njit(cache=True, parallel=False, fastmath=True)
def process_one_iter(Kg1, Kg2, n, WEIGHTS, SPACINGS):
    L1, R1 = process_one_beam(Kg1, n, WEIGHTS, SPACINGS)
    L2, R2 = process_one_beam(Kg2, n, WEIGHTS, SPACINGS)

    with objmode(emd_l="float64", emd_r="float64"):
        emd_l = wasserstein_distance(L1, L2)
        emd_r = wasserstein_distance(R1, R2)

    return emd_l, emd_r


@njit(cache=True, parallel=False, fastmath=True)
def process_all_iters(n_iters, Kg1, Kg2, n_vehicles, WEIGHTS, SPACINGS):
    emd_l = np.zeros(n_iters)
    emd_r = np.zeros(n_iters)

    for i in range(n_iters):
        emd_l[i], emd_r[i] = process_one_iter(Kg1, Kg2, n_vehicles, WEIGHTS, SPACINGS)

    return emd_l, emd_r


# Run these as a static simulation for healthy and damaged cases
beam_1 = Beam()
beam_2 = Beam()

emd_L = np.zeros((len(delta1_vals), ITERATIONS))
emd_R = np.zeros((len(delta1_vals), ITERATIONS))

for i, (d1, xd1, d2, xd2) in enumerate(
    zip(delta1_vals, xd1_vals, delta2_vals, xd2_vals)
):
    print("i=", i)

    beam_1.reset_damage()
    beam_1.inflict_damage_at_x(xd1, d1)
    print("updated beam 1")

    beam_2.reset_damage()
    beam_2.inflict_damage_at_x(xd2, d2)
    print("updated beam 2")

    emd_L[i, :], emd_R[i, :] = process_all_iters(
        ITERATIONS, beam_1.Kg, beam_2.Kg, N_VEHICLES, WEIGHTS, SPACINGS
    )

print(emd_L.mean(axis=1))
print(emd_L.std(axis=1))

print(emd_R.mean(axis=1))
print(emd_R.std(axis=1))
