import pandas as pd
import numpy as np

import BeamModel as bm
from BeamModel import Beam
from Simulation import perform_dynamic_sim

import ajf_plts

# using file name to seed RNG
rng = ajf_plts.get_rng(__file__)

# Fixed temperature T = 25 degrees C
# Fixed velocity, v = 15 m/s

wim_df = pd.read_parquet("./wim_data.parquet")

# Filter to leave only 5 axle vehicles
wim_df = wim_df.loc[wim_df.Axles.isin([5])]

# Round to 1 dp
wim_df = wim_df.round(1)

SPACING_LABELS = ["S0", "S1", "S2", "S3", "S4"]
WEIGHT_LABELS = ["W1", "W2", "W3", "W4", "W5"]

# Filter to leave only the most common axle plan
ax_plan = wim_df.groupby(SPACING_LABELS).agg("count").W1.idxmax()
wim_df = wim_df.loc[(wim_df[SPACING_LABELS] == ax_plan).all(axis=1)]

wim_df = wim_df.drop(columns=["Axles", "dayofweek", "hour", "S5", "W6"])

delta_vals = [0.0, 0.1, 0.2]

N_VEHICLES = (200 + 2000)  # per damage level

fe_Fs = 100
fe_Ts = 1.0 / fe_Fs


results = []


def process_row(arr, kg):
    disp = perform_dynamic_sim(kg, arr[5:], arr[:5], 15)[1]
    return np.max(disp[bm.LHS_ROT_IDX, :])


for delta in delta_vals:
    curr_df = wim_df.sample(N_VEHICLES, replace=True, random_state=rng).reset_index(drop=True)

    # Run these as a static simulation for healthy and damaged cases
    beam = Beam()
    Beam.E = 2.913e10 - 0.125 * 25
    beam.inflict_damage_at_x(bm.BEAM_LENGTH / 3.0, delta)

    print("starting", delta)
    curr_df["max_l"] = curr_df.apply(
        process_row, axis=1, raw=True, result_type="reduce", kg=beam.Kg
    )

    curr_df["delta"] = delta

    results.append(curr_df)

df = pd.concat(results).reset_index(drop=True).reset_index()

df.reset_index().to_parquet("./data/fig10.parquet")
