import sys, os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import copy
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ajf_bridge_fem import *

RUN_ID = sys.argv[1]
BLK_ID = sys.argv[2]
IN_FILE = sys.argv[3]
DOFS = np.array(eval(sys.argv[4]))
DOF_NAMES = np.array(eval(sys.argv[5]))
MODEL = sys.argv[6]
RUN_PATH = sys.argv[7]

# Load vehicle and input data
#print(f'{BLK_ID:} loading {IN_FILE}')
in_df = pd.read_feather(IN_FILE)

if MODEL == "LB":
    beam = construct_model_LB()
elif MODEL == "GH":
    beam = construct_model_GH()
else:
    error("Incorrect model name")

#print(f'{BLK_ID}: constructed model')

# Run processing
def process_row(row, fem, dofs):
    WEIGHT_COLS = [f'W{i+1}' for i in range(6)]
    SPACING_COLS = [f'S{i}' for i in range(6)]
    
    fem.E = row.E_val
    fem.reset_crack_damage()
    fem.add_crack_damage(row.x_dam, row.delta)
    fem.update_model(w_1=row.freq)
    steps, disp = fem.perform_static_sim(row[WEIGHT_COLS].values, 
                                         row[SPACING_COLS].values)
    return dict(zip(dofs, [np.ptp(disp[d]) for d in dofs]))

#print(f'{BLK_ID}: starting processing')

import time

start = time.time()

temp_results = in_df.apply(process_row, axis=1, fem=beam, dofs=DOFS)

end = time.time()

print(f'{BLK_ID}: finished processing in {end - start}s')

save_file = f'{RUN_PATH}/{RUN_ID}_{BLK_ID}_results.feather'

results = pd.DataFrame(temp_results.to_list())
results.columns = DOF_NAMES

in_df.join(results).to_feather(save_file)

print(f'{BLK_ID}: saved to {save_file}, shutting down')