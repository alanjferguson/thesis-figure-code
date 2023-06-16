# Copyright (C) 2020 Alan J. Ferguson
# All rights reserved

import numpy as np
from scipy import linalg

from numba import njit
from numba import types
from numba import float32, int32
import numba as nb

from BeamModel import NODE_X_VALS
from BeamModel import LE
from BeamModel import GLOBAL_MASS_MATRIX
from BeamModel import NUM_DOF
from BeamModel import BEAM_LENGTH

import code

###############################################################################
# GENERAL UTILITY METHODS
###############################################################################
@njit(cache=True, parallel=False, fastmath=True)
def _assemble_force_vecs(Paxle, axlePositions, force_pos_nudge=LE / 100):
    Le = LE  # We use this quite a lot!

    # work out number of steps and axles there are
    numSteps, numAxles = axlePositions.shape

    # First we flatten positions and remove values outside bridge
    # We use step_idxs to track where to put these values back
    flat_pos_all = axlePositions.ravel()
    on_bridge_mask = (0.0 <= flat_pos_all) * (flat_pos_all <= BEAM_LENGTH)

    flat_pos = flat_pos_all[on_bridge_mask]
    step_idx = (
        np.arange(numSteps)
        .repeat(numAxles)
        .reshape(-1, numAxles)
        .ravel()[on_bridge_mask]
    )

    # Next we add a very small offset to positions which would be directly
    # over a node, as the code which resolves the external forces into
    # nodal forces assumes forces are always between 2 nodes. This is a bit
    # of a hack but doesn't affect the end result as long as nudge is
    # sufficiently small enough
    for i in range(flat_pos.size):
        if flat_pos[i] == 0:
            flat_pos[i] += force_pos_nudge
        else:
            for x in NODE_X_VALS:
                if flat_pos[i] == x:
                    # With the rest of the nodes subtract this time
                    flat_pos[i] -= force_pos_nudge

    # get equivalent element indices
    axleElemIndices = np.floor(flat_pos / Le)

    # get start index for each force
    axleStartIndices = 2 * axleElemIndices.astype(np.intp)

    # calculate axle delta along element
    axleElemDelta = flat_pos - axleElemIndices * Le

    # assemble overall force vector for each step
    Fext = np.zeros((NUM_DOF, numSteps))

    # pre computing these values for speed and tidiness
    a = axleElemDelta
    a2 = a**2
    a3 = a**3
    L2 = Le**2
    L3 = Le**3

    P = Paxle.repeat(numSteps).reshape(-1, numSteps).T.flatten()[on_bridge_mask]

    local_F0 = P * (1.0 - 3.0 * a2 / L2 + 2.0 * a3 / L3)
    local_F1 = P * (a - 2.0 * a2 / Le + a3 / L2)
    local_F2 = P * (3.0 * a2 / L2 - 2.0 * a3 / L3)
    local_F3 = P * (-1.0 * a2 / Le + a3 / L2)

    for i, step_i in enumerate(step_idx):
        Fext[axleStartIndices[i], step_i] += local_F0[i]
        Fext[axleStartIndices[i] + 1, step_i] += local_F1[i]
        Fext[axleStartIndices[i] + 2, step_i] += local_F2[i]
        Fext[axleStartIndices[i] + 3, step_i] += local_F3[i]

    return Fext


###############################################################################
# STATIC MODEL
# - Solves F = K x for varying steps of force ensemble
###############################################################################


@njit(cache=True, parallel=False, fastmath=True)
def perform_static_sim(Kg, Paxle, axleSpacings, force_step=LE / 10):
    # Calculate number of steps to cover beam
    num_steps = int((BEAM_LENGTH + max(axleSpacings)) / force_step + 1)

    # Calculate all step offsets
    stepOffsets = np.arange(num_steps) * force_step

    # calculate point load positions at each spatial step
    axle_positions = stepOffsets.reshape(-1, 1) - axleSpacings

    # calculate force vectors at each step
    Fext = _assemble_force_vecs(Paxle, axle_positions)

    # solve F = K * x at each spatial step
    disp = np.linalg.solve(Kg, Fext)

    return stepOffsets, disp


###############################################################################
# DYNAMIC MODEL
# - Solves F(t) = Ma(t) + Cv(t) + Kx(t) for each timestep t
###############################################################################


@njit(cache=True, parallel=False, fastmath=True)
def perform_dynamic_sim(Kg, Paxle, axleSpacings, ax_vel, time_step=0.01, theta=1.6):
    Mg = GLOBAL_MASS_MATRIX

    # Constants for Wilson-Theta Method below
    A0 = 6.0 / ((theta * time_step) ** 2.0)
    A1 = 3.0 / (theta * time_step)
    A2 = 2.0 * A1
    A3 = (theta * time_step) / 2.0
    A4 = A0 / theta
    A5 = -A2 / theta
    A6 = 1.0 - (3.0 / theta)
    A7 = time_step / 2.0
    A8 = (time_step**2.0) / 6.0

    # Calculate time vector for vehicle to cross, i.e. time for the
    # first axle to cross length of bridge + length of the vehicle
    # to clear bridge
    end_time = (BEAM_LENGTH + np.max(axleSpacings)) / ax_vel
    time = np.arange(end_time / time_step + 1) * time_step
    num_timesteps = len(time)

    # pre-alloc disp, vel, acc, dispth and F_eff vectors
    disp = np.zeros((NUM_DOF, num_timesteps))
    vel = np.zeros_like(disp)
    acc = np.zeros_like(disp)

    # Next we calculate the the accleration at t=0, the following method for
    # solving the differential equations is the wilson theta method as outlined
    # on page 409 of 'structural dynamics theory and applications'.and you need
    # to look at these notes to be able to follow what is going on in next few
    # lines of code. (However it is worth noting that there is a sign mistake
    # in one of the equations given in the book, basically in the equation that
    # calculates accleration at t=0 there is a plus sign shown where there
    # should be a minus sign i made a note of this in the book and the equation
    # given below is correct

    # NOTE: I assume we always start with beam at rest
    # acc[0, :] = np.linalg.solve(Mg, Fext[0, :]) - (Kg@disp[0, :])

    # effective stiffness matrix Ke
    Ke = Kg + (A0 * Mg)  # + (a1*Cg) <- Cg always 0

    # Calculate corresponding axle positions and nodal forces
    axleOffsets = time * ax_vel
    axle_positions = axleOffsets.reshape(-1, 1) - axleSpacings
    Fext = _assemble_force_vecs(Paxle, axle_positions)

    # Now carry out Wilson-Theta solve stepwise
    # Carry out Wilson-Theta solve
    # NOTE: Damping (Cg) is zero so is missing from eqns below
    Ke_inv = np.linalg.inv(Ke)

    for i in range(num_timesteps):

        # Procedure 1: Calculate the effective force vector at
        # time [t + (th*tk)]

        # need to be careful here, and easiest way to explain why is with an
        # example. lets say that you have a bridge 25m long and vehicle is
        # travling at 20m/s, tk=0.1 and th=1.4. based on the above parameters
        # esssentially the 2nd column in the effective force array tells you
        # the effective force vector at 0.014 seconds.
        F_eff = (
            Fext[:, i - 1]
            + (theta * (Fext[:, i] - Fext[:, i]))
            + Mg @ (A0 * disp[:, i - 1] + A2 * vel[:, i - 1] + 2 * acc[:, i - 1])
        )

        # NOTE: as stiffness and force matrices get larger the code
        # below might be faster
        disp_th = Ke_inv @ F_eff
        # disp_th = np.linalg.solve(Ke, F_eff)

        # Procedure 3:Calculate the displacements, velocities and acclerations
        # at time [t + tk]

        acc[:, i] += (
            (A4 * (disp_th - disp[:, i - 1])) + A5 * vel[:, i - 1] + A6 * acc[:, i - 1]
        )

        vel[:, i] += vel[:, i - 1] + A7 * (acc[:, i] + acc[:, i - 1])

        disp[:, i] += (
            disp[:, i - 1]
            + time_step * (vel[:, i - 1])
            + A8 * (acc[:, i] + (2 * acc[:, i - 1]))
        )

    return time, disp, vel, acc
