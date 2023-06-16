#!/usr/bin/env python3

"""
Created on Sat Jun  6 21:44:30 2020

@author: alanferguson
"""
import math

import numpy as np

from numba import njit

# BEAM MODEL PARAMETERS
# MUST HAVE AN EVEN NUMBER OF ELEMENTS OR MID SPAN IS WRONG
NUM_ELEMENTS = 68  # number of elements in bridge
LE = 0.50  # length of element in metres
# NEED TO UPDATE FIXED_DOF BELOW IF THESE CHANGE

# BEAM PHYSICAL PROPERTIES
DEFAULT_E = 3e10  # beam Young's modulus
MOMENT_INERTIA = 6.0  # beam moment of inertia (m^4)
AREA = 10.0  # beam cross section area (m^2)
DEPTH = 2.2  # beam depth (m)
L_CRACK = 1.5 * DEPTH  # length of beam affected by crack
DENSITY = 2446.0  # beam material density (kg/m^3)

# FIXED MODEL PROPERTIES
# (DO NOT CHANGE THESE AT ANYTIME UNLESS YOU KNOW WHAT YOU'RE AT)
NODE_PER_ELEM = 2
DOF_PER_NODE = 2
DOF_PER_ELEM = NODE_PER_ELEM * DOF_PER_NODE

# DERIVED MODEL PROPERTIES / VALUES
NUM_NODES = ((NODE_PER_ELEM - 1) * NUM_ELEMENTS) + 1  # total nodes in system
NUM_DOF = DOF_PER_NODE * NUM_NODES  # total num DOFs in system
BEAM_LENGTH = NUM_ELEMENTS * LE  # total length of bridge in metres
NODE_X_VALS = np.linspace(0, NUM_ELEMENTS, NUM_NODES)  # x position of nodes

# USEFUL CONSTANT INDICES
LHS_ROT_IDX = 1
RHS_ROT_IDX = NUM_DOF - 2
MID_DIS_IDX = NUM_DOF // 2 - 1

###############################################################################
# METHODS
###############################################################################


@njit(cache=True, parallel=False, fastmath=True)
def _applyBoundaryCondsMatr(mat):
    # Need to define this here as Numba doesn't like globals
    FIXED_DOF = [0, -2]  # restrained degrees of freedom

    for dof in FIXED_DOF:
        # zero rows and columns
        mat[dof, :] = 0
        mat[:, dof] = 0

        # set main diagonal elements to 1
        mat[dof, dof] = 1

    return mat


def _applyBoundaryCondsVect(vec):
    # Need to define this here as Numba doesn't like globals
    FIXED_DOF = [0, NUM_ELEMENTS * 2]  # restrained degrees of freedom

    for dof in FIXED_DOF:
        # zero rows and columns
        vec[dof] = 0

    return vec


# MASS MATRICES DOESN'T CHANGE DURING RUNTIME


@njit(cache=True, parallel=False, fastmath=True)
def _assemble_mass_matrix():
    e = (DENSITY * AREA * LE) / 420

    # Elementary Mass Matrix
    Me = np.array(
        [
            [156 * e, 22 * LE * e, 54 * e, -13 * LE * e],
            [22 * LE * e, 4 * e * (LE**2), 13 * LE * e, -3 * e * (LE**2)],
            [54 * e, 13 * LE * e, 156 * e, -22 * LE * e],
            [-13 * LE * e, -3 * e * (LE**2), -22 * LE * e, 4 * e * (LE**2)],
        ]
    )

    Mg = np.zeros((NUM_DOF, NUM_DOF))

    # iterate for all nodes (e.g. every second dof)
    for i in range(0, NUM_DOF - DOF_PER_ELEM + 1, 2):
        Mg[i : i + 4, i : i + 4] += Me

    return Mg


# Construct Mass matrix and apply boundary conditions
GLOBAL_MASS_MATRIX = _applyBoundaryCondsMatr(_assemble_mass_matrix()).copy()


@njit(cache=True, parallel=False, fastmath=True)
def _get_elem_stiff_mat(E, I):
    a = 12 / LE**3
    b = 6 / LE**2
    c = 4 / LE
    d = 2 / LE

    return (
        E * I * np.array([[a, b, -a, b], [b, c, -b, d], [-a, -b, a, -b], [b, d, -b, c]])
    )


@njit(cache=True, parallel=False, fastmath=True)
def _assemble_undamaged_K_matr(E):
    # Assemble elementary stiffness matrix
    Ke = _get_elem_stiff_mat(E, MOMENT_INERTIA)

    Kg = np.zeros((NUM_DOF, NUM_DOF), dtype=np.float64)

    # iterate for all nodes (e.g. every second dof)
    for i in range(0, NUM_DOF - DOF_PER_ELEM + 1, 2):
        Kg[i : i + 4, i : i + 4] += Ke

    return Kg


@njit(cache=True, parallel=False, fastmath=True)
def _get_I_at_x(x, x_crack, delta):
    L_crack = 1.5 * DEPTH
    crack_dist = np.abs(x - x_crack)
    coeff = (DEPTH - (delta * DEPTH)) ** 3 / DEPTH**3
    I_crack = coeff * MOMENT_INERTIA
    C = MOMENT_INERTIA - I_crack
    I_x = I_crack + crack_dist * (MOMENT_INERTIA - I_crack) / L_CRACK
    return np.where(crack_dist > L_CRACK, MOMENT_INERTIA, I_x)


@njit(cache=True, parallel=False, fastmath=True)
def _assemble_damaged_K_matr(E, x_crack, delta):
    Kg = np.zeros((NUM_DOF, NUM_DOF), dtype=np.float64)

    Ke = _get_elem_stiff_mat(E, MOMENT_INERTIA)

    x_vals = (np.arange(NUM_ELEMENTS) + 0.5) * LE

    I_vals = _get_I_at_x(x_vals, x_crack, delta)

    # iterate for all nodes (e.g. every second dof)
    for dof_idx, I_x in zip(range(0, NUM_DOF - DOF_PER_ELEM + 1, 2), I_vals):
        Kg[dof_idx : dof_idx + 4, dof_idx : dof_idx + 4] += (I_x / MOMENT_INERTIA) * Ke

    return Kg


@njit(cache=True, parallel=False, fastmath=True)
def _calcStiffnessReduction(E, damElem, delta):
    lc = 1.5 * DEPTH  # length affected by the crack
    # total number of elements affected by crack
    TN = math.ceil(lc / LE)

    # length of first/last beam element unaffected by crack
    L_unaff = (TN * LE) - lc
    P_unaff = L_unaff / LE  # proportion of first/last element unaffected
    P_aff = 1 - P_unaff  # proportion of first/last element affected by crack
    L_aff = LE * P_aff  # length of first/last element affected by crack

    # coefficient defines how Ir varies for a given delta
    coefficient = (DEPTH - (delta * DEPTH)) ** 3 / DEPTH**3
    Ir = coefficient * MOMENT_INERTIA

    IdamHalf = np.zeros(TN)

    IdamHalf[0] = (MOMENT_INERTIA * P_unaff) + P_aff * (
        MOMENT_INERTIA - ((MOMENT_INERTIA - Ir) * ((L_aff / 2) / lc))
    )

    IdamHalf[1:] = [
        MOMENT_INERTIA - ((MOMENT_INERTIA - Ir) * ((L_aff + (x * LE) - (LE / 2)) / lc))
        for x in range(1, TN)
    ]

    # now flip Idam for the second half
    Idam = np.concatenate((IdamHalf, IdamHalf[::-1]), axis=0)

    # now assemble as before
    Kg = np.zeros((NUM_DOF, NUM_DOF))

    Ke = _get_elem_stiff_mat(E, MOMENT_INERTIA)

    for i in range(0, 2 * (damElem - TN), 2):
        Kg[i : i + 4, i : i + 4] += Ke

    for e, I_red in zip(range(2 * (damElem - TN), 2 * (damElem + TN), 2), Idam):
        Kg[e : e + 4, e : e + 4] += _get_elem_stiff_mat(E, I_red)

    for i in range(2 * (damElem + TN), NUM_DOF - DOF_PER_ELEM + 1, 2):
        Kg[i : i + 4, i : i + 4] += Ke

    return Kg


def get_E_val_from_temp(temp):
    return (-0.125 * temp + 29.13) * 1e9


@njit(cache=True, parallel=False, fastmath=True)
def update_E_val(Kg, old_E, new_E):
    return Kg * (new_E / old_E)


@njit(cache=True, parallel=False, fastmath=True)
def inflict_damage(Kg, E, damageElem, delta):
    Kg = _calcStiffnessReduction(E, damageElem, delta)
    return _applyBoundaryCondsMatr(Kg)


@njit(cache=True, parallel=False, fastmath=True)
def inflict_damage_at_x(E, x_crack, delta):
    Kg = _assemble_damaged_K_matr(E, x_crack, delta)
    return _applyBoundaryCondsMatr(Kg)


@njit(cache=True, parallel=False, fastmath=True)
def assemble_undam_K_matr(E):
    return _applyBoundaryCondsMatr(_assemble_undamaged_K_matr(E))


class Beam(object):
    """Class for performing static and dynamic cracked beam simulations"""

    def __init__(self):
        # setup default values
        self._currentE = DEFAULT_E
        self._current_delta = 0.0
        self._damaged_elem = NUM_ELEMENTS // 3

        self.Mg = _applyBoundaryCondsMatr(_assemble_mass_matrix()).copy()

        # Assemble healthy, undamaged stiffness matrix, and apply boundary CondsMatr
        self._undamagedKg = _assemble_undamaged_K_matr(self._currentE)
        self.Kg = _applyBoundaryCondsMatr(self._undamagedKg)

        # Set initial E value

    @property
    def E(self):
        return self._currentE

    @E.setter
    def E(self, newE):
        # scale Kg by newE/currentE
        self.Kg *= newE / self._currentE
        self._currentE = newE

    @property
    def delta(self):
        return self._current_delta

    @delta.setter
    def delta(self, new_delta):
        if self._current_delta != new_delta:
            self.reset_damage()
            self.inflict_damage(self._damaged_elem, new_delta)
            self._current_delta = new_delta

    def inflict_damage(self, damageElem, delta):
        self.Kg = inflict_damage(self.Kg, self.E, damageElem, delta)

    def inflict_damage_at_x(self, x_crack, delta):
        self.Kg = inflict_damage_at_x(self.E, x_crack, delta)

    def reset_damage(self):
        self.Kg = _applyBoundaryCondsMatr(self._undamagedKg)
