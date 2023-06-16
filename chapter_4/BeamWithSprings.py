#!/usr/bin/env python3

"""
Created on Sat Jun  6 21:44:30 2020

@author: alanferguson
"""

import numpy as np

from numba import njit

from pprint import pformat

import code

# Going to simulate a beam with elastic supports (linear + rotational) at each end

# NOTES: 
#   * MUST HAVE AN EVEN NUMBER OF BEAM ELEMENTS OR MID SPAN IS WRONG

# FE MODEL PROPERTIES
# (DO NOT CHANGE THESE AT ANYTIME UNLESS YOU KNOW WHAT YOU'RE AT)
NODE_PER_BEAM_ELEM = 2
DOF_PER_NODE = 2 
DOF_PER_ELEM = NODE_PER_BEAM_ELEM * DOF_PER_NODE

##############################################################################
# DOFs are laid out as follows:
# Beam element nodes come first as follows
# NB = N_BEAM_ELEMS
#          |       BEAM LENGTH = N. ELEMS * LE           |
#          |<------------------------------------------->|
#          |                                             |
#          +-------+-------+-------+~...~+-------+-------+
#   Element|   0   |   1   |   2   |     | N_B-2 | N_B-1 |
#          +-------+-------+-------+~...~+-------+-------+
#          *       *       *       *     *       *       *
#     Node 0       1       2       3 ... N_B-2   N_B-1   N_B
#
# Disp DOF 0       2       4       6 ...
#  Rot DOF 1       3       5       7 ...
# 
# In general for beam elements:
#     Left Node No = Elem No
#     Right Node No = Elem No + 1
#     Disp. DOF = Node No * 2
#     Rot DOF = (Node No * 2) + 1
# 
# Then the support nodes for the springs at the end as follows:
#     LHS Support Node = N_B + 1
#     RHS Support Node = N_B + 2
# DOFs for spring support nodes are laid out as above
#
# Assuming the springs are at nodes 0 and N_B
#     At Disp DOF 0:
#         K = K_beam + K_spring_lin
#         M = M_beam + M_spring_lin (we're going to assume massless springs, i.e. M=0)
#     At Rotation DOF 1:
#         K = K_beam + K_spring_rot
#         M = M_beam + M_spring_rot (we're going to assume massless springs, i.e. M=0)
##############################################################################

###############################################################################
# METHODS
###############################################################################

#@njit(cache=True,parallel=False,fastmath=True)
def _applyBoundaryCondsMatr(mat, fix_dofs, diag_val):
    for dof in fix_dofs:
        # zero rows and columns
        mat[dof, :] = 0
        mat[:, dof] = 0

        # set main diagonal elements to 1
        mat[dof, dof] = diag_val

    return mat

#@njit(cache=True,parallel=False,fastmath=True)
def _get_beam_elem_mass_matr(density, area, Le):
    return ((density * area * Le) / 420.0) * \
            np.array([[156, 22 * Le, 54, -13 * Le],
                     [22 * Le, 4 * (Le ** 2), 13 * Le, -3 * (Le ** 2)],
                     [54, 13 * Le, 156, -22 * Le],
                     [-13 * Le, -3 * (Le ** 2), -22 * Le, 4 * (Le ** 2)]])

#@njit(cache=True,parallel=False,fastmath=True)
def _assemble_beam_mass_matr(density, area, L):
    n_dof = (len(L)+1) * 2
    
    M_beam = np.zeros((n_dof, n_dof))
    
    for elem, (d, a, l) in enumerate(zip(density, area, L)):
        start = elem * 2
        end = (elem+2) * 2
        M_beam[start:end,start:end] += _get_beam_elem_mass_matr(d, a, l)
        
    return M_beam

#@njit(cache=True,parallel=False,fastmath=True)
def _get_beam_elem_stiff_matr(E, I, Le):
    a = 12 / Le ** 3
    b = 6 / Le ** 2
    c = 4 / Le
    d = 2 / Le

    return E * I * np.array([[a, b, -a, b],
                             [b, c, -b, d],
                             [-a, -b, a, -b],
                             [b, d, -b, c]])

#@njit(cache=True,parallel=False,fastmath=True)
def _assemble_beam_stiff_matr(E, I, L):
    n_dof = (len(L)+1) * 2
    
    K_beam = np.zeros((n_dof, n_dof))
    
    for elem, (e, i, l) in enumerate(zip(E, I, L)):
        start = elem * 2
        end = (elem+2) * 2
        K_beam[start:end,start:end] += _get_beam_elem_stiff_matr(e, i, l)
        
    return K_beam

def calc_modal_freqs(Mg, Kg, n_fix_dof=0):
    D = np.linalg.inv(Mg) @ Kg
    eigen_vals = np.linalg.eigvals(D)
    freqs = np.sort(np.sqrt(eigen_vals[eigen_vals>=0.0]) / (2.0 * np.pi))
    # omits freqs associated with boundary conditions
    # we just don't return freqs if they're negative / complex
    return np.real(freqs[n_fix_dof:]) 

def _assemble_damping_matr(xi, M, K, n_fix_dof=0, damp_type='rayleigh'):
    modal_freqs = calc_modal_freqs(M, K, n_fix_dof)
    
    # Stiffness proportional damping
    if damp_type == 'stiffness':
        alpha = (2.0 * xi) / (2.0 * np.pi * modal_freqs[0])
        return alpha * K
    
    # Mass proportional damping
    if damp_type == 'mass':
        return None # TODO: make this error out properly
    
    # Rayleigh Damping
    if damp_type == 'rayleigh':
        w1 = modal_freqs[0] * (2.0*np.pi)
        w2 = modal_freqs[1] * (2.0*np.pi)
        a = (2.0*xi*w1*w2)/(w1+w2)
        b = (2.0*xi)/(w1+w2)
        return a*M + b*K
    
    # TODO: make this error properly
    return None

#@njit(cache=True,parallel=False,fastmath=True)
def _calc_I_val_reductions(self, I, L, damElem, delta):
    n_dof = (len(L)+1) * 2
    
    lc = 1.5 * self.depth  # length affected by the crack
    # total number of elements affected by crack
    TN = math.ceil(lc / LE)

    # length of first/last beam element unaffected by crack
    L_unaff = (TN * LE) - lc
    P_unaff = L_unaff / self.Le  # proportion of first/last element unaffected
    P_aff = 1 - P_unaff  # proportion of first/last element affected by crack
    L_aff = self.Le * P_aff  # length of first/last element affected by crack

    # coefficient defines how Ir varies for a given delta
    coefficient = 1 - (DEPTH - (delta * DEPTH)) ** 3 / DEPTH ** 3
    Ir = coefficient * MOMENT_INERTIA

    IdamHalf = np.zeros(TN)

    IdamHalf[0] = P_aff * (Ir * ((L_aff / 2) / lc))
    IdamHalf[1:] = [Ir * ((L_aff + ((x + 1) * LE) - (LE / 2)) / lc)
                    for x in range(1, TN)]

    # now flip Idam for the second half
    Idam = np.concatenate((IdamHalf, IdamHalf[::-1]), axis=0)

    # now assemble as before
    K_r = np.zeros((n_dof, n_dof))

    for e, I_red in zip(range(2 * (damElem - TN), 2 * (damElem + TN + 1), 2), Idam):
        K_r[e:e + 4, e:e + 4] += _get_beam_elem_stiff_matr(E, I_red)

    return K_r

#@njit(cache=True, parallel=False, fastmath=True)
def _inflict_damage(K_beam, E, damageElem, delta):
    K_beam = _calcStiffnessReduction(E, damageElem, delta)
    return K_beam

def _get_E_val_from_temp(temp):
    return -0.125 * temp + 29.13

import code
###############################################################################
# GENERAL UTILITY METHODS
###############################################################################
#@njit(cache=True,parallel=False,fastmath=True)
def _assemble_force_vecs(Paxle, axlePositions, node_x_vals, num_dof, contact_length):
    force_pos_nudge=np.min(np.diff(axlePositions, axis=0))/10.0

    # work out number of steps and axles there are
    numSteps, numAxles = axlePositions.shape

    # First we flatten positions and remove values outside bridge
    # We use step_idxs to track where to put these values back
    flat_pos_all = axlePositions.ravel()
    on_bridge_mask = (np.min(node_x_vals) <= flat_pos_all) * \
                     (flat_pos_all <= np.max(node_x_vals))

    flat_pos = flat_pos_all[on_bridge_mask]
    step_idx = np.arange(numSteps).repeat(
        numAxles).reshape(-1, numAxles).ravel()[on_bridge_mask]

    #code.interact(local=locals())
    # Allows for smoother entry of forces at start and end
    if contact_length > 0.0:
        end_offsets = np.minimum(flat_pos, np.max(node_x_vals)-flat_pos)
        P_coeff = np.where(end_offsets<contact_length,end_offsets/contact_length,1.0)
    else:
        P_coeff = np.ones(flat_pos.shape)
    
    # Next we add a very small offset to positions which would be directly
    # over a node, as the code which resolves the external forces into
    # nodal forces assumes forces are always between 2 nodes. This is a bit
    # of a hack but doesn't affect the end result as long as nudge is
    # sufficiently small enough
    for i in range(flat_pos.size):
        if flat_pos[i] == 0:
            flat_pos[i] += force_pos_nudge
        else:
            for x in node_x_vals:
                if flat_pos[i] == x:
                    # With the rest of the nodes subtract this time
                    flat_pos[i] -= force_pos_nudge
    
    # get equivalent element indices
    axleElemIndices = np.searchsorted(node_x_vals, flat_pos) - 1

    # get start index for each force
    axleStartIndices = 2 * axleElemIndices.astype(np.intp)

    # calculate axle delta along element
    axleElemDelta = flat_pos - node_x_vals[axleElemIndices]
    
    # assemble overall force vector for each step
    Fext = np.zeros((num_dof, numSteps))
    
    # pre computing these values for speed and tidiness
    a = axleElemDelta
    a2 = a ** 2
    a3 = a ** 3
    L = np.diff(node_x_vals)[axleElemIndices]
    L2 = L ** 2
    L3 = L ** 3

    P = Paxle.repeat(numSteps).reshape(-1,
                                       numSteps).T.flatten()[on_bridge_mask]
    P *= P_coeff

    local_F0 = P * (1.0 - 3.0 * a2 / L2 + 2.0 * a3 / L3)
    local_F1 = P * (a - 2.0 * a2 / L + a3 / L2)
    local_F2 = P * (3.0 * a2 / L2 - 2.0 * a3 / L3)
    local_F3 = P * (-1.0 * a2 / L + a3 / L2)

    for i, step_i in enumerate(step_idx):
        Fext[axleStartIndices[i], step_i] += local_F0[i]
        Fext[axleStartIndices[i]+1, step_i] += local_F1[i]
        Fext[axleStartIndices[i]+2, step_i] += local_F2[i]
        Fext[axleStartIndices[i]+3, step_i] += local_F3[i]

    return Fext

###############################################################################
# STATIC MODEL
# - Solves F = K x for varying steps of force ensemble
###############################################################################
#@njit(cache=True,parallel=False,fastmath=True)
def _perform_static_sim(Kg, node_x_vals, num_dof, Paxle, axleSpacings, force_step, contact_length):
    # TODO: Fix this HACK!!!!
    Le = np.median(np.diff(node_x_vals))
    beam_length = np.ptp(node_x_vals)
    
    # Calculate number of steps to cover beam
    num_steps = int((np.max(node_x_vals) + max(axleSpacings)) / force_step + 1)

    # Calculate all step offsets
    stepOffsets = np.arange(num_steps) * force_step

    # calculate point load positions at each spatial step
    axle_positions = stepOffsets.reshape(-1,1) - axleSpacings

    # calculate force vectors at each step
    Fext = _assemble_force_vecs(Paxle, axle_positions, node_x_vals, num_dof, contact_length)

    # solve F = K * x at each spatial step
    disp = np.linalg.solve(Kg,Fext)

    return stepOffsets, disp

###############################################################################
# DYNAMIC MODEL
# - Solves F(t) = Ma(t) + Cv(t) + Kx(t) for each timestep t
###############################################################################
#@njit(cache=True,parallel=False,fastmath=True)
def _perform_dynamic_sim(Kg, Cg, Mg, node_x_vals, num_dof, Paxle, axleSpacings, ax_vel, time_step, theta, contact_length):
    # TODO: Fix this HACK!!!
    Le = np.median(np.diff(node_x_vals))
    beam_length = np.ptp(node_x_vals)
    
    # Constants for Wilson-Theta Method below
    A0 = 6.0 / ((theta * time_step) ** 2.0)
    A1 = 3.0 / (theta * time_step)
    A2 = 2.0 * A1
    A3 = (theta * time_step) / 2.0
    A4 = A0 / theta
    A5 = -A2 / theta
    A6 = 1.0 - (3.0 / theta)
    A7 = time_step / 2.0
    A8 = (time_step ** 2.0) / 6.0

    # Calculate time vector for vehicle to cross, i.e. time for the
    # first axle to cross length of bridge + length of the vehicle
    # to clear bridge
    end_time = (beam_length+np.max(axleSpacings))/ax_vel
    time = np.arange(end_time/time_step + 1)*time_step
    num_timesteps = len(time)

    # pre-alloc disp, vel, acc, dispth and F_eff vectors
    disp = np.zeros((num_dof, num_timesteps))
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
    Ke = Kg + (A0*Mg) + (A1*Cg)

    # Calculate corresponding axle positions and nodal forces
    axleOffsets = time * ax_vel
    axle_positions = axleOffsets.reshape(-1,1) - axleSpacings
    Fext = _assemble_force_vecs(Paxle, axle_positions, node_x_vals, num_dof, contact_length)

    # Now carry out Wilson-Theta solve stepwise
    # Carry out Wilson-Theta solve
    # NOTE: Damping (Cg) is zero so is missing from eqns below

    for i in range(num_timesteps):

        # Procedure 1: Calculate the effective force vector at
        # time [t + (th*tk)]

        # need to be careful here, and easiest way to explain why is with an
        # example. lets say that you have a bridge 25m long and vehicle is
        # travling at 20m/s, tk=0.1 and th=1.4. based on the above parameters
        # esssentially the 2nd column in the effective force array tells you
        # the effective force vector at 0.014 seconds.
        F_eff = Fext[:,i-1] + (theta*(Fext[:,i]-Fext[:,i]))
        F_eff += Mg @ (A0 * disp[:,i-1]+A2 * vel[:,i-1]+2*acc[:,i-1])
        F_eff += Cg @ (A1 * disp[:,i-1] + 2.0*vel[:,i-1] + A3 * acc[:,i-1])

        # Procedure 2: Solve for the displacements at time [t + (th*tk)]
        disp_th = np.linalg.solve(Ke, F_eff)

        # Procedure 3:Calculate the displacements, velocities and acclerations
        # at time [t + tk]
        acc[:,i] += (A4*(disp_th - disp[:,i-1])) + A5*vel[:,i-1] + A6*acc[:,i-1]
        vel[:,i] += vel[:,i-1] + A7 * (acc[:,i] + acc[:,i-1])
        disp[:,i] += disp[:,i-1] + time_step * (vel[:,i-1]) + A8*(acc[:,i] + (2*acc[:,i-1]))

    return time, disp, vel, acc

class SpringDamperSupport(object):
    def __init__(self, beam_node, k_lin, k_rot, c_lin, c_rot):
        self.beam_node = beam_node
        self.K_lin = np.array([[k_lin, -k_lin,],
                               [-k_lin, k_lin]])
        self.C_lin = np.array([[c_lin, -c_lin,],
                               [-c_lin, c_lin]])
        self.K_rot = np.array([[k_rot, -k_rot,],
                               [-k_rot, k_rot]])
        self.C_rot = np.array([[c_rot, -c_rot,],
                               [-c_rot, c_rot]])

class CrackDamage(object):
    def __init__(self, x_dam, delta):
        self.x_dam = x_dam
        self.delta = delta
    def __repr__(self):
        return pformat(vars(self))
    

class Beam(object):
    """Class for performing static and dynamic cracked beam simulations"""
    def __init__(self, n_beam_elems, beam_elem_len):
        # BEAM+SUPPORT PROPERTIES
        self.N_BEAM_ELEMS = n_beam_elems # number of elements in bridge beam/deck
        
        self.L = beam_elem_len          # length of each beam element in metres
        
        # DERIVED MODEL PROPERTIES / VALUES
        self.E = 3e10         # beam Young's modulus
        self.I = 6.0            # beam moment of inertia (m^4)
        self.density = 2446.0   # beam material density (kg/m^3)
        self.area = 4.0        # beam cross section area (m^2)
        self.depth = 2.0        # beam depth (m)
        
        self.damp_ratio = 0.0  # Xi (%)
        
        # Default to no supports
        self.supports = []
        
        # No restrained DOFs in beam to start
        self._fix_dofs = np.array([])
        
        # Default to no damage
        self.cracks = []
        
        self.update_model()
    
    @property
    def L(self):
        return self._L
    
    @L.setter
    def L(self, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            self._L = val
        else:
            self._L = val * np.ones((self.N_BEAM_ELEMS))
            
    @property
    def E(self):
        return self._E
    
    @E.setter
    def E(self, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            self._E = val
        else:
            self._E = val * np.ones((self.N_BEAM_ELEMS))
    
    @property
    def I(self):
        return self._I
    
    @I.setter
    def I(self, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            self._I = val
        else:
            self._I = val * np.ones((self.N_BEAM_ELEMS))
    
    @property
    def density(self):
        return self._density
    
    @density.setter
    def density(self, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            self._density = val
        else:
            self._density = val * np.ones((self.N_BEAM_ELEMS))
    
    @property
    def area(self):
        return self._area
    
    @area.setter
    def area(self, val):
        if isinstance(val, (list, tuple, np.ndarray)):
            self._area = val
        else:
            self._area = val * np.ones((self.N_BEAM_ELEMS)) 

    def update_model(self):
        """Must call this after changing model parameters"""
        # Nodes for beam
        self.N_BEAM_NODES = ((NODE_PER_BEAM_ELEM - 1) * self.N_BEAM_ELEMS) + 1
        # Nodes for ends of springs
        self.N_SUPP_NODES = len(self.supports)
        # Total nodes in system
        self.NUM_NODES = self.N_BEAM_NODES + self.N_SUPP_NODES
        
        # DOFs in beam
        self.N_BEAM_DOF = DOF_PER_NODE * self.N_BEAM_NODES
        # DOFs at spring ends
        self.N_SUPP_DOF = DOF_PER_NODE * self.N_SUPP_NODES
        # Total DOFs in system
        self.NUM_DOF = (DOF_PER_NODE * self.NUM_NODES)
        
        # x coords for nodes
        self.NODE_X_VALS = np.concatenate([[0.0], np.cumsum(self.L)])
        
        # x coords for elem centres
        self.ELEM_MID_X_VALS = (self.NODE_X_VALS[1:] + self.NODE_X_VALS[:-1]) / 2.0
        
        # USEFUL INDICES INTO DOFS
        self.LHS_ROT_IDX = 1
        self.RHS_ROT_IDX = self.N_BEAM_DOF - 1
        self.MID_DIS_IDX = (self.N_BEAM_DOF // 2) - 1 
        
        # Calculate restrained DOFs
        self._rest_dofs = np.concatenate((self._fix_dofs,
                                          np.arange(self.N_BEAM_DOF, self.NUM_DOF)))
        
        # Calculate I values after cracks are applied
        I_vals = self._I
        for crack in self.cracks:
            I_vals += self.calc_I_reduction(crack.x_dam, crack.delta)
        
        # Update beam M, K, and C matrices
        self.M_beam = _assemble_beam_mass_matr(self._density, self._area, self._L)
        self.K_beam = _assemble_beam_stiff_matr(self._E, I_vals, self._L)
        self.C_beam = _assemble_damping_matr(self.damp_ratio,
                                             self.M_beam,
                                             self.K_beam,
                                             0)
        
        # setup global M, K, and C matrices
        self.M_global = np.zeros((self.NUM_DOF, self.NUM_DOF))
        self.K_global = np.zeros((self.NUM_DOF, self.NUM_DOF))
        self.C_global = np.zeros((self.NUM_DOF, self.NUM_DOF))
        
        # assemble beams into global M, K, and C matrices
        self.M_global[:self.N_BEAM_DOF,:self.N_BEAM_DOF] += self.M_beam
        self.K_global[:self.N_BEAM_DOF,:self.N_BEAM_DOF] += self.K_beam
        self.C_global[:self.N_BEAM_DOF,:self.N_BEAM_DOF] += self.C_beam
        
        # assemble supports into global M, K, and C matrices
        for i, s in enumerate(self.supports):
            v_dofs = np.array([s.beam_node*DOF_PER_NODE, self.N_BEAM_DOF + (i*DOF_PER_NODE)]) 
            r_dofs = v_dofs + 1

            self.K_global[v_dofs[:,np.newaxis], v_dofs] += s.K_lin
            self.C_global[v_dofs[:,np.newaxis], v_dofs] += s.C_lin

            self.K_global[r_dofs[:,np.newaxis], r_dofs] += s.K_rot
            self.C_global[r_dofs[:,np.newaxis], r_dofs] += s.C_rot
        
        # apply boundary conditions
        self.Mg = self.apply_bound_conds(self.M_global, diag_val=1.0)
        self.Kg = self.apply_bound_conds(self.K_global, diag_val=1.0)
        self.Cg = self.apply_bound_conds(self.C_global, diag_val=1.0)
    
    def x_pos_to_node(self, x):
        return np.abs(self.NODE_X_VALS - x).argmin().astype(int)
    
    def node_to_dis_dof(self, node):
        return int(2*node)
    
    def node_to_rot_dof(self, node):
        return int(2*node) + 1

    def x_pos_to_dis_dof(self, x):
        node = self.x_pos_to_node(x)
        return self.node_to_dis_dof(node)
    
    def x_pos_to_rot_dof(self, x):
        node = self.x_pos_to_node(x)
        return self.node_to_rot_dof(node)
            
    def pin_node(self, node):
        dof = np.multiply(node, 2) # displacement DOF
        self._fix_dofs = np.append(self._fix_dofs, dof) 
    
    def fix_node(self, node):
        dof = np.multiply(node, 2)
        self._fix_dofs = np.append(self._fix_dofs, [dof, dof+1])
        
    def add_springdamper_support(self,
                                 beam_node,
                                 k_lin=0.0,
                                 k_rot=0.0,
                                 c_lin=0.0,
                                 c_rot=0.0):
        self.supports.append(SpringDamperSupport(beam_node=beam_node,
                                                 k_lin=k_lin,
                                                 k_rot=k_rot,
                                                 c_lin=c_lin,
                                                 c_rot=c_rot))
        
    def remove_spring_support(self, node):
        idx = np.searchsorted(self.supp_nodes, node)
        idx = idx[idx != len(self.supp_nodes)]

        self.supp_nodes = np.delete(self.supp_nodes, idx)
        self.supp_K_lin = np.delete(self.supp_K_lin, idx)
        self.supp_K_rot = np.delete(self.supp_K_rot, idx)
    
    def add_crack_damage(self, x_dam, delta):
        self.cracks.append(CrackDamage(x_dam, delta))
        
    def reset_crack_damage(self):
        self.cracks = []
        
    def calc_I_reduction(self, x_dam, delta):
        length_affected = 1.5 * self.depth  # length affected by the crack
        elem_to_crack = np.abs(self.ELEM_MID_X_VALS - x_dam) / length_affected
        # coefficient defines how Ir varies for a given delta
        coefficient = 1 - (self.depth - (delta * self.depth)) ** 3 / self.depth ** 3
        return self.I * np.where(elem_to_crack < 1.0, coefficient * (elem_to_crack-1.0), 0.0)
    
    def calc_modal_freqs(self):
        return calc_modal_freqs(self.Mg, self.Kg, len(self._rest_dofs))
    
    def apply_bound_conds(self, mat, diag_val=1.0):
        return _applyBoundaryCondsMatr(mat, self._rest_dofs.astype(int), diag_val=diag_val)
    
    def perform_dynamic_sim(self, P_axles, S_axles, vel, time_step=0.01, theta=1.6, contact_length=0.0):
        return _perform_dynamic_sim(self.Kg, self.Cg, self.Mg, self.NODE_X_VALS, self.NUM_DOF,
                                    P_axles, S_axles, vel, 
                                    time_step, theta, contact_length)

    def perform_static_sim(self, P_axles, S_axles, force_step=None, contact_length=0.0):
        if force_step == None:
            force_step = np.min(self.L)/10
        return _perform_static_sim(self.Kg, self.NODE_X_VALS, self.NUM_DOF, P_axles, S_axles, force_step, contact_length)