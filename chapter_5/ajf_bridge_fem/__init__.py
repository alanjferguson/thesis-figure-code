import numpy as np

import BeamWithSprings as bm

################################################################################
# Gracehill
################################################################################

GH_side_span_length = 11.4  # m
GH_center_span_length = 19.0  # m
GH_bridge_length = GH_center_span_length + 2.0 * GH_side_span_length

GH_bridge_width = 8 # m
GH_slab_depth = 0.65 # m

GH_slab_area = GH_bridge_width * GH_slab_depth
GH_slab_I = (GH_bridge_width * GH_slab_depth**3.0) / 12.0

GH_slab_E = 30e9
GH_slab_rho = 2400 # kg m^-3

# Column properties
GH_col_E = None # will take E value from deck
GH_col_I = (3.1 * 0.9**3.0) / 12.0
GH_col_area = 3.1 * 0.9
GH_col_L = 2.5
GH_col_rho = 2400.0

# Damping (taken from modal test)
GH_bridge_damping = 0.018


def construct_model_GH(elem_length=1.0):
    # Setup beam object
    n_side_span_elems = int(GH_side_span_length // elem_length)
    n_side_span_elems += n_side_span_elems % 2  # ensures even no. of elems

    n_cent_span_elems = int(GH_center_span_length // elem_length)
    n_cent_span_elems += n_cent_span_elems % 2  # ensures even no. of elems

    L1 = np.linspace(
        0.0,
        GH_side_span_length,
        n_side_span_elems + 1,
    )
    L2 = np.linspace(
        GH_side_span_length,
        GH_side_span_length + GH_center_span_length,
        n_cent_span_elems + 1,
    )
    L3 = L1 + GH_side_span_length + GH_center_span_length
    L = np.diff(np.concatenate([L1, L2[1:], L3[1:]]))

    beam = bm.Beam(len(L), L)

    beam.E = GH_slab_E
    beam.I = GH_slab_I
    beam.area = GH_slab_area
    beam.density = GH_slab_rho
    beam.depth = GH_slab_depth



    beam.damp_ratio = GH_bridge_damping

    ## ADD ROT. SPRINGS AT PIERS
    for x in [GH_side_span_length, GH_side_span_length + GH_center_span_length]:
        beam.add_column_support(
            beam_node=beam.x_pos_to_node(x),
            E_column=GH_col_E,
            I_column=GH_col_I,
            A_column=GH_col_area,
            L_column=GH_col_L,
            rho_column=GH_col_rho,
        )

    # PIN ENDS
    beam.pin_node(beam.x_pos_to_node(0))  # LHS END
    beam.pin_node(beam.x_pos_to_node(GH_bridge_length))  # RHS END

    beam.update_model()

    return beam

################################################################################
# Line Bridge
################################################################################

LB_span_length = 34.0

# TODO: recreate calcs to compute these values
LB_deck_E = 30e9
LB_deck_I = 6.0
LB_deck_area = 10.0
LB_deck_depth = 2.2
LB_deck_rho = 2400.0

# Damping taken from modal test
LB_damping = 0.021

def construct_model_LB(elem_len=1.0):
    n_elems = int(LB_span_length // elem_len)
    n_elems += n_elems % 2  # ensures even no. of elems
    L = np.diff(np.linspace(
        0.0,
        LB_span_length,
        n_elems + 1,
    ))
    
    beam = bm.Beam(len(L), L)

    beam.E = LB_deck_E
    beam.I = LB_deck_I 
    beam.density = LB_deck_rho
    beam.area = LB_deck_area
    beam.depth = LB_deck_depth
    beam.damp_ratio = LB_damping

    # PIN ENDS
    beam.pin_node(beam.x_pos_to_node(0))  # LHS END
    beam.pin_node(beam.x_pos_to_node(np.sum(beam.L)))  # RHS END

    beam.update_model()

    return beam
