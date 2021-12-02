from typing import Dict, FrozenSet

import numpy as onp
import jax
import jax.numpy as jnp

BEVERIDGE_HINZE = {
    "gamma_xy": "beveridge-hinze",
    "exponent": "beveridge-hinze",
    "beta": "beveridge-hinze",
    "parameters": {"c": 0.545},
    "data": "beveridge-hinze",
    "sane_correction": False,
}

MODERN = {
    "gamma_xy": "mataga-nishimoto",
    "exponent": "slater",
    "beta": "linderberg",
    "data": "crc",
    "sane_correction": True,
}

Z = {
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}

Z_EFF = {
    "B": 2.25,
    "C": 3.25,
    "N1": 3.90,
    "N2": 4.25,
    "O1": 4.55,
    "O2": 4.9,
    "F": 5.55,
    "Si": 4.15,
    "P1": 4.80,
    "P2": 5.15,
    "S1": 5.45,
    "S2": 5.80,
    "Cl": 6.45,
    "Br": 7.95,
    "I": 7.95,
}

CHARGES = {
    "B": -1,
    "C": 0,
    "N1": 0,
    "N2": 1,
    "O1": 0,
    "O2": 1,
    "F": 1,
    "Si": 0,
    "P1": 0,
    "P2": 1,
    "S1": 0,
    "S2": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

ELECTRON_CONFIGURATIONS = {
    "B": {1: 2, 2: 3},
    "C": {1: 2, 2: 3},
    "N1": {1: 2, 2: 4},
    "N2": {1: 2, 2: 3},
    "O1": {1: 2, 2: 5},
    "O2": {1: 2, 2: 4},
    "F": {1: 2, 2: 5},
    "Si": {1: 2, 2: 8, 3: 3},
    "P1": {1: 2, 2: 8, 3: 4},
    "P2": {1: 2, 2: 8, 3: 3},
    "S1": {1: 2, 2: 8, 3: 5},
    "S2": {1: 2, 2: 8, 3: 4},
    "Cl": {1: 2, 2: 8, 3: 5},
    "Br": {1: 2, 2: 8, 3: 18, 4: 5},
    "I": {1: 2, 2: 8, 3: 18, 4: 18, 5: 5},
}

VE_COEFFICIENTS = {
    "C": [0.678456, 6.3395448, -16.4146245],
    "N1": [0.71540207, 7.68420836, -26.57495969],
    "N2": [0.84807, 7.924431, -20.51828063],
    "O1": [0.95156786, 6.4853475, -31.26981683],
    "O2": [0.85848248, 9.22105004, -31.88995163],
    "F": [1.20653125, 6.20289687, -31.23356273],
    "S1": [0.1513546, 8.30919074, -37.04744978],
    "S2": [0.14891127, 8.68808268, -31.70962956],
}

SYMBOLS = {
    "B": "B",
    "C": "C",
    "N1": "N",
    "N2": "N",
    "O1": "O",
    "O2": "O",
    "F": "F",
    "Si": "Si",
    "P1": "P",
    "P2": "P",
    "S1": "S",
    "S2": "S",
    "Cl": "Cl",
}

IP_BH = {
    "B": 1.06,
    "C": 11.16,
    "N1": 14.12,
    "N2": 28.71,
    "O1": 17.70,
    "O2": 34.08,
    "F": 40.70,
    "Si": 9.17,
    "P1": 11.64,
    "P2": 20.68,
    "S1": 12.70,
    "S2": 23.74,
    "Cl": 27.28,
}

IP_CRC = {
    "B": 1.011,
    "C": 11.164,
    "N1": 14.093,
    "N2": 28.717,
    "O1": 17.701,
    "O2": 34.122,
    "F": 40.697,
    "Si": 9.177,
    "P1": 11.146,
    "P2": 20.732,
    "S1": 12.706,
    "S2": 23.740,
    "Cl": 27.296,
}

EA_CRC = {
    "B": 0.115,
    "C": 0.168,
    "N1": 1.659,
    "N2": 11.956,
    "O1": 2.456,
    "O2": 15.305,
    "F": 18.519,
    "Si": 1.925,
    "P1": 1.776,
    "P2": 10.266,
    "S1": 2.764,
    "S2": 11.648,
    "Cl": 14.501,
}

N = {
    "B": 2,
    "C": 2,
    "N1": 2,
    "N2": 2,
    "O1": 2,
    "O2": 2,
    "F": 2,
    "Si": 2,
    "P1": 3,
    "P2": 3,
    "S1": 3,
    "S2": 3,
    "Cl": 3,
    "Br": 4,
    "I": 5,
}

EA_BH = {
    "B": -1.45,
    "C": 0.03,
    "N1": 1.78,
    "N2": 11.96,
    "O1": 2.47,
    "O2": 15.30,
    "F": 18.52,
    "Si": 2.00,
    "P1": 1.80,
    "P2": 10.76,
    "S1": 2.76,
    "S2": 11.65,
    "Cl": 14.51,
}

N_STAR = {
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 3.7,
    5: 4.0,
    6: 4.2,
}

N_ELECTRONS = {
    "B": 0,
    "C": 1,
    "Cl": 2,
    "F": 2,
    "N1": 1,
    "N2": 2,
    "O1": 1,
    "O2": 2,
    "P1": 1,
    "P2": 2,
    "S1": 1,
    "S2": 2,
    "Si": 1,
}

H_X = {
        "B": -0.45,
        "C": 0.00,
        "Cl": 1.48,
        "F": 2.71,
        "N1": 0.51,
        "N2": 1.37,
        "O1": 0.97,
        "O2": 2.09,
        "P1": 0.19,
        "P2": 0.75,
        "S1": 0.46,
        "S2": 1.11,
        "Si": 0.00,
        }

h_x_flat, h_x_tree = jax.tree_util.tree_flatten(H_X)
    # h_xy  = {'C':{'C':1.0,'Cl':1.,'F':1.33,'N1':0.7,'N2':0.9,'O1':0.99,'O2':0.34,'S2':0.68},
    #         'N1':{'N1':1.27}}

H_XY ={
frozenset(["B", "B"]): 0.87,
        frozenset(["B", "C"]): 0.73,
        frozenset(["B", "Cl"]): 0.41,
        frozenset(["B", "F"]): 0.26,
        frozenset(["B", "N1"]): 0.66,
        frozenset(["B", "N2"]): 0.53,
        frozenset(["B", "O1"]): 0.6,
        frozenset(["B", "O2"]): 0.35,
        frozenset(["B", "P1"]): 0.53,
        frozenset(["B", "P2"]): 0.54,
        frozenset(["B", "S1"]): 0.51,
        frozenset(["B", "S2"]): 0.44,
        frozenset(["B", "Si"]): 0.57,
        frozenset(["C", "C"]): 1.00,
        frozenset(["C", "Cl"]): 0.62,
        frozenset(["C", "F"]): 0.52,
        frozenset(["C", "N1"]): 1.02,
        frozenset(["C", "N2"]): 0.89,
        frozenset(["C", "O1"]): 1.06,
        frozenset(["C", "O2"]): 0.66,
        frozenset(["C", "P1"]): 0.77,
        frozenset(["C", "P2"]): 0.76,
        frozenset(["C", "S1"]): 0.81,
        frozenset(["C", "S2"]): 0.69,
        frozenset(["C", "Si"]): 0.75,
        frozenset(["Cl", "Cl"]): 0.68,
        frozenset(["Cl", "F"]): 0.51,
        frozenset(["Cl", "N1"]): 0.77,
        frozenset(["Cl", "N2"]): 0.80,
        frozenset(["Cl", "O1"]): 0.88,
        frozenset(["Cl", "O2"]): 0.70,
        frozenset(["Cl", "P1"]): 0.35,
        frozenset(["Cl", "P2"]): 0.55,
        frozenset(["Cl", "S1"]): 0.52,
        frozenset(["Cl", "S2"]): 0.59,
        frozenset(["Cl", "Si"]): 0.34,
        frozenset(["F", "F"]): 1.04,
        frozenset(["F", "N1"]): 0.65,
        frozenset(["F", "N2"]): 0.77,
        frozenset(["F", "O1"]): 0.92,
        frozenset(["F", "O2"]): 0.94,
        frozenset(["F", "P1"]): 0.21,
        frozenset(["F", "P2"]): 0.22,
        frozenset(["F", "S1"]): 0.28,
        frozenset(["F", "S2"]): 0.32,
        frozenset(["F", "Si"]): 0.17,
        frozenset(["N1", "N1"]): 1.09,
        frozenset(["N1", "N2"]): 0.99,
        frozenset(["N1", "O1"]): 1.14,
        frozenset(["N1", "O2"]): 0.80,
        frozenset(["N1", "P1"]): 0.78,
        frozenset(["N1", "P2"]): 0.81,
        frozenset(["N1", "S1"]): 0.83,
        frozenset(["N1", "S2"]): 0.78,
        frozenset(["N1", "Si"]): 0.72,
        frozenset(["N2", "N2"]): 0.98,
        frozenset(["N2", "O1"]): 1.13,
        frozenset(["N2", "O2"]): 0.89,
        frozenset(["N2", "P1"]): 0.55,
        frozenset(["N2", "P2"]): 0.64,
        frozenset(["N2", "S1"]): 0.68,
        frozenset(["N2", "S2"]): 0.73,
        frozenset(["N2", "Si"]): 0.43,
        frozenset(["O1", "O1"]): 1.26,
        frozenset(["O1", "O2"]): 1.02,
        frozenset(["O1", "P1"]): 0.75,
        frozenset(["O1", "P2"]): 0.82,
        frozenset(["O1", "S1"]): 0.84,
        frozenset(["O1", "S2"]): 0.85,
        frozenset(["O1", "Si"]): 0.65,
        frozenset(["O2", "O2"]): 0.95,
        frozenset(["O2", "P1"]): 0.31,
        frozenset(["O2", "P2"]): 0.39,
        frozenset(["O2", "S1"]): 0.43,
        frozenset(["O2", "S2"]): 0.54,
        frozenset(["O2", "Si"]): 0.24,
        frozenset(["P1", "P1"]): 0.63,
        frozenset(["P1", "P2"]): 0.58,
        frozenset(["P1", "S1"]): 0.65,
        frozenset(["P1", "S2"]): 0.48,
        frozenset(["P1", "Si"]): 0.62,
        frozenset(["P2", "P2"]): 0.63,
        frozenset(["P2", "S1"]): 0.65,
        frozenset(["P2", "S2"]): 0.60,
        frozenset(["P2", "Si"]): 0.52,
        frozenset(["S1", "S1"]): 0.68,
        frozenset(["S1", "S2"]): 0.58,
        frozenset(["S1", "Si"]): 0.61,
        frozenset(["S2", "S2"]): 0.63,
        frozenset(["S2", "Si"]): 0.40,
        frozenset(["Si", "Si"]): 0.64,
    }
h_xy_flat, h_xy_tree = jax.tree_util.tree_flatten(H_XY)

# R_XY = H_XY
Y_XY = H_XY
y_xy_flat, y_xy_tree = jax.tree_util.tree_flatten(Y_XY)
Y_XY = jax.tree_util.tree_unflatten(y_xy_tree,0.3*jnp.ones((len(y_xy_flat),)))


# J. Chem. Soc. Perkin Trans. II, S1–S19, 1987.
R_XY ={
frozenset(["B", "B"]): 1.775,
        frozenset(["B", "C"]): 1.716,
        frozenset(["B", "Cl"]): 1.751,
        frozenset(["B", "F"]): 1.366,
        frozenset(["B", "N1"]): 1.611,
        frozenset(["B", "N2"]): 1.611,
        frozenset(["B", "O1"]): 1.468,
        frozenset(["B", "O2"]): 1.367,
        frozenset(["B", "P1"]): 1.922, 
        frozenset(["B", "P2"]): 1.922, 
        frozenset(["B", "S1"]): 1.930,
        frozenset(["B", "S2"]): 1.896,
        frozenset(["B", "Si"]): 1., #not reported
        frozenset(["C", "C"]): 1.455,
        frozenset(["C", "Cl"]): 1.734,
        frozenset(["C", "F"]): 1.340,
        frozenset(["C", "N1"]): 1.336,
        frozenset(["C", "N2"]): 1.339,
        frozenset(["C", "O1"]): 1.333,
        frozenset(["C", "O2"]): 1.354,
        frozenset(["C", "P1"]): 1.800,
        frozenset(["C", "P2"]): 1.806,
        frozenset(["C", "S1"]): 1.808,
        frozenset(["C", "S2"]): 1.789,
        frozenset(["C", "Si"]): 1.857,
        frozenset(["Cl", "Cl"]): 2.306,
        frozenset(["Cl", "F"]): 1.628,
        frozenset(["Cl", "N1"]): 1.611,
        frozenset(["Cl", "N2"]): 1.611,
        frozenset(["Cl", "O1"]): 1.414,
        frozenset(["Cl", "O2"]): 1.414,
        frozenset(["Cl", "P1"]): 1.997,
        frozenset(["Cl", "P2"]): 2.008,
        frozenset(["Cl", "S1"]): 2.072,
        frozenset(["Cl", "S2"]): 2.072,
        frozenset(["Cl", "Si"]): 2.072,
        frozenset(["F", "F"]): 1.412,
        frozenset(["F", "N1"]): 1.317,
        frozenset(["F", "N2"]): 1.317,
        frozenset(["F", "O1"]): 1.354,
        frozenset(["F", "O2"]): 1.354,
        frozenset(["F", "P1"]): 1.579,
        frozenset(["F", "P2"]): 1.495,
        frozenset(["F", "S1"]): 1.527,
        frozenset(["F", "S2"]): 1.527,
        frozenset(["F", "Si"]): 1.588,
        frozenset(["N1", "N1"]): 1.414,
        frozenset(["N1", "N2"]): 1.124,
        frozenset(["N1", "O1"]): 1.378,
        frozenset(["N1", "O2"]): 1.218,
        frozenset(["N1", "P1"]): 1.730,
        frozenset(["N1", "P2"]): 1.730,
        frozenset(["N1", "S1"]): 1.748,
        frozenset(["N1", "S2"]): 1.748,
        frozenset(["N1", "Si"]): 1.711,
        frozenset(["N2", "N2"]): 1.245,
        frozenset(["N2", "O1"]): 1.218,
        frozenset(["N2", "O2"]): 1.212,
        frozenset(["N2", "P1"]): 1.683,
        frozenset(["N2", "P2"]): 1.652,
        frozenset(["N2", "S1"]): 1.600,
        frozenset(["N2", "S2"]): 1.748,
        frozenset(["N2", "Si"]): 1.748,
        frozenset(["O1", "O1"]): 1.464,
        frozenset(["O1", "O2"]): 1.464,
        frozenset(["O1", "P1"]): 1.513,
        frozenset(["O1", "P2"]): 1.489,
        frozenset(["O1", "S1"]): 1.497,
        frozenset(["O1", "S2"]): 1.497,
        frozenset(["O1", "Si"]): 1.663,
        frozenset(["O2", "O2"]): 1.464,
        frozenset(["O2", "P1"]): 1.621,
        frozenset(["O2", "P2"]): 1.573,
        frozenset(["O2", "S1"]): 1.436,
        frozenset(["O2", "S2"]): 1.577,
        frozenset(["O2", "Si"]): 1.663,
        frozenset(["P1", "P1"]): 2.256,
        frozenset(["P1", "P2"]): 2.256,
        frozenset(["P1", "S1"]): 1.954,
        frozenset(["P1", "S2"]): 1.922,
        frozenset(["P1", "Si"]): 2.264,
        frozenset(["P2", "P2"]): 2.133,
        frozenset(["P2", "S1"]): 1.922,
        frozenset(["P2", "S2"]): 1.954,
        frozenset(["P2", "Si"]): 2.264,
        frozenset(["S1", "S1"]): 2.051,
        frozenset(["S1", "S2"]): 2.051,
        frozenset(["S1", "Si"]): 2.145,
        frozenset(["S2", "S2"]): 2.051,
        frozenset(["S2", "Si"]): 2.145,
        frozenset(["Si", "Si"]): 2.359,
    }


# Values taken from https://cccbdb.nist.gov/diatomicexpbondx.asp
R_XY_diatomics_NIST ={
frozenset(["B", "B"]): 1.590,
        frozenset(["B", "C"]): 1.491,
        frozenset(["B", "Cl"]): 1.719,
        frozenset(["B", "F"]): 1.267,
        frozenset(["B", "N1"]): 1.325,
        frozenset(["B", "N2"]): 1.325,
        frozenset(["B", "O1"]): 1.205,
        frozenset(["B", "O2"]): 1.205,
        frozenset(["B", "P1"]): 1., #not reported
        frozenset(["B", "P2"]): 1., #not reported
        frozenset(["B", "S1"]): 1.609,
        frozenset(["B", "S2"]): 1.609,
        frozenset(["B", "Si"]): 1., #not reported
        frozenset(["C", "C"]): 1.243,
        frozenset(["C", "Cl"]): 1.649,
        frozenset(["C", "F"]): 1.276,
        frozenset(["C", "N1"]): 1.172,
        frozenset(["C", "N2"]): 1.172,
        frozenset(["C", "O1"]): 1.128,
        frozenset(["C", "O2"]): 1.128,
        frozenset(["C", "P1"]): 1.562,
        frozenset(["C", "P2"]): 1.562,
        frozenset(["C", "S1"]): 1.535,
        frozenset(["C", "S2"]): 1.535,
        frozenset(["C", "Si"]): 1.722,
        frozenset(["Cl", "Cl"]): 1.988,
        frozenset(["Cl", "F"]): 1.628,
        frozenset(["Cl", "N1"]): 1.611,
        frozenset(["Cl", "N2"]): 1.611,
        frozenset(["Cl", "O1"]): 1.596,
        frozenset(["Cl", "O2"]): 1.596,
        frozenset(["Cl", "P1"]): 2.018,
        frozenset(["Cl", "P2"]): 2.018,
        frozenset(["Cl", "S1"]): 1.975,
        frozenset(["Cl", "S2"]): 1.975,
        frozenset(["Cl", "Si"]): 2.061,
        frozenset(["F", "F"]): 1.412,
        frozenset(["F", "N1"]): 1.317,
        frozenset(["F", "N2"]): 1.317,
        frozenset(["F", "O1"]): 1.354,
        frozenset(["F", "O2"]): 1.354,
        frozenset(["F", "P1"]): 1.593,
        frozenset(["F", "P2"]): 1.593,
        frozenset(["F", "S1"]): 1.599,
        frozenset(["F", "S2"]): 1.599,
        frozenset(["F", "Si"]): 1.604,
        frozenset(["N1", "N1"]): 1.098,
        frozenset(["N1", "N2"]): 1.098,
        frozenset(["N1", "O1"]): 1.154,
        frozenset(["N1", "O2"]): 1.154,
        frozenset(["N1", "P1"]): 1.491,
        frozenset(["N1", "P2"]): 1.491,
        frozenset(["N1", "S1"]): 1.497,
        frozenset(["N1", "S2"]): 1.497,
        frozenset(["N1", "Si"]): 1.575,
        frozenset(["N2", "N2"]): 1.098,
        frozenset(["N2", "O1"]): 1.154,
        frozenset(["N2", "O2"]): 1.154,
        frozenset(["N2", "P1"]): 1.491,
        frozenset(["N2", "P2"]): 1.491,
        frozenset(["N2", "S1"]): 1.497,
        frozenset(["N2", "S2"]): 1.497,
        frozenset(["N2", "Si"]): 1.575,
        frozenset(["O1", "O1"]): 1.208,
        frozenset(["O1", "O2"]): 1.208,
        frozenset(["O1", "P1"]): 1.476,
        frozenset(["O1", "P2"]): 1.476,
        frozenset(["O1", "S1"]): 1.481,
        frozenset(["O1", "S2"]): 1.481,
        frozenset(["O1", "Si"]): 1.510,
        frozenset(["O2", "O2"]): 1.208,
        frozenset(["O2", "P1"]): 1.476,
        frozenset(["O2", "P2"]): 1.476,
        frozenset(["O2", "S1"]): 1.481,
        frozenset(["O2", "S2"]): 1.481,
        frozenset(["O2", "Si"]): 1.510,
        frozenset(["P1", "P1"]): 1.893,
        frozenset(["P1", "P2"]): 1.893,
        frozenset(["P1", "S1"]): 1.900,
        frozenset(["P1", "S2"]): 1.900,
        frozenset(["P1", "Si"]): 2.078,
        frozenset(["P2", "P2"]): 1.893,
        frozenset(["P2", "S1"]): 1.900,
        frozenset(["P2", "S2"]): 1.900,
        frozenset(["P2", "Si"]): 2.078,
        frozenset(["S1", "S1"]): 1.889,
        frozenset(["S1", "S2"]): 1.889,
        frozenset(["S1", "Si"]): 1.929,
        frozenset(["S2", "S2"]): 1.889,
        frozenset(["S2", "Si"]): 1.929,
        frozenset(["Si", "Si"]): 2.246,
    }

def update_h_x(h_x):
    xc = h_x['C']
    xc_tree = jax.tree_unflatten(h_x_tree,xc*jnp.ones_like(jnp.array(h_x_flat)))
    f_dif_pytrees = lambda x, y: y-x
    return jax.tree_map(f_dif_pytrees,xc_tree, h_x)


def update_h_xy(h_xy):
    key = frozenset(['C', 'C'])
    xcc = h_xy[key]
    xcc_tree = jax.tree_unflatten(h_xy_tree,xcc*jnp.ones_like(jnp.array(h_xy_flat)))
    f_div_pytrees = lambda x, y: y/x
    return jax.tree_map(f_div_pytrees,xcc_tree, h_xy)
