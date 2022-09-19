import os
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import jit

import numpy as onp

from huxel.molecule import myMolecule
from huxel.parameters import H_X, H_XY, N_ELECTRONS
from huxel.parameters import R_XY_Bohr, R_XY_AA
from huxel.parameters import Y_XY_Bohr, Y_XY_AA
from huxel.parameters import h_x_tree, h_x_flat, h_xy_tree, h_xy_flat, r_xy_tree, r_xy_flat
from huxel.parameters import f_dif_pytrees, f_div_pytrees, f_mult_pytrees, f_sum_pytrees
from huxel.parameters import au_to_eV, Bohr_to_AA

PRNGKey = Any


# --------------------------------
#     PARAMETERS
def load_pre_opt_params(files: dict) -> Tuple:
    """Load parameters from a file

    Args:
        files (dict): file's name 

    Returns:
        Tuple: number of epochs, loss function value, validation loss function 
    """
    if os.path.isfile(files["f_loss_opt"]):
        D = onp.load(files["f_loss_opt"], allow_pickle=True)
        epochs = D.item()["epoch"]
        loss_tr = D.item()["loss_tr"]
        loss_val = D.item()["loss_val"]
        return epochs, loss_tr, loss_val


def random_pytrees(_pytree: dict, key: PRNGKey, minval: float = -1.0, maxval: float = 1.0) -> Tuple:
    """Generate a pytree with random values [minval,maxval]

    Args:
        _pytree (dict): base pytree
        key (PRNGKey): a PRNG key used as the random key.
        minval (float, optional): minimum allowed value. Defaults to -1.0.
        maxval (float, optional): maximum allowed value. Defaults to 1.0.

    Returns:
        Tuple: random pytree, new PRNG key
    """
    _pytree_flat, _pytree_tree = jax.tree_util.tree_flatten(_pytree)
    _pytree_random_flat = jax.random.uniform(
        key, shape=(len(_pytree_flat),), minval=minval, maxval=maxval
    )
    _new_pytree = jax.tree_util.tree_unflatten(
        _pytree_tree, _pytree_random_flat)
    _, subkey = jax.random.split(key)
    return _new_pytree, subkey


def get_init_params_homo_lumo() -> Tuple:
    """Initial parameters for linear transformation for HOMO-LUMO gap. Obtained from a linear fit.

    Returns:
        Tuple: parameters
    """
    alpha = jnp.array([-2.252276274030775])
    beta = jnp.array([2.053257355175381])  # params_lr.item()["beta"]
    return jnp.array(alpha), jnp.array(beta)


def get_init_params_polarizability() -> Tuple:
    """Initial parameters for linear transformation for polarizability. Obtained from a linear fit.

    Returns:
        Tuple: parameters
    """
    # params_lr = onp.load("huxel/data/lr_params.npy", allow_pickle=True)
    alpha = jnp.ones(1)
    # jnp.array([116.85390527250595]) #params_lr.item()["beta"]
    beta = jnp.zeros(1)
    return jnp.array(alpha), jnp.array(beta)


def get_y_xy_random(key: PRNGKey) -> Tuple:
    """Random parameters for atom-atom parameters

    Args:
        key (PRNGKey): a PRNG key used as the random key.

    Returns:
        Tuple: random pytree
    """
    y_xy_flat, y_xy_tree = jax.tree_util.tree_flatten(Y_XY_AA)
    y_xy_random_flat = jax.random.uniform(
        key, shape=(len(y_xy_flat),), minval=-0.1, maxval=0.1
    )
    y_xy_random_flat = y_xy_random_flat + 0.3
    _, subkey = jax.random.split(key)
    y_xy_random = jax.tree_util.tree_unflatten(y_xy_tree, y_xy_random_flat)
    return y_xy_random, subkey


def get_params_pytrees(hl_a: float, hl_b: float, pol_a: float, pol_b: float, h_x: dict, h_xy: dict, r_xy: dict, y_xy: dict) -> Any:
    """Full set of parameters

    Args:
        hl_a (float): HOMO-LUMO linear parameter
        hl_b (float): HOMO-LUMO linear parameter
        pol_a (float): Polarizability linear parameter
        pol_b (float): Polarizability linear parameter
        h_x (dict): Hückel model diagonal parameter (energy of an electron in a 2p orbital)
        h_xy (dict): Hückel model of diagonal parameter (energy of an electron in the bond i-j)
        r_xy (dict): distance-dependence parameter
        y_xy (dict): length-scale parameter 

    Returns:
        Any: parameters
    """
    params_init = {
        "hl_params": {"a": hl_a, "b": hl_b},
        "pol_params": {"a": pol_a, "b": pol_b},
        "h_x": h_x,
        "h_xy": h_xy,
        "r_xy": r_xy,
        "y_xy": y_xy,
    }
    return params_init


# include alpha y beta in the new parameters
def get_default_params(observable: str = "homo_lumo") -> Any:
    """Load literature parameters 

    Args:
        observable (str, optional): target observable. Defaults to "homo_lumo".

    Returns:
        Any: parameters pytree
    """
    params_hl = get_init_params_homo_lumo()  # homo_lumo
    params_pol = get_init_params_polarizability()  # (jnp.ones(1), jnp.ones(1))

    if observable.lower() == 'homo_lumo' or observable.lower() == 'hl':
        R_XY = R_XY_AA
        Y_XY = Y_XY_AA
    elif observable.lower() == 'polarizability' or observable.lower() == 'pol':
        R_XY = R_XY_Bohr
        Y_XY = Y_XY_Bohr

    return get_params_pytrees(params_hl[0], params_hl[1], params_pol[0], params_pol[1], H_X, H_XY, R_XY, Y_XY)


def get_params_bool(params_wdecay_: dict) -> dict:
    """pytree of booleans where weight decay will be used. array used in masks in OPTAX

    Args:
        params_wdecay_ (dict): parameters pytree 

    Returns:
        dict: parameters pytree 
    """
    params = get_default_params()

    params_bool = params
    params_flat, params_tree = jax.tree_util.tree_flatten(params)
    params_bool = jax.tree_util.tree_unflatten(
        params_tree, jnp.zeros(len(params_flat), dtype=bool)
    )  # all FALSE

    if 'all' in params_wdecay_:
        # all True
        return jax.tree_util.tree_unflatten(params_tree, jnp.ones(len(params_flat), dtype=bool))
    else:
        for pb in params_wdecay_:  # ONLY TRUE
            if isinstance(params[pb], dict):
                p_flat, p_tree = jax.tree_util.tree_flatten(params[pb])
                params_bool[pb] = jax.tree_util.tree_unflatten(
                    p_tree, jnp.ones(len(p_flat), dtype=bool)
                )
            else:
                params_bool[pb] = jnp.ones(params[pb].shape, dtype=bool)

        else:
            return params_bool


def get_random_params(files: dict, key: PRNGKey) -> Tuple:
    """Random parameters if file does not exists

    Args:
        files (dict): file name where parameters are saved
        key (PRNGKey): a PRNG key used as the random key.

    Returns:
        Tuple: parameters pytree, new PRNG key
    """
    if not os.path.isfile(files["f_w"]):
        params_init = get_default_params()
        # params_lr,params_coulson = params_init

        hl_a_random = jax.random.uniform(
            key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        hl_b_random = jax.random.uniform(
            subkey, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(subkey)

        pol_a_random = jax.random.uniform(
            key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        pol_b_random = jax.random.uniform(
            subkey, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(subkey)

        h_x = params_init["h_x"]
        h_x_random, subkey = random_pytrees(h_x, subkey, -1.0, 1.0)

        h_xy = params_init["h_xy"]
        h_xy_random, subkey = random_pytrees(h_xy, subkey, 0.0, 1.0)

        r_xy = params_init["r_xy"]
        r_xy_random, subkey = random_pytrees(r_xy, subkey, 1.0, 3.0)

        y_xy = params_init["y_xy"]
        y_xy_random, subkey = get_y_xy_random(subkey)

        params = get_params_pytrees(
            hl_a_random, hl_b_random, pol_a_random, pol_b_random, h_x_random, h_xy_random, r_xy_random, y_xy_random
        )

        f = open(files["f_out"], "a+")
        print("Random initial parameters", file=f)
        print("-----------------------------------", file=f)
        f.close()
        return params, subkey
    else:
        params = get_init_params(files)
        return params, key


def get_init_params(files: dict, obs: str = "homo_lumo") -> Any:
    """Initial parameters for target observable

    Args:
        files (dict): dictionary with file's name
        obs (str, optional): target observable. Defaults to "homo_lumo".

    Returns:
        Any: _description_
    """
    params_init = get_default_params()
    if os.path.isfile(files["f_w"]):
        params = onp.load(files["f_w"], allow_pickle=True)
        print(files["f_w"])
        # params_lr,params_coulson = params
        hl_a = params.item()["hl_params"]["a"]
        hl_b = params.item()["hl_params"]["b"]
        pol_a = params.item()["pol_params"]["a"]
        pol_b = params.item()["pol_params"]["b"]

        h_x = params.item()["h_x"]
        h_xy = params.item()["h_xy"]
        r_xy = params.item()["r_xy"]
        y_xy = params.item()["y_xy"]

        params = get_params_pytrees(
            hl_a, hl_b, pol_a, pol_b, h_x, h_xy, r_xy, y_xy)

        f = open(files["f_out"], "a+")
        print("Reading parameters from prev. optimization", file=f)
        print("-----------------------------------", file=f)
        f.close()

        return params
    else:
        f = open(files["f_out"], "a+")
        print("Standard initial parameters", file=f)
        print("-----------------------------------", file=f)
        f.close()
        return params_init


def get_external_field(observable: str = 'homo_lumo', magnitude: Any = 0.) -> Any:
    """External field value

    Args:
        observable (str, optional): target observable. Defaults to 'homo_lumo'.
        magnitude (Any, optional): external field magnitude. Defaults to 0..

    Returns:
        Any: external field
    """
    if observable.lower() == 'polarizability' or observable.lower() == 'pol':
        if isinstance(magnitude, float):
            return magnitude*jnp.ones(3)
        elif isinstance(magnitude, list):
            return jnp.asarray(magnitude)
        else:  # default
            return jnp.zeros(3)
    else:
        return None


@jit
def update_h_x(h_x: dict) -> dict:
    """Normalization of the Hückel model diagonal parameters with respect to C atom

    Args:
        h_x (dict): pytree

    Returns:
        dict: parameters normalized with respect to C atom parameter
    """
    xc = h_x["C"]
    xc_tree = jax.tree_unflatten(
        h_x_tree, xc * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_dif_pytrees, xc_tree, h_x)


@jit
def update_h_xy(h_xy: dict) -> dict:
    """Normalization of the Hückel model of diagonal parameters with respect to C-C atoms parameter

    Args:
        h_xy (dict): pytree

    Returns:
        dict: parameters normalized with respect to C-C atoms parameter
    """
    key = frozenset(["C", "C"])
    xcc = h_xy[key]
    xcc_tree = jax.tree_unflatten(
        h_xy_tree, xcc * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_div_pytrees, xcc_tree, h_xy)


@jit
def update_h_x_au_to_eV(h_x: dict, pol_a: Any) -> dict:
    """Unit conversion to a.u. to eV

    Args:
        h_x (dict): parameters
        pol_a (Any): conversion value

    Returns:
        dict: parameters
    """
    x_tree = jax.tree_unflatten(
        h_x_tree, (pol_a/au_to_eV) * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_mult_pytrees, x_tree, h_x)


@jit
def update_h_xy_au_to_eV(h_xy: dict, pol_a: Any) -> dict:
    """Unit conversion to a.u. to eV

    Args:
        h_x (dict): parameters
        pol_a (Any): conversion value

    Returns:
        dict: parameters
    """
    xy_tree = jax.tree_unflatten(
        h_xy_tree, (pol_a/au_to_eV) * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_mult_pytrees, xy_tree, h_xy)


@jit
def update_r_xy_Bohr_to_AA(r_xy: dict) -> dict:
    """Unit conversion to Bohr to Armstrong for distance dependence parameters

    Args:
        r_xy (dict): parameters

    Returns:
        dict: parameters
    """
    xy_tree = jax.tree_unflatten(
        r_xy_tree, (Bohr_to_AA) * jnp.ones_like(jnp.array(r_xy_flat)))
    return jax.tree_map(f_div_pytrees, xy_tree, r_xy)


@jit
def normalize_params_wrt_C(params: dict) -> dict:
    """Normalization of the Hückel model with respect to C atom parameter

    Args:
        params (dict): parameters

    Returns:
        dict: normalized parameters
    """
    h_x = update_h_x(params["h_x"])
    h_xy = update_h_xy(params["h_xy"])

    new_params = get_params_pytrees(
        params["hl_params"]["a"],
        params["hl_params"]["b"],
        params["pol_params"]["a"],
        params["pol_params"]["b"],
        h_x,
        h_xy,
        params["r_xy"],
        params["y_xy"],
    )
    return new_params


@jit
def normalize_params_polarizability(params: dict) -> dict:
    """Normalization of the Hückel model with respect to C atom parameter, for polarizability

    Args:
        params (dict): parameters

    Returns:
        dict: normalized parameters
    """
    # params_norm_c = normalize_params_wrt_C(params)
    params_norm_c = params
    pol_a = params_norm_c["pol_params"]["a"]

    h_x = update_h_x_au_to_eV(params_norm_c["h_x"], pol_a)
    h_xy = update_h_xy_au_to_eV(params_norm_c["h_xy"], pol_a)

    new_params = get_params_pytrees(
        params_norm_c["hl_params"]["a"],
        params_norm_c["hl_params"]["b"],
        params_norm_c["pol_params"]["a"],
        params_norm_c["pol_params"]["b"],
        h_x,
        h_xy,
        params_norm_c["r_xy"],
        params_norm_c["y_xy"],
    )
    return new_params
