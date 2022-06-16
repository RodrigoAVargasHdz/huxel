import os
import datetime
from typing import Any

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
#     FILES
def get_r_dir_old(method:str):
    if method == "exp":
        return "./Results_xyz/"
    elif method == "linear":
        return "./Results_xyz_linear/"
    elif method == "constant":
        return "./Results_xyz_constant/"
    elif method == "exp_freezeR":
        return "./Results_xyz_freezeR/"
    elif method == "randW":
        return "./Results_xyz_constant_random_params/"


def get_r_dir(method:str, bool_randW:bool):
    if bool_randW:
        r_dir = "./Results_{}_randW/".format(method)
    else:
        r_dir = "./Results_{}/".format(method)

    if not os.path.exists(r_dir):
        os.mkdir(r_dir)
    return r_dir


def get_files_names(obs:str, N:int, l:int, beta:str, randW:bool, opt_name:str="Adam"):
    # r_dir = './Results_xyz/'
    r_dir = get_r_dir(beta, randW)

    f_job = "huckel_{}_N_{}_l_{}_{}".format(obs, N, l, opt_name)
    f_out = "{}/out_{}.txt".format(r_dir, f_job)
    f_w = "{}/parameters_{}.npy".format(r_dir, f_job)
    f_pred = "{}/Prediction_{}.npy".format(r_dir, f_job)
    f_data = "{}/Data_{}.npy".format(r_dir, f_job)
    f_loss_opt = "{}/Loss_tr_val_itr_{}.npy".format(r_dir, f_job)

    files = {
        "f_job": f_job,
        "f_out": f_out,
        "f_w": f_w,
        "f_pred": f_pred,
        "f_data": f_data,
        "f_loss_opt": f_loss_opt,
        "r_dir": r_dir,
        "obs": obs,
    }
    return files


def get_params_file_itr(files:dict, itr:int):
    # r_dir = './Results_xyz/'
    f_job = files["f_job"]
    r_dir = files["r_dir"]
    file_ = "{}/params_{}_itr_{}.npy".format(r_dir, f_job, itr)
    return file_


# --------------------------------
#     HEAD OF FILE
def print_head(
    files:dict, obs:str, N:int, l:int, lr:float, w_decay:Any, n_epochs:int, batch_size:int, opt_name:str, beta:str, list_Wdecay:list
):
    f = open(files["f_out"], "a+")
    print("-----------------------------------", file=f)
    print("Starting time", file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print("-----------------------------------", file=f)
    print(files["f_out"], file=f)
    print("Observable = {}".format(obs), file=f)
    print("N = {}, l = {}".format(N, l), file=f)
    print("lr = {}, w decay = {}".format(lr, w_decay), file=f)
    print("batch size = {}".format(batch_size), file=f)
    print("N Epoch = {}".format(n_epochs), file=f)
    print("Opt method = {}".format(opt_name), file=f)
    print("f beta: {}".format(beta), file=f)
    print("W Decay {}: ".format(list_Wdecay), file=f)
    print("-----------------------------------", file=f)
    f.close()


#     TAIL OF FILE
def print_tail(files:dict):
    f = open(files["f_out"], "a+")
    print("-----------------------------------", file=f)
    print("Finish time", file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print("-----------------------------------", file=f)
    f.close()

# --------------------------------
#     PARAMETERS
def load_pre_opt_params(files:dict):
    if os.path.isfile(files["f_loss_opt"]):
        D = onp.load(files["f_loss_opt"], allow_pickle=True)
        epochs = D.item()["epoch"]
        loss_tr = D.item()["loss_tr"]
        loss_val = D.item()["loss_val"]
        return epochs, loss_tr, loss_val


def random_pytrees(_pytree:dict, key:PRNGKey, minval:float=-1.0, maxval:float=1.0):
    _pytree_flat, _pytree_tree = jax.tree_util.tree_flatten(_pytree)
    _pytree_random_flat = jax.random.uniform(
        key, shape=(len(_pytree_flat),), minval=minval, maxval=maxval
    )
    _new_pytree = jax.tree_util.tree_unflatten(_pytree_tree, _pytree_random_flat)
    _, subkey = jax.random.split(key)
    return _new_pytree, subkey


def get_init_params_homo_lumo():
    # params_lr = onp.load("huxel/data/lr_params.npy", allow_pickle=True)
    alpha = jnp.array([-2.252276274030775]) #params_lr.item()["alpha"] * jnp.ones(1)
    beta = jnp.array([2.053257355175381]) #params_lr.item()["beta"]
    return jnp.array(alpha), jnp.array(beta)


def get_init_params_polarizability():
    # params_lr = onp.load("huxel/data/lr_params.npy", allow_pickle=True)
    alpha = jnp.ones(1)
    beta = jnp.array([116.85390527250595]) #params_lr.item()["beta"]
    return jnp.array(alpha), jnp.array(beta)


def get_y_xy_random(key:PRNGKey):
    y_xy_flat, y_xy_tree = jax.tree_util.tree_flatten(Y_XY_AA)
    y_xy_random_flat = jax.random.uniform(
        key, shape=(len(y_xy_flat),), minval=-0.1, maxval=0.1
    )
    y_xy_random_flat = y_xy_random_flat + 0.3
    _, subkey = jax.random.split(key)
    y_xy_random = jax.tree_util.tree_unflatten(y_xy_tree, y_xy_random_flat)
    return y_xy_random, subkey


def get_params_pytrees(hl_a:float, hl_b:float, pol_a:float, pol_b:float, h_x:dict, h_xy:dict, r_xy:dict, y_xy:dict):
    params_init = {
        "hl_params":{"a": hl_a,"b": hl_b},
        "pol_params":{"a": pol_a,"b": pol_b},
        "h_x": h_x,
        "h_xy": h_xy,
        "r_xy": r_xy,
        "y_xy": y_xy,
    }
    return params_init


# include alpha y beta in the new parameters
def get_default_params(observable:str="homo_lumo"):
    params_hl = get_init_params_homo_lumo() #homo_lumo
    params_pol = get_init_params_polarizability() #(jnp.ones(1), jnp.ones(1))
    
    if observable.lower() == 'homo_lumo' or observable.lower() == 'hl':
        R_XY = R_XY_AA
        Y_XY = Y_XY_AA
    elif observable.lower() == 'polarizability' or observable.lower() == 'pol':
        R_XY = R_XY_Bohr
        Y_XY = Y_XY_Bohr

    return get_params_pytrees(params_hl[0], params_hl[1], params_pol[0], params_pol[1], H_X, H_XY, R_XY, Y_XY)


def get_params_bool(params_wdecay_:dict):
    """return params_bool where weight decay will be used. array used in masks in OPTAX"""
    params = get_default_params()
    params_bool = params
    params_flat, params_tree = jax.tree_util.tree_flatten(params)
    params_bool = jax.tree_util.tree_unflatten(
        params_tree, jnp.zeros(len(params_flat), dtype=bool)
    )  # all FALSE

    for pb in params_wdecay_:  # ONLY TRUE
        if isinstance(params[pb], dict):
            p_flat, p_tree = jax.tree_util.tree_flatten(params[pb])
            params_bool[pb] = jax.tree_util.tree_unflatten(
                p_tree, jnp.ones(len(p_flat), dtype=bool)
            )
        else:
            params_bool[pb] = jnp.ones(params[pb].shape, dtype=bool)

    return params_bool

def get_random_params(files:dict, key:PRNGKey):
    if not os.path.isfile(files["f_w"]):
        params_init = get_default_params()
        # params_lr,params_coulson = params_init

        hl_a_random = jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        hl_b_random = jax.random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(subkey)

        pol_a_random = jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        pol_b_random = jax.random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
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
            hl_a_random, hl_b_random, pol_a_random , pol_b_random, h_x_random, h_xy_random, r_xy_random, y_xy_random
        )

        f = open(files["f_out"], "a+")
        print("Random initial parameters", file=f)
        print("-----------------------------------", file=f)
        f.close()
        return params, subkey
    else:
        params = get_init_params(files)
        return params, key


def get_init_params(files:dict, obs:str="homo_lumo"):
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

        params = get_params_pytrees(hl_a, hl_b, pol_a, pol_b, h_x, h_xy, r_xy, y_xy)

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


def get_external_field(observable:str='homo_lumo',magnitude:Any=0.):
    if observable.lower() == 'polarizability' or observable.lower() == 'pol':
        if isinstance(magnitude, float):
            return magnitude*jnp.ones(3)
        elif isinstance(magnitude, list):
            return jnp.asarray(magnitude)
        else: #default
            return jnp.zeros(3)
    else:
        return None   

@jit
def update_h_x(h_x:dict):
    xc = h_x["C"]
    xc_tree = jax.tree_unflatten(h_x_tree, xc * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_dif_pytrees, xc_tree, h_x)

@jit
def update_h_xy(h_xy:dict):
    key = frozenset(["C", "C"])
    xcc = h_xy[key]
    xcc_tree = jax.tree_unflatten(h_xy_tree, xcc * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_div_pytrees, xcc_tree, h_xy)

@jit
def update_h_x_au_to_eV(h_x:dict, pol_a:Any):
    x_tree = jax.tree_unflatten(h_x_tree, (pol_a/au_to_eV) * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_mult_pytrees, x_tree, h_x)

@jit
def update_h_xy_au_to_eV(h_xy:dict, pol_a:Any):
    xy_tree = jax.tree_unflatten(h_xy_tree, (pol_a/au_to_eV) * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_mult_pytrees, xy_tree, h_xy)

@jit
def update_r_xy_Bohr_to_AA(r_xy:dict):
    xy_tree = jax.tree_unflatten(r_xy_tree, (Bohr_to_AA) * jnp.ones_like(jnp.array(r_xy_flat)))
    return jax.tree_map(f_div_pytrees, xy_tree, r_xy)

@jit
def normalize_params_wrt_C(params:dict):
    h_x =  update_h_x(params["h_x"])
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
def normalize_params_polarizability(params:dict):
    params_norm_c = normalize_params_wrt_C(params)
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
