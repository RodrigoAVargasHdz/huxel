import os
import datetime
from typing import Any

import jax
import jax.numpy as jnp
from jax import jit

import numpy as onp

# --------------------------------
#     FILES


def get_r_dir_old(method: str):
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


def get_r_dir(method: str, bool_randW: bool):
    if bool_randW:
        r_dir = "./Results_{}_randW/".format(method)
    else:
        r_dir = "./Results_{}/".format(method)

    if not os.path.exists(r_dir):
        os.mkdir(r_dir)
    return r_dir


def get_files_names(obs: str, N: int, l: int, beta: str, randW: bool, opt_name: str = "Adam"):
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


def get_params_file_itr(files: dict, itr: int):
    # r_dir = './Results_xyz/'
    f_job = files["f_job"]
    r_dir = files["r_dir"]
    file_ = "{}/params_{}_itr_{}.npy".format(r_dir, f_job, itr)
    return file_


# --------------------------------
#     HEAD OF FILE
def print_head(
    files: dict, obs: str, N: int, l: int, lr: float, w_decay: Any, n_epochs: int, batch_size: int, opt_name: str, beta: str, list_Wdecay: list
):
    f = open(files["f_out"], "a+")
    print("-----------------------------------", file=f)
    print("Starting time", file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print("-----------------------------------", file=f)
    print(files["f_out"], file=f)
    print("Observable = {}".format(obs), file=f)
    print("f beta: {}".format(beta), file=f)
    print("N = {}, l = {}".format(N, l), file=f)
    print("Opt method = {}".format(opt_name), file=f)
    print("lr = {}, w decay = {}".format(lr, w_decay), file=f)
    print("batch size = {}".format(batch_size), file=f)
    print("N Epoch = {}".format(n_epochs), file=f)
    print("W Decay {}: ".format(list_Wdecay), file=f)
    print("-----------------------------------", file=f)
    f.close()


#     TAIL OF FILE
def print_tail(files: dict):
    f = open(files["f_out"], "a+")
    print("-----------------------------------", file=f)
    print("Finish time", file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print("-----------------------------------", file=f)
    f.close()
