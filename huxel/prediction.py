import os
import time
import datetime

import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from huxel.data import get_raw_data
from huxel.huckel import linear_model_pred
from huxel.beta_functions import _f_beta
from huxel.utils import (
    get_files_names,
    batch_to_list_class,
    get_init_params,
    get_default_params,
)

from jax.config import config

jax.config.update("jax_enable_x64", True)


def _pred(n_tr=50, l=0, beta="exp", bool_randW=False):
    opt_name = "AdamW"
    # files
    files = get_files_names(n_tr, l, beta, bool_randW, opt_name)

    # print info about the optimiation
    # print_head(files,n_tr,l,lr,w_decay,n_epochs,batch_size,opt_name)

    # if os.path.isfile(files['f_pred']):
    #     assert 0

    # initialize parameters
    params_init = get_init_params(files)
    # params0 = get_default_params()

    # get data
    _, D = get_raw_data()
    D = batch_to_list_class(D)

    f_beta = _f_beta(beta)

    # prediction
    y_pred, z_pred, y_true = linear_model_pred(params_init, D, f_beta)

    # prediction original parameters
    # params0 = get_default_params()
    # params_lr, params = params0
    # alpha,beta = params_lr
    # y_pred,z_pred,y_true = linear_model_pred(params0,D)

    print("finish prediction")

    R = {
        "y_pred": y_pred,
        "z_pred": z_pred,
        "y_true": y_true,
        # 'y0_pred': y0_pred,
        # 'z0_pred': z0_pred,
        # 'y0_true': y0_true,
    }

    jnp.save(files["f_pred"], R)
    # jnp.save('./Results/Prediction_coulson.npy',R)


def _pred_def(beta="exp"):
    opt_name = "AdamW"
    # files
    r_dir = "Results_default/"

    f_job = "huckel_xyz_default".format(opt_name)
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
    }

    # initialize parameters
    params0 = get_default_params()

    # get data
    _, D = get_raw_data()
    D = batch_to_list_class(D)

    f_beta = _f_beta(beta)

    # prediction
    y_pred, z_pred, y_true = linear_model_pred(params0, D, f_beta)

    print("finish prediction")

    R = {
        "y_pred": y_pred,
        "z_pred": z_pred,
        "y_true": y_true,
        # 'y0_pred': y0_pred,
        # 'z0_pred': z0_pred,
        # 'y0_true': y0_true,
    }

    jnp.save(files["f_pred"], R)
