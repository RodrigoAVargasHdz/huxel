from typing import Any
import jax
import jax.numpy as jnp

from huxel.data_utils import get_raw_data, batch_to_list_class
from huxel.huckel import homo_lumo_pred, polarizability_pred
from huxel.observables import _f_observable
from huxel.utils import (
    get_external_field,
    get_init_params,
    get_default_params,
)
from huxel.outfiles_utils import get_files_names

from jax.config import config

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_disable_jit', True)


def _pred(obs: str = "homo_lumo", n_tr: int = 50, l: int = 0, beta: str = "exp", bool_randW: bool = False):

    if obs.lower() == 'hl' or obs.lower() == 'homo_lumo':
        external_field = None
    elif obs.lower() == 'pol' or obs.lower() == 'polarizability':
        external_field = 0.

    opt_name = "AdamW"

    # files
    files = get_files_names(obs, n_tr, l, beta, bool_randW, opt_name)

    # if os.path.isfile(files['f_pred']):
    #     assert 0

    # initialize parameters
    params = get_init_params(files)

    # get data
    _, D = get_raw_data()
    D = batch_to_list_class(D, obs)

    # prediction
    f_pred = _f_observable(obs, beta, external_field)
    y_pred, z_pred, y_true = f_pred(params, D)

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


def _pred_def(obs: str = "homo_lumo", beta: str = "exp", pred_data: str = "val"):

    if obs.lower() == 'hl' or obs.lower() == 'homo_lumo':
        external_field = None
    elif obs.lower() == 'pol' or obs.lower() == 'polarizability':
        external_field = 0.

    # get data
    D_tr, D_val = get_raw_data()

    if pred_data == "val" or pred_data == "validation":
        D = D_val
        data_ = 'val'
    if pred_data == "tr" or pred_data == "training":
        D = D_tr
        data_ = 'training'

    D = batch_to_list_class(D)

    # files
    r_dir = "Results_default/"

    f_job = "huckel_{}_default_{}".format(obs, data_)
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

    # prediction
    external_field = 0.
    external_field = get_external_field(obs, external_field)

    f_pred = _f_observable(obs, beta, external_field)
    y_pred, z_pred, y_true = f_pred(params0, D)

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
