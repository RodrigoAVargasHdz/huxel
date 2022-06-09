import time
from typing import Any
import numpy as onp

import jax
import jax.numpy as jnp
from jax import random, lax, value_and_grad

from flax import optim
import optax

from huxel.data_utils import get_tr_val_data
from huxel.utils import (
    get_files_names,
    get_init_params,
    get_random_params,
    get_params_bool,
)
from huxel.utils import print_head, print_tail, get_params_file_itr
from huxel.utils import save_tr_and_val_loss, batch_to_list_class
from huxel.observables import _f_observable, _loss_function, _preprocessing_params

from jax.config import config
jax.config.update("jax_enable_x64", True)

# label_parmas_all = ['alpha', 'beta', 'h_x', 'h_xy', 'r_xy', 'y_xy']
def _optimization(
    obs:str='homo_lumo',
    n_tr:int=1,
    batch_size:int=100,
    lr:float=2e-3,
    l:int=0,
    beta:str="exp",
    list_Wdecay:list=None,
    bool_randW:bool=False,
    external_field:Any=None,
):

    # optimization parameters
    # if n_tr < 100 is considered as porcentage of the training data
    w_decay = 1e-4
    n_epochs = 25
    opt_name = "AdamW"

    # files
    files = get_files_names(obs, n_tr, l, beta, bool_randW, opt_name)

    # print info about the optimiation
    print_head(
        files, obs, n_tr, l, lr, w_decay, n_epochs, batch_size, opt_name, beta, list_Wdecay
    )

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    D_tr, D_val, batches, n_batches, subkey = get_tr_val_data(
        files, n_tr, subkey, batch_size
    )

    # change D-val for list of myMolecules
    batch_val = batch_to_list_class(D_val)

    # initialize parameters
    if bool_randW:
        params_init, subkey = get_random_params(files, subkey)
    else:
        params_init = get_init_params(files)

    params_bool = get_params_bool(list_Wdecay)

    f_params_preprocessing = _preprocessing_params(obs)

    # select the function for off diagonal elements for H
    f_loss_batch = _loss_function(obs, beta, external_field)
    grad_fn = value_and_grad(f_loss_batch, argnums=(0,), has_aux=True)

    params_flat, params_tree = jax.tree_util.tree_flatten(params_init)
    # print(params_flat)

    batch = batch_to_list_class(next(batches))
    
    (loss,_), grad = grad_fn(params_init,batch)

    print(loss)
    grad_params_flat, _ = jax.tree_util.tree_flatten(grad)
    print(grad_params_flat)

    assert 0

    # OPTAX ADAM
    # schedule = optax.exponential_decay(init_value=lr,transition_steps=25,decay_rate=0.1)
    optimizer = optax.adamw(learning_rate=lr, mask=params_bool)
    opt_state = optimizer.init(params_init)
    params = params_init

    # @jit
    def train_step(params, optimizer_state, batch):
        (loss,_), grads = grad_fn(params, batch)
        updates, opt_state = optimizer.update(grads[0], optimizer_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    loss_val0 = 1e16
    f_params = params_init
    loss_tr_ = []
    loss_val_ = []
    for epoch in range(n_epochs + 1):
        start_time_epoch = time.time()
        loss_tr_epoch = []
        for _ in range(n_batches):
            batch = batch_to_list_class(next(batches))
            params, opt_state, loss_tr = train_step(params, opt_state, batch)
            loss_tr_epoch.append(loss_tr)

        loss_tr_mean = jnp.mean(jnp.asarray(loss_tr_epoch).ravel())
        loss_val,_ = f_loss_batch(params, batch_val)

        f = open(files["f_out"], "a+")
        time_epoch = time.time() - start_time_epoch
        print(epoch, loss_tr_mean, loss_val, time_epoch, file=f)
        f.close()

        loss_tr_.append(loss_tr_mean)
        loss_val_.append(loss_val)

        if loss_val < loss_val0:
            loss_val0 = loss_val
            f_params = params #f_params_preprocessing(params)
            jnp.save(files["f_w"], f_params)
            jnp.save(get_params_file_itr(files, epoch), f_params)

    save_tr_and_val_loss(files, loss_tr_, loss_val_, n_epochs + 1)

    print_tail(files)
