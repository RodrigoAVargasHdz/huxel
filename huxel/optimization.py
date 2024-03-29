import time
from typing import Any
import jax
import jax.numpy as jnp
from jax import value_and_grad

import optax

from huxel.data_utils import get_tr_val_data
from huxel.utils import (
    get_init_params,
    get_random_params,
    get_params_bool,
)
from huxel.outfiles_utils import get_files_names,  print_head, print_tail, get_params_file_itr
from huxel.data_utils import save_tr_and_val_loss, batch_to_list_class
from huxel.observables import _loss_function

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_disable_jit', True)

# label_parmas_all = ['alpha', 'beta', 'h_x', 'h_xy', 'r_xy', 'y_xy']


def _optimization(
    obs: str = 'homo_lumo',
    n_tr: int = 1,
    batch_size: int = 100,
    lr: float = 2e-3,
    l: int = 0,
    beta: str = "exp",
    list_Wdecay: list = None,
    bool_randW: bool = False,
    external_field: Any = None,
) -> None:
    """Optimization routine

    Args:
        obs (str, optional): target observable. Defaults to 'homo_lumo'.
        n_tr (int, optional): number of training data. Defaults to 1.
        batch_size (int, optional): batch size. Defaults to 100.
        lr (float, optional): learning rate. Defaults to 2e-3.
        l (int, optional): label. Defaults to 0.
        beta (str, optional): atom-atom interaction. Defaults to "exp".
        list_Wdecay (list, optional): list of parameters for weight decay. Defaults to None.
        bool_randW (bool, optional): boolean for random initial parameters. Defaults to False. (False -> literature parameters; True -> random parameters)
        external_field (Any, optional): external field value. Defaults to None.

    """

    # optimization parameters
    w_decay = 1e-4
    n_epochs = 25
    opt_name = "AdamW"

    # files
    files = get_files_names(obs, n_tr, l, beta, bool_randW, opt_name)

    # print info about the optimization
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
    batch_val = batch_to_list_class(D_val, obs)

    # initialize parameters
    if bool_randW:
        params_init, subkey = get_random_params(files, subkey)
    else:
        params_init = get_init_params(files)

    params_bool = get_params_bool(list_Wdecay)

    # select the function for off diagonal elements for H
    f_loss_batch = _loss_function(obs, beta, external_field)
    grad_fn = value_and_grad(f_loss_batch, argnums=(0,), has_aux=True)

    # OPTAX optimizer
    # schedule = optax.exponential_decay(init_value=lr,transition_steps=25,decay_rate=0.1)
    optimizer = optax.adamw(learning_rate=lr, mask=params_bool)

    opt_state = optimizer.init(params_init)
    params = params_init

    # @jit
    def train_step(params, optimizer_state, batch):
        (loss, _), grads = grad_fn(params, batch)
        updates, opt_state = optimizer.update(
            grads[0], optimizer_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    loss_val0 = 1e16
    f_params = params_init
    loss_tr_ = []
    loss_val_ = []
    for epoch in range(n_epochs + 1):
        start_time_epoch = time.time()
        loss_tr_epoch = []
        for _ in range(n_batches):
            batch = batch_to_list_class(next(batches), obs)
            params, opt_state, loss_tr = train_step(params, opt_state, batch)
            loss_tr_epoch.append(loss_tr)

        loss_tr_mean = jnp.mean(jnp.asarray(loss_tr_epoch).ravel())
        loss_val, _ = f_loss_batch(params, batch_val)

        f = open(files["f_out"], "a+")
        time_epoch = time.time() - start_time_epoch
        print(epoch, loss_tr_mean, loss_val, time_epoch, file=f)
        # print(params["pol_params"],file=f)
        f.close()

        loss_tr_.append(loss_tr_mean)
        loss_val_.append(loss_val)

        if loss_val < loss_val0:
            loss_val0 = loss_val
            f_params = params  # f_params_preprocessing(params)
            jnp.save(files["f_w"], f_params)
            jnp.save(get_params_file_itr(files, epoch), f_params)

    save_tr_and_val_loss(files, loss_tr_, loss_val_, n_epochs + 1)

    print_tail(files)
