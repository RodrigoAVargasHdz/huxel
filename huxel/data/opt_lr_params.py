import sys
import os

import jax
import jax.numpy as jnp
import numpy as onp

from scipy import stats
import matplotlib 
import matplotlib.pyplot as plt

from huxel.data_utils import get_tr_val_data, batch_to_list_class
from huxel.utils import get_files_names, get_default_params
from huxel.observables import _f_observable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def optimize_parameters(obs:str):

    n_tr = 5000
    batch_size = 128
    l = 0
    beta = 'c'
    bool_randW = False
    opt_name = 'AdamW'

    # files
    files = get_files_names(obs, n_tr, l, beta, bool_randW, opt_name)

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    D_tr, D_val, batches, n_batches, subkey = get_tr_val_data(
        files, n_tr, subkey, batch_size
    )

    # model
    f_obs = _f_observable(obs,beta)

    # initialize parameters
    params0 = get_default_params()

    batch_tr = batch_to_list_class(D_tr)

    y_pred,z_pred, y_true = f_obs(params0,batch_tr)

    x = z_pred#[:,onp.newaxis]
    y = y_true#[:,onp.newaxis]
    print(y.shape,x.shape)

    res = stats.linregress(x, y)
    print(f'Obs = {obs}')
    print(f'slope = {res.slope}')
    print(f'intercept = {res.intercept}')

    plt.figure(0)
    plt.scatter(z_pred,y_true)
    plt.ylabel('DFT')
    plt.xlabel('Polarizability')
    plt.savefig(f'fig_huckel_vs_DFT_{obs}.png')

if __name__ == "__main__":

    # optimize_parameters('homo_lumo')
    optimize_parameters('polarizability') 