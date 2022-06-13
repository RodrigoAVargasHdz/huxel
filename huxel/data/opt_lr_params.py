import sys
import os

import jax
import jax.numpy as jnp
import numpy as onp

from scipy import stats
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

from huxel.data_utils import get_tr_val_data, batch_to_list_class
from huxel.utils import get_files_names, get_default_params
from huxel.observables import _f_observable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def optimize_parameters(obs:str):

    n_tr = 101
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
    y = y_pred#[:,onp.newaxis]
    print(y.shape,x.shape)

    res = stats.linregress(x, y)
    print(res.slope)
    print(res.intercept)

    # # Create linear regression object
    # regr = linear_model.LinearRegression()

    # # Train the model using the training sets
    # regr.fit(x, y)


    # # The coefficients
    # print("Coefficients: \n", regr.coef_)
    # print(regr.get_params())

if __name__ == "__main__":

    optimize_parameters('homo_lumo')