import os
import time
import datetime
import numpy as onp
import argparse

import jax
import jax.numpy as jnp
from jax import random
from jax import lax,value_and_grad

from flax import optim
import optax

# from huxel.molecule import myMolecule
from huxel.data import get_tr_val_data
from huxel.beta_functions import _f_beta
from huxel.huckel import linear_model_pred
from huxel.parameters import update_h_x,update_h_xy
from huxel.utils import get_files_names, batch_to_list_class, get_init_params, get_random_params
from huxel.utils import print_head, print_tail, get_params_file_itr
from huxel.utils import save_tr_and_val_loss

from jax.config import config
jax.config.update('jax_enable_x64', True)


def f_loss_batch(params_tot,batch,f_beta):
    params_lr,params_c = params_tot
    h_x, h_xy, r_xy, y_xy = params_c
    h_x = update_h_x(h_x)
    h_xy = update_h_xy(h_xy)
    params_c = (h_x, h_xy, r_xy, y_xy)
    params_tot = (params_lr,params_c)

    y_pred,z_pred,y_true = linear_model_pred(params_tot,batch,f_beta)

    # diff_y = jnp.abs(y_pred-y_true)
    diff_y = (y_pred-y_true)**2
    return jnp.mean(diff_y)

def _optimization(n_tr=50,batch_size=100,lr=2E-3,l=0,beta='exp',bool_randW=False):

    # optimization parameters
    # if n_tr < 100 is considered as porcentage of the training data 
    w_decay = 1E-4
    n_epochs = 50
    opt_name = 'Adam'

    # files
    files = get_files_names(n_tr,l,beta,bool_randW,opt_name)

    # print info about the optimiation
    print_head(files,n_tr,l,lr,w_decay,n_epochs,batch_size,opt_name,beta)

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    D_tr,D_val,batches,n_batches,subkey = get_tr_val_data(files,n_tr,subkey,batch_size)

    # change D-val for list of myMolecules
    batch_val = batch_to_list_class(D_val)

    # initialize parameters
    if bool_randW:
        params_init,subkey = get_random_params(files,subkey)
    else:
        params_init = get_init_params(files)

    # select the function for off diagonal elements for H
    f_beta = _f_beta(beta)
    # f_loss_batch_ = lambda params,batch: f_loss_batch(params,batch,f_beta)

    # @jit
    def train_step(optimizer, batch,f_beta):
        grad_fn = value_and_grad(f_loss_batch,argnums=(0,))
        loss, grad = grad_fn(optimizer.target, batch,f_beta)
        optimizer = optimizer.apply_gradient(grad[0])
        return optimizer, loss

    optimizer = optim.Adam(learning_rate=lr,weight_decay=w_decay).create(params_init)
    optimizer = jax.device_put(optimizer)   

    loss_val0 = 1E16
    f_params = params_init
    loss_tr_ = []
    loss_val_ = []
    for epoch in range(n_epochs+1):
        start_time_epoch = time.time()
        loss_tr_epoch = []
        for _ in range(n_batches):
            batch = batch_to_list_class(next(batches))
            optimizer, loss_tr = train_step(optimizer, batch,f_beta)
            loss_tr_epoch.append(loss_tr)

        loss_tr_mean = jnp.mean(jnp.asarray(loss_tr_epoch).ravel())
        params = optimizer.target

        loss_val = f_loss_batch(params,batch_val,f_beta)

        f = open(files['f_out'],'a+')
        time_epoch = time.time() - start_time_epoch
        print(epoch,loss_tr,loss_val,time_epoch,file=f)   
        f.close()

        loss_tr_.append(loss_tr_mean)
        loss_val_.append(loss_val)

        if loss_val < loss_val0:
            loss_val0 = loss_val
            f_params = params
            jnp.save(files['f_w'],f_params)
            jnp.save(get_params_file_itr(files,epoch),f_params)

    save_tr_and_val_loss(files,loss_tr_,loss_val_,n_epochs+1)

    print_tail(files)

def main():
    parser = argparse.ArgumentParser(description='opt overlap NN')
    parser.add_argument('--N', type=int, default=5, help='traning data')
    parser.add_argument('--l', type=int, default=0, help='label')
    parser.add_argument('--lr', type=float, default=2E-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batches')
    parser.add_argument('--beta', type=str, default='exp_freezeR', help='beta function')
    parser.add_argument('--randW', type=bool, default=False, help='random initial params')

    # bathch_size = #1024#768#512#256#128#64#32
    args = parser.parse_args()
    l = args.l
    n_tr = args.N
    lr = args.lr
    batch_size = args.batch_size
    beta = args.beta
    bool_randW = args.randW

    _optimization(n_tr,batch_size,lr,l,beta,bool_randW)

if __name__ == "__main__":
    main()
