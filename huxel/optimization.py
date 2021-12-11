import os
import time
import datetime
import numpy as onp
import argparse

import jax
import jax.numpy as jnp
from jax import random
from jax import lax,value_and_grad

# from flax import optim
import optax

from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver

# from huxel.molecule import myMolecule
from huxel.data import get_tr_val_data, get_batches
from huxel.beta_functions import _f_beta
from huxel.huckel import linear_model_pred
from huxel.utils import get_files_names, get_init_params, get_random_params, get_params_bool
from huxel.utils import print_head, print_tail, get_params_file_itr, update_params_all
from huxel.utils import save_tr_and_val_loss, batch_to_list_class
from huxel.parameters import R_XY

from jax.config import config
jax.config.update('jax_enable_x64', True)

# label_parmas_all = ['alpha', 'beta', 'h_x', 'h_xy', 'r_xy', 'y_xy']

def f_loss_batch(params,params_r,data,f_beta):
    # data = batch_to_list_class(data)

    params = update_params_all(params)
    y_pred,z_pred,y_true = linear_model_pred(params,params_r,data,f_beta)

    # diff_y = jnp.abs(y_pred-y_true)
    diff_y = (y_pred-y_true)**2
    return jnp.mean(diff_y)

def _optimization(n_tr=50,batch_size=100,lr=2E-3,l=0,beta='exp',list_Wdecay=None,bool_randW=False):

    # optimization parameters
    # if n_tr < 100 is considered as porcentage of the training data 
    w_decay = 5E-4
    n_epochs = 50
    opt_name = 'AdamW'

    # files
    files = get_files_names(n_tr,l,beta,bool_randW,opt_name)

    # print info about the optimiation
    print_head(files,n_tr,l,lr,w_decay,n_epochs,batch_size,opt_name,beta,list_Wdecay)

    f_beta = _f_beta(beta)

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    D_tr,D_val,batches,n_batches,subkey = get_tr_val_data(files,n_tr,subkey,batch_size)
    batch_val = batch_to_list_class(D_val)
    batch_val_size = batch_size
    batches_val, n_batches_val = get_batches(D_val,batch_val_size,subkey)
    _, subkey = jax.random.split(subkey)

    f_loss = lambda params,params_r,data: f_loss_batch(params,params_r,data,f_beta)

    params,params_r = get_init_params(files) 

    def step_in(opt_in, params,params_r, state, data):
        loss_tr, grad = value_and_grad(f_loss)(params,params_r, data)
        updates, state = opt_in.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss_tr

    def step_out(opt_out,opt_in,params_r, params,state_out,state_in,data_out):
        loss_tr_epoch = []
        for _ in range(n_batches):
            batch_in = batch_to_list_class(next(batches))
            params,state_in,loss_tr = step_in(opt_in,params,params_r,state_in,batch_in)
            loss_tr_epoch.append(loss_tr)
        loss_tr_mean = jnp.mean(jnp.asarray(loss_tr_epoch).ravel())
        loss_val, grad = value_and_grad(f_loss,argnums=1)(params,params_r, data_out)
        updates, state_out = opt_out.update(grad, state_out, params_r)
        params_r = optax.apply_updates(params_r, updates)
        return params_r, state_out, (loss_val,loss_tr_mean), (params,state_in,opt_in,opt_out)

    # OPTAX ADAM
    # schedule = optax.exponential_decay(init_value=lr,transition_steps=25,decay_rate=0.1)
    optimizer_in = optax.adamw(learning_rate=lr,weight_decay=w_decay)
    opt_in_state = optimizer_in.init(params)
    
    optimizer_out = optax.adamw(learning_rate=lr,weight_decay=0.)
    opt_out_state = optimizer_out.init(params_r)

    f_params = params
    loss_tr_ = []
    loss_val_ = []
    loss_val0 = 1E6
    for epoch in range(n_epochs+1):
        start_time_epoch = time.time()
        loss_val_epoch = []
        for _ in range(n_batches_val):
            batch = batch_to_list_class(next(batches_val))
            params_r, opt_out_state, (loss_val,loss_tr_mean), (params,opt_in_state ,optimizer_in,optimizer_out) = step_out(optimizer_out, optimizer_in,params_r, params,opt_out_state,opt_in_state,batch)
            loss_val_epoch.append(loss_val)

        loss_val_epoch_mean = jnp.mean(jnp.asarray(loss_val_epoch).ravel())
        time_epoch = time.time() - start_time_epoch

        loss_val_tot = f_loss_batch(params,params_r,batch_val,f_beta)

        f = open(files['f_out'],'a+')
        print(epoch,loss_tr_mean,loss_val_epoch_mean,loss_val_tot,time_epoch,file=f)   
        f.close()
    
        loss_tr_.append(loss_tr_mean)
        loss_val_.append(loss_val_tot)

        if loss_val_epoch_mean < loss_val0:
            loss_val0 = loss_val_epoch_mean
            f_params = update_params_all(params)
            f_params['r_xy'] = params_r
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
    parser.add_argument('--beta', type=str, default='exp', help='beta function')
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
