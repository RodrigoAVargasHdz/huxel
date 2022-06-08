import time
import argparse
from attr import has
import numpy as onp
import jax.numpy as jnp
from huxel.huckel import homo_lumo_pred

from huxel.molecule import myMolecule

from huxel.optimization import _optimization as _opt
from huxel.prediction import _pred, _pred_def

def main_test_benzene():

    atom_types = ["C", "C", "C", "C", "C", "C"]
    smile = "C6"
    obj = 'polarizability'
    beta_ = 'c'
    list_Wdecay = ['']
    bool_randW = False
    l = 0
    lr = 1E-2
    batch_size = 64
    n_tr = 1
    job_ = 'opt'
    external_field = 0.0

    conectivity_matrix = jnp.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    # Kjell
    xyz = jnp.array([[ 1.40000000e+00,  3.70074342e-17,  0.00000000e+00],
       [ 7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
       [-7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
       [-1.40000000e+00,  2.08457986e-16,  0.00000000e+00],
       [-7.00000000e-01,  1.21243557e+00,  0.00000000e+00],
       [ 7.00000000e-01,  1.21243557e+00,  0.00000000e+00]])

    

    homo_lumo_grap_ref = -7.01 - (-0.42)

    molec = myMolecule(
        id="benzene",
        smiles=smile,
        atom_types=atom_types,
        conectivity_matrix=conectivity_matrix,
        homo_lumo_grap_ref=homo_lumo_grap_ref,
        polarizability_ref=1.31, #Kjell 
        xyz=xyz,
    )
    molec.get_dm()

    from huxel.huckel import f_polarizability
    from huxel.utils import get_default_params, get_external_field
    from huxel.beta_functions import _f_beta
    from huxel.observables import _f_observable

    from jax import value_and_grad

    params0 = get_default_params()
    f_beta = _f_beta(beta_)
    # f_obs = _f_observable(obj,beta_,external_field)
    external_field = get_external_field(obj,external_field)


    print(f_polarizability(params0,molec,f_beta,external_field))
    # v,g = value_and_grad(f_obs,argnums=(0,))(params0,[molec])
    # # v,g = value_and_grad(f_homo_lumo_gap,argnums=(0,),has_aux=True)(params0,molec,f_beta)
    # print(v)
    # # g = g[0] 
    # print(g)

    # for index,key in enumerate(g):
    #     print(g[key])

    
def main_test():
    import jax
    from jax import value_and_grad

    from huxel.data_utils import get_tr_val_data, split_trainig_test, get_batches
    from huxel.utils import get_files_names, batch_to_list_class, get_default_params, get_external_field
    from huxel.beta_functions import _f_beta
    # from huxel.optimization import loss_polarizability
    from huxel.observables import _loss_function

    n_tr = 1
    l = 0
    beta = 'c'
    bool_randW = False 
    opt_name = 'AdamW'
    batch_size = 5

    # files
    files = get_files_names(n_tr, l, beta, bool_randW, opt_name)
    print(files)

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    Dtr, Dval = split_trainig_test(1, subkey) 
    batches, n_batches = get_batches(Dtr, batch_size, subkey)

    D_tr, D_val, batches, n_batches, subkey = get_tr_val_data(
        files, n_tr, subkey, batch_size
    )
    
    # change D-val for list of myMolecules
    batch_val = batch_to_list_class(D_val)

    # initialize parameters
    params0 = get_default_params()
    f_beta = _f_beta(beta)
    external_field = get_external_field('polarizability',0.1)

    batch = batch_to_list_class(next(batches))
    for b in batch:
        print(b.smiles,b.polarizability_ref)

    # select the function for off diagonal elements for H
    # f_beta = _f_beta(beta)

    # grad_fn = value_and_grad(loss_polarizability, argnums=(0,),has_aux=True)

    # (loss,y_pred), grads = grad_fn(params0, batch, f_beta, external_field)
    # print(loss)
    # # print(grads)
    # print(y_pred)

    print('--------------')
    f_loss = _loss_function('polarizability',beta, 0.01)
    print(f_loss(params0,batch))
    _,_ = value_and_grad(f_loss, argnums=(0,),has_aux=True)(params0,batch)

    print('--------------')
    f_loss = _loss_function('homo_lumo',beta)
    print(f_loss(params0,batch))
    _,_ = value_and_grad(f_loss, argnums=(0,),has_aux=True)(params0,batch)


if __name__ == "__main__":
    main_test_benzene()
    # main_test()

