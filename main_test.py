import pickle
import time
import argparse
from attr import has
import numpy as onp
import jax.numpy as jnp
from huxel.huckel import homo_lumo_pred

from huxel.molecule import myMolecule

from huxel.optimization import _optimization as _opt
from huxel.prediction import _pred, _pred_def
from huxel.utils import normalize_params_polarizability


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

    connectivity_matrix = jnp.array(
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
    xyz = jnp.array([[1.40000000e+00,  3.70074342e-17,  0.00000000e+00],
                     [7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
                     [-7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
                     [-1.40000000e+00,  2.08457986e-16,  0.00000000e+00],
                     [-7.00000000e-01,  1.21243557e+00,  0.00000000e+00],
                     [7.00000000e-01,  1.21243557e+00,  0.00000000e+00]])

    homo_lumo_grap_ref = -7.01 - (-0.42)

    molec = myMolecule(
        id0="benzene",
        smiles=smile,
        atom_types=atom_types,
        connectivity_matrix=connectivity_matrix,
        homo_lumo_grap_ref=homo_lumo_grap_ref,
        polarizability_ref=1.31,  # Kjell
        xyz=xyz,
    )
    molec.get_dm()
    molec.get_dm_AA_to_Bohr()
    molec.get_xyz_AA_to_Bohr()

    from huxel.huckel import f_polarizability
    from huxel.utils import get_default_params, get_external_field
    from huxel.beta_functions import _f_beta
    from huxel.observables import _f_observable

    from jax import value_and_grad

    params0 = get_default_params()
    params0 = normalize_params_polarizability(params0)
    # print(params0['h_xy'])
    f_beta = _f_beta(beta_)
    # f_obs = _f_observable(obj,beta_,external_field)
    external_field = get_external_field(obj, external_field)

    print(f_polarizability(params0, molec, f_beta, external_field))
    print(molec.xyz_Bohr)
    print(molec.dm)
    print("---------------------")
    # v,g = value_and_grad(f_obs,argnums=(0,))(params0,[molec])
    # # v,g = value_and_grad(f_homo_lumo_gap,argnums=(0,),has_aux=True)(params0,molec,f_beta)
    # print(v)
    # # g = g[0]
    # print(g)

    # for index,key in enumerate(g):
    #     print(g[key])


def main_test_anthracene():

    obj = 'polarizability'
    beta_ = 'c'
    external_field = 0.

    smile = 'C12=CC=CC=C1C=C3C(C=CC=C3)=C2'
    atom_types = ['C', 'C', 'C', 'C', 'C', 'C',
                  'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
    conectivity_matrix = jnp.array(
        [[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    )

    xyz = jnp.array([[1.22598e+00, -7.05260e-01, -2.25000e-03],
                     [2.45083e+00, -1.39265e+00, -8.22000e-03],
                     [3.66029e+00, -6.96370e-01, -3.78000e-03],
                     [3.66083e+00,  6.94950e-01,  2.97000e-03],
                     [2.45185e+00,  1.39255e+00,  1.62000e-03],
                     [1.22619e+00,  7.06340e-01, 0.00000e+00],
                     [1.08000e-03,  1.39380e+00,  1.59000e-03],
                     [-1.22333e+00,  7.05700e-01,  1.50000e-03],
                     [-1.22320e+00, -7.05590e-01,  2.70000e-03],
                     [-2.44795e+00, -1.39337e+00,  3.36000e-03],
                     [-3.65746e+00, -6.97250e-01, -3.10000e-04],
                     [-3.65830e+00,  6.93930e-01, -1.96000e-03],
                     [-2.44954e+00,  1.39180e+00,  5.30000e-04],
                     [1.52000e-03, -1.39301e+00,  1.42000e-03]]
                    )

    homo_lumo_grap_ref = 0.

    molec = myMolecule(
        id="antrhacene",
        smiles=smile,
        atom_types=atom_types,
        connectivity_matrix=conectivity_matrix,
        homo_lumo_grap_ref=homo_lumo_grap_ref,
        polarizability_ref=0.,  # Kjell
        xyz=xyz,
    )
    molec.get_dm()
    molec.get_dm_AA_to_Bohr()
    molec.get_xyz_AA_to_Bohr()

    from huxel.huckel import f_polarizability
    from huxel.utils import get_default_params, get_external_field
    from huxel.beta_functions import _f_beta
    from huxel.observables import _f_observable

    from jax import value_and_grad

    params0 = get_default_params()
    params0 = normalize_params_polarizability(params0)
    # print(params0['h_xy'])
    f_beta = _f_beta(beta_)
    # f_obs = _f_observable(obj,beta_,external_field)
    external_field = get_external_field(obj, external_field)

    print(f_polarizability(params0, molec, f_beta, external_field))


def main_test_KJell():
    from huxel.huckel import f_polarizability
    from huxel.utils import get_default_params, get_external_field
    from huxel.beta_functions import _f_beta
    from huxel.observables import _f_observable

    from jax import value_and_grad

    import numpy as onp

    obj = 'polarizability'
    beta_ = 'c'
    external_field = 0.

    a_file = open("molecule_test.pkl", "rb")
    molecules_pkl = pickle.load(a_file)

    for index, key in enumerate(molecules_pkl):
        s = molecules_pkl[key]['smile']

        xyz_w_atoms = onp.asarray(molecules_pkl[key]['xyz'])
        xyz = jnp.asarray(xyz_w_atoms[:, 1:].astype(float))
        dm = jnp.asarray(molecules_pkl[key]['dm'], dtype=jnp.int16)

        homo_lumo_grap_ref = 0.
        atoms_list = ['C' for i in range(xyz.shape[0])]

        molec = myMolecule(
            id=index,
            smiles=s,
            atom_types=atoms_list,
            connectivity_matrix=dm,
            homo_lumo_grap_ref=homo_lumo_grap_ref,
            polarizability_ref=0.,  # Kjell
            xyz=xyz,
        )
        molec.get_dm()
        molec.get_dm_AA_to_Bohr()
        molec.get_xyz_AA_to_Bohr()

        params0 = get_default_params()
        params0 = normalize_params_polarizability(params0)
        # print(params0['h_xy'])
        # print(params0['h_xy'])
        f_beta = _f_beta(beta_)
        # f_obs = _f_observable(obj,beta_,external_field)
        external_field = get_external_field(obj, external_field)

        print(s)
        print(atoms_list)
        print(f_polarizability(params0, molec, f_beta, external_field))
        print(xyz)
        print(molec.xyz_Bohr)
        print(molec.dm)
        print(params0)
        assert 0


def main_test():
    import jax
    from jax import value_and_grad

    from huxel.data_utils import get_tr_val_data, split_training_test, get_batches
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

    Dtr, Dval = split_training_test(1, subkey)
    batches, n_batches = get_batches(Dtr, batch_size, subkey)

    D_tr, D_val, batches, n_batches, subkey = get_tr_val_data(
        files, n_tr, subkey, batch_size
    )

    # change D-val for list of myMolecules
    batch_val = batch_to_list_class(D_val)

    # initialize parameters
    params0 = get_default_params()
    f_beta = _f_beta(beta)
    external_field = get_external_field('polarizability', 0.1)

    batch = batch_to_list_class(next(batches))
    for b in batch:
        print(b.smiles, b.polarizability_ref)

    # select the function for off diagonal elements for H
    # f_beta = _f_beta(beta)

    # grad_fn = value_and_grad(loss_polarizability, argnums=(0,),has_aux=True)

    # (loss,y_pred), grads = grad_fn(params0, batch, f_beta, external_field)
    # print(loss)
    # # print(grads)
    # print(y_pred)

    print('--------------')
    f_loss = _loss_function('polarizability', beta, 0.01)
    print(f_loss(params0, batch))
    _, _ = value_and_grad(f_loss, argnums=(0,), has_aux=True)(params0, batch)

    print('--------------')
    f_loss = _loss_function('homo_lumo', beta)
    print(f_loss(params0, batch))
    _, _ = value_and_grad(f_loss, argnums=(0,), has_aux=True)(params0, batch)


if __name__ == "__main__":
    main_test_benzene()
    # main_test_anthracene()
    main_test_KJell()
    # main_test()
