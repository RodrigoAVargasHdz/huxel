import os
from typing import Any, Tuple

import numpy as onp
# import numpy.random as onpr

import jax
import jax.numpy as jnp

from huxel.molecule import myMolecule
from huxel.utils import load_pre_opt_params

PRNGKey = Any


def get_raw_data(r_data: str = 'huxel/data/') -> Tuple:
    """All data set, Training and Validation

    Args:
        r_data (str, optional): path to folder. Defaults to 'huxel/data/'.

    Returns:
        Tuple: Training and validation data set
    """
    return (
        onp.load(
            os.path.join(r_data, 'data_gdb13_training.npy'),
            allow_pickle=True,
        ),
        onp.load(
            os.path.join(r_data, 'data_gdb13_test.npy'),
            allow_pickle=True
        ),
    )


def get_batches(Dtr: Any, batch_size: int, key: PRNGKey) -> Tuple:
    """Batches

    Args:
        Dtr (Any): Tuple with training data
        batch_size (int): batch size
        key (PRNGKey): a PRNG key used as the random key.

    Returns:
        Tuple: batches generator, number of batches

    Yields:
        Iterator[Tuple]: _description_
    """
    # Dtr = get_data()
    # Xtr,ytr = Dtr
    N = len(Dtr)

    n_complete_batches, leftover = divmod(N, batch_size)
    n_batches = n_complete_batches + bool(leftover)

    def data_stream():
        # rng = onpr.RandomState(0)
        while True:
            # perm = rng.permutation(N)
            perm = jax.random.permutation(key, jnp.arange(N))
            for i in range(n_batches):
                batch_idx = perm[i * batch_size: (i + 1) * batch_size]
                yield Dtr[batch_idx.tolist()]

    batches = data_stream()

    return batches, n_batches


def split_training_test(n_tr: int, key: PRNGKey, D: Any = None, n_val: int = 1000) -> Tuple:
    """Split training and validation data

    Args:
        n_tr (int): number of training data. If n_tr <= 99, N is considered as % of the data set
        key (PRNGKey): a PRNG key used as the random key.
        D (Any, optional): Data set. Defaults to None, loads data using get_raw_data() function.
        n_val (int, optional): Validation data number. Default 1000.

    Returns:
        Tuple: training and validation data tuples
    """
    if D is None:
        D, _ = get_raw_data()
    N_tot = len(D)

    # % of the total data
    if n_tr <= 99:
        n_tr = int(N_tot * n_tr / 100)

    n_val = n_tr + n_val  # extra 1000 points for validation

    # represents the absolute number of test samples
    N_tst = N_tot - n_tr

    j_ = jnp.arange(N_tot)
    j_ = jax.random.permutation(key, j_, axis=0)
    j_tr = j_[:n_tr]
    j_val = j_[n_tr:n_val]

    D_tr = D[j_tr]
    D_val = D[j_val]
    return D_tr, D_val


def get_tr_val_data(files: dict, n_tr: int, subkey: PRNGKey, batch_size: int) -> Tuple:
    """Training and validation data set

    Args:
        files (dict): files name
        n_tr (int): number of training data 
        subkey (PRNGKey): a PRNG key used as the random key.
        batch_size (int):batch size

    Returns:
        Tuple: Training data tuple, Validation data tuple, batches generator, number of batches, new PRNG key
    """
    if os.path.isfile(files["f_data"]):
        _D = onp.load(files["f_data"], allow_pickle=True)
        D_tr = _D.item()["Training"]
        D_val = _D.item()["Validation"]
        n_batches = _D.item()["n_batches"]
        batches, n_batches = get_batches(D_tr, batch_size, subkey)
    else:
        D_tr, D_val = split_training_test(n_tr, subkey)
        _, subkey = jax.random.split(subkey)  # new key
        batches, n_batches = get_batches(D_tr, batch_size, subkey)
        _, subkey = jax.random.split(subkey)  # new key
        save_tr_and_val_data(files, D_tr, D_val, n_batches)

    return D_tr, D_val, batches, n_batches, subkey


def data_normalization(y: Any):
    """Data normalization

    Args:
        y (Any): array

    Returns:
        _type_: mean and standard deviation
    """
    mu = jnp.mean(y)
    std = jnp.std(y)
    return mu, std


def batch_to_list_class(batch: Any, obs: str = 'homo_lumo') -> Any:
    """Transforms batch to a list of molecules

    Args:
        batch (Any): batch of molecules
        obs (str, optional): target observable. Defaults to 'homo_lumo'.

    Returns:
        Any: _description_
    """
    #     pytree to class-myMolecule
    batch = batch_to_list(batch)
    batch_ = []
    for b in batch:
        m = myMolecule(
            id0=b["id"],
            smiles=b["smiles"],
            atom_types=b["atom_types"],
            connectivity_matrix=b["conectivity_matrix"],
            homo_lumo_gap_ref=b["homo_lumo_grap_ref"],
            polarizability_ref=b['polarizability_ref'],
            xyz=b['xyz'],
            dm=b["dm"],
        )
        if obs.lower() == 'polarizability' or obs.lower() == "pol":
            m.get_dm_AA_to_Bohr()
            m.get_xyz_AA_to_Bohr()

        batch_.append(m)
    return batch_


def batch_to_list(batch: Any) -> Any:
    """Atom types of a batch to list

    Args:
        batch (Any): batch

    Returns:
        Any: transformed list
    """
    # numpy array to list
    batch = batch.tolist()
    for b in batch:
        if not isinstance(b["atom_types"], list):
            b["atom_types"] = b["atom_types"].tolist()
    return batch


def save_tr_and_val_data(files: dict, D_tr: Any, D_val: Any, n_batches: int) -> None:
    """Save Training and Validation data to a file

    Args:
        files (dict): dictionary with files name
        D_tr (Any): Training data
        D_val (Any): Validation data
        n_batches (int): number of batches
    """
    file = files["f_data"]
    D = {
        "Training": D_tr,
        "Validation": D_val,
        "n_batches": n_batches,
    }
    jnp.save(file, D, allow_pickle=True)


def save_tr_and_val_loss(files: dict, loss_tr: float, loss_val: float, n_epochs: int) -> None:
    """Save loss values to a file

    Args:
        files (dict): dictionary with files name
        loss_tr (float): loss training values
        loss_val (float): validation training values
        n_epochs (int): epochs
    """
    epochs = jnp.arange(n_epochs + 1)
    loss_tr_ = jnp.asarray(loss_tr).ravel()
    loss_val_ = jnp.asarray(loss_val).ravel()

    if os.path.isfile(files["f_loss_opt"]):
        pre_epochs, pre_loss_tr, pre_loss_val = load_pre_opt_params(files)
        loss_tr_ = jnp.append(loss_tr_, pre_loss_tr)
        loss_val_ = jnp.append(loss_val_, pre_loss_val)
        # epochs = jnp.arange(0,pre_epochs.shape[0] + epochs.shape[0])
        epochs = jnp.arange(loss_tr_.shape[0])

    D = {
        "epoch": epochs,
        "loss_tr": loss_tr_,
        "loss_val": loss_val_,
        # 'loss_test':loss_test,
    }
    jnp.save(files["f_loss_opt"], D, allow_pickle=True)
