from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, lax
from jax.tree_util import tree_flatten, tree_unflatten, tree_multimap

from huxel.parameters import H_X, N_ELECTRONS, H_X, H_XY
from huxel.molecule import myMolecule
from huxel.utils import normalize_params_wrt_C, normalize_params_polarizability


jax.config.update("jax_enable_x64", True)
jax.config.update('jax_disable_jit', True)

# -------


def homo_lumo_pred(params: dict, batch: Any, f_beta: Callable) -> Tuple:
    """Linear transformation of the predicted HOMO-LUMO gap for a batch of molecules 

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (Callable): atom-atom interaction

    Returns:
        Tuple: Linearly transformed predicted HOMO-LUMO gap
    """
    z_pred, y_true = f_homo_lumo_batch(params, batch, f_beta)
    y_pred = params["hl_params"]["a"]*z_pred + params["hl_params"]["b"]
    return y_pred, z_pred, y_true


def f_homo_lumo_batch(params: dict, batch: Any, f_beta: Callable) -> Tuple:
    """Computes the HOMO-LUMO gap for a batch of molecules 

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (Callable): atom-atom interaction

    Returns:
        Tuple: Predicted HOMO-LUMO gap
    """
    y_pred = jnp.ones(1)
    y_true = jnp.ones(1)
    for m in batch:
        yi, _ = f_homo_lumo(params, m, f_beta)
        y_pred = jnp.append(y_pred, yi)
        y_true = jnp.append(y_true, m.homo_lumo_grap_ref)
    return y_pred[1:], y_true[1:]


def f_homo_lumo(params: dict, molecule, f_beta) -> Tuple:
    """HOMO-LUMO gap per single molecule

    Args:
        params (dict): Hückel model's parameters
        molecule (_type_): single molecule
        f_beta (_type_): atom-atom interaction

    Returns:
        Tuple: predicted HOMO-LUMO gap, Hückel matrix and eigenvalues
    """
    # atom_types,connectivity_matrix = molecule
    h_m, electrons = _construct_huckel_matrix(params, molecule, f_beta)
    e_, _ = _solve(h_m)

    n_orbitals = h_m.shape[0]
    occupations, spin_occupations, n_occupied, n_unpaired = _set_occupations(
        jax.lax.stop_gradient(electrons), jax.lax.stop_gradient(e_), jax.lax.stop_gradient(n_orbitals))
    idx_temp = jnp.nonzero(occupations)[0]
    homo_idx = jnp.argmax(idx_temp)
    lumo_idx = homo_idx + 1
    homo_energy = e_[homo_idx]
    lumo_energy = e_[lumo_idx]
    val = lumo_energy - homo_energy
    return val, (h_m, e_)
# -------


def polarizability_pred(params: dict, batch: Any, f_beta: Callable, external_field: Any = None) -> Tuple:
    """Molecular polarizability

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (Callable): atom-atom interaction_
        external_field (Any, optional): External field. Defaults to None.

    Returns:
        Tuple: Predicted polarizability, Reference polarizability
    """
    z_pred, y_true = f_polarizability_batch(
        params, batch, f_beta, external_field)
    y_pred = z_pred  # + params["pol_params"]["b"]
    return y_pred, z_pred, y_true


def f_polarizability_batch(params: dict, batch: Any, f_beta: callable, external_field: Any = None) -> Tuple:
    """Molecular polarizability for a batch of molecules

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (callable): atom-atom interaction_
        external_field (Any, optional): External field. Defaults to None.

    Returns:
        Tuple: Predicted polarizability, Reference polarizability
    """
    y_pred = jnp.ones(1)
    y_true = jnp.ones(1)
    for m in batch:
        yi = f_polarizability(params, m, f_beta, external_field)
        y_pred = jnp.append(y_pred, yi)
        y_true = jnp.append(y_true, m.polarizability_ref)
    return y_pred[1:], y_true[1:]


def f_polarizability(params: dict, molecule: Any, f_beta: callable, external_field: Any = None) -> Any:
    """Molecular polarizability

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (callable): atom-atom interaction_
        external_field (Any, optional): External field. Defaults to None.

    Returns:
        Any: Predicted polarizability
    """
    polarizability_tensor = jax.hessian(f_energy, argnums=(3))(
        params, molecule, f_beta, external_field)
    polarizability = (1/3.)*jnp.trace(polarizability_tensor)
    return polarizability


def f_energy(params: dict, molecule: Any, f_beta: Callable, external_field: Any = None) -> Any:
    """Hückel model's energy

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (callable): atom-atom interaction_
        external_field (Any, optional): External field. Defaults to None.

    Returns:
        Any: Energy
    """
    h_m, electrons = _construct_huckel_matrix(params, molecule, f_beta)

    if external_field != None:
        h_m_field = _construct_huckel_matrix_field(molecule, external_field)
        h_m = h_m + h_m_field

    e_, _ = _solve(h_m)

    n_orbitals = h_m.shape[0]
    occupations, spin_occupations, n_occupied, n_unpaired = _set_occupations(
        jax.lax.stop_gradient(electrons), jax.lax.stop_gradient(e_), jax.lax.stop_gradient(n_orbitals))
    return jnp.dot(occupations, e_)

# --------------------------------------------------------


def _construct_huckel_matrix(params: dict, molecule, f_beta: Callable, bool_AA_to_Bhor: bool = True) -> Tuple:
    """Hückel matrix (batch)

    Args:
        params (dict): Hückel model's parameters
        batch (Any): list of molecules (batch)
        f_beta (callable): atom-atom interaction_
        bool_AA_to_Bhor (bool, optional): Armstrong (AA) or Bhor units

    Returns:
        Tuple: Hückel matrix, number of electrons
    """
    atom_types = molecule.atom_types
    conectivity_matrix = molecule.connectivity_matrix
    # CHECK THIS FOR POLARIZABILITY UNITS PROBLEM OR f_BETA(R) FUNCTIONS!!
    dm = molecule.dm

    huckel_matrix = jnp.zeros_like(conectivity_matrix, dtype=jnp.float32)
    # off diagonal terms
    for i, j in zip(*jnp.nonzero(conectivity_matrix)):
        atom_type_i = atom_types[i]
        atom_type_j = atom_types[j]
        key = frozenset([atom_type_i, atom_type_j])

        beta_ = f_beta(params['h_xy'][key], params['r_xy']
                       [key], params['y_xy'][key], dm[i, j])

        huckel_matrix = huckel_matrix.at[i, j].set(beta_)

    # diagonal terms
    diag = jnp.stack([params['h_x'][c] for c in atom_types])
    huckel_matrix = huckel_matrix + jnp.diag(diag.ravel())

    electrons = _electrons(atom_types)

    return huckel_matrix, electrons


def _construct_huckel_matrix_field(molecule: Any, field: Any) -> Any:
    """Diagonal elements of Hückel matrix in the presence of an external field

    Args:
        molecule (Any):
        field (Any): 

    Returns:
        Any: Diagonal of the Hückel matrix in the presence of an external field
    """
    # atom_types = molecule.atom_types
    # xyz = molecule.xyz
    xyz = molecule.xyz_Bohr

    # diagonal terms
    diag_ri = jnp.asarray([jnp.diag(xyz[:, i])for i in range(3)])
    def field_r(fi, xi): return fi*xi
    diag_ri_tensor = vmap(field_r, in_axes=(0, 0))(field, diag_ri)
    diag_ri = jnp.sum(diag_ri_tensor, axis=0)
    return diag_ri


def _electrons(atom_types: list) -> Any:
    """number of electrons in each Hückel orbital

    Args:
        atom_types (list): type of atoms 

    Returns:
        Any: number of electron in each atom
    """
    return jnp.stack([N_ELECTRONS[atom_type] for atom_type in atom_types])


def _solve(huckel_matrix: Any) -> Tuple:
    """Diagonalization of the Hückel matrix

    Args:
        huckel_matrix (Any):  Hückel matrix

    Returns:
        Tuple: Eigenvalues and Eigenvectors
    """
    eig_vals, eig_vects = jnp.linalg.eigh(huckel_matrix)
    return eig_vals[::-1], eig_vects.T[::-1, :]


def _get_multiplicity(n_electrons: int) -> Any:
    """Multiplicity

    Args:
        n_electrons (int): number of electrons

    Returns:
        Any: multiplicity
    """
    return (n_electrons % 2) + 1


def _set_occupations(electrons: int, energies: Any, n_orbitals: int) -> Tuple:
    """Occupation

    Args:
        electrons (int): number of electrons
        energies (Any): Hückel's eigenvalues
        n_orbitals (int): number of orbitals

    Returns:
        Tuple: occupation, spin occupation, number of occupied orbitals, number of unpair electrons
    """
    charge = 0
    n_dec_degen = 3
    n_electrons = jnp.sum(electrons) - charge
    multiplicity = _get_multiplicity(n_electrons)
    n_excess_spin = multiplicity - 1

    # Determine number of singly and doubly occupied orbitals.
    n_doubly = int((n_electrons - n_excess_spin) / 2)
    n_singly = n_excess_spin

    # Make list of electrons to distribute in orbitals
    all_electrons = [2] * n_doubly + [1] * n_singly

    # Set up occupation numbers
    occupations = jnp.zeros(n_orbitals, dtype=jnp.int32)
    spin_occupations = jnp.zeros(n_orbitals, dtype=jnp.int32)

    # Loop over unique rounded orbital energies and degeneracies and fill with
    # electrons
    energies_rounded = energies.round(n_dec_degen)
    unique_energies, degeneracies = jnp.unique(
        energies_rounded, return_counts=True)
    for energy, degeneracy in zip(jnp.flip(unique_energies), jnp.flip(degeneracies)):
        if len(all_electrons) == 0:
            break

        # Determine number of electrons with and without excess spin.
        electrons_ = 0
        spin_electrons_ = 0
        for _ in range(degeneracy):
            if len(all_electrons) > 0:
                pop_electrons = all_electrons.pop(0)
                electrons_ += pop_electrons
                if pop_electrons == 1:
                    spin_electrons_ += 1

        # Divide electrons evenly among orbitals
        # occupations[jnp.where(energies_rounded == energy)] += electrons / degeneracy
        occupations = occupations.at[energies_rounded == energy].add(
            electrons_ / degeneracy)

        # spin_occupations[np.where(energies_rounded == energy)] += (spin_electrons / degeneracy)
        spin_occupations = occupations.at[energies_rounded == energy].add(
            spin_electrons_ / degeneracy)

    n_occupied = jnp.count_nonzero(occupations)
    n_unpaired = int(
        jnp.sum(occupations[:n_occupied][occupations[:n_occupied] != 2])
    )
    return occupations, spin_occupations, n_occupied, n_unpaired
