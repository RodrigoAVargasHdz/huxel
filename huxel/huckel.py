import jax
import jax.numpy as jnp
from jax import random
from jax import jit,vmap,lax,value_and_grad,jacfwd
from jax.tree_util import tree_flatten, tree_unflatten, tree_multimap

from huxel.parameters import H_X, N_ELECTRONS, H_X, H_XY
from huxel.molecule import myMolecule

def linear_model_pred(params_tot,batch,f_beta):
    params_lr, params = params_tot
    alpha,beta = params_lr

    z_pred,y_true = f_homo_lumo_gap_batch(params,batch,f_beta)
    y_pred = beta*z_pred + alpha
    return y_pred,z_pred,y_true

def f_homo_lumo_gap_batch(params,batch,f_beta):
    y_pred = jnp.ones(1)
    y_true = jnp.ones(1)
    for m in batch:
        yi,_ = f_homo_lumo_gap(params,m,f_beta)
        y_pred = jnp.append(y_pred,yi)   
        y_true = jnp.append(y_true,m.homo_lumo_grap_ref)   
    return y_pred[1:],y_true[1:]

def f_homo_lumo_gap(params,molecule,f_beta):
    # atom_types,conectivity_matrix = molecule
    h_m,electrons = _construct_huckel_matrix(params,molecule,f_beta)
    e_,_ = _solve(h_m)

    n_orbitals = h_m.shape[0]
    occupations, spin_occupations, n_occupied, n_unpaired = _set_occupations(jax.lax.stop_gradient(electrons),jax.lax.stop_gradient(e_),jax.lax.stop_gradient(n_orbitals))
    idx_temp = jnp.nonzero(occupations)[0]
    homo_idx = jnp.argmax(idx_temp)
    lumo_idx = homo_idx + 1
    homo_energy = e_[homo_idx]
    lumo_energy = e_[lumo_idx]
    val = lumo_energy - homo_energy
    return val,(h_m,e_)
# -------
def _construct_huckel_matrix(params,molecule,f_beta):
    # atom_types,conectivity_matrix = molecule 
    atom_types = molecule.atom_types
    conectivity_matrix = molecule.conectivity_matrix
    dm = molecule.dm
    # atom_types = molecule['atom_types']
    # conectivity_matrix = molecule['conectivity_matrix']
    h_x, h_xy, r_xy, y_xy = params


    huckel_matrix = jnp.zeros_like(conectivity_matrix,dtype=jnp.float32)
    # off diagonal terms
    for i,j in zip(*jnp.nonzero(conectivity_matrix)): 
        atom_type_i = atom_types[i]
        atom_type_j = atom_types[j]
        key = frozenset([atom_type_i, atom_type_j])

        beta_ = f_beta(h_xy[key],r_xy[key],y_xy[key],dm[i,j])
        
        huckel_matrix = huckel_matrix.at[i,j].set(beta_)

    # diagonal terms
    diag = jnp.stack([h_x[c] for c in atom_types])
    huckel_matrix = huckel_matrix + jnp.diag(diag.ravel())
    
    electrons = _electrons(atom_types)

    return huckel_matrix, electrons

def _electrons(atom_types):
    return jnp.stack([N_ELECTRONS[atom_type] for atom_type in atom_types])

def _solve(huckel_matrix):
    eig_vals,eig_vects = jnp.linalg.eigh(huckel_matrix)
    return eig_vals[::-1],eig_vects.T[::-1, :]

def _get_multiplicty(n_electrons):
    return (n_electrons % 2) + 1

def _set_occupations(electrons,energies,n_orbitals):
    charge = 0
    n_dec_degen = 3
    n_electrons = jnp.sum(electrons) - charge
    multiplicity = _get_multiplicty(n_electrons)
    n_excess_spin = multiplicity - 1

    # Determine number of singly and doubly occupied orbitals.
    n_doubly = int((n_electrons - n_excess_spin) / 2)
    n_singly = n_excess_spin

    # Make list of electrons to distribute in orbitals
    all_electrons = [2] * n_doubly + [1] * n_singly

    # Set up occupation numbers
    occupations = jnp.zeros(n_orbitals,dtype=jnp.int32)
    spin_occupations = jnp.zeros(n_orbitals,dtype=jnp.int32)

    # Loop over unique rounded orbital energies and degeneracies and fill with
    # electrons
    energies_rounded = energies.round(n_dec_degen)
    unique_energies, degeneracies = jnp.unique(energies_rounded, return_counts=True)
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
        occupations = occupations.at[energies_rounded == energy].add(electrons_ / degeneracy)

        # spin_occupations[np.where(energies_rounded == energy)] += (spin_electrons / degeneracy)
        spin_occupations = occupations.at[energies_rounded == energy].add(spin_electrons_  / degeneracy)

    n_occupied = jnp.count_nonzero(occupations)
    n_unpaired = int(
        jnp.sum(occupations[:n_occupied][occupations[:n_occupied] != 2])
        )
    return occupations, spin_occupations, n_occupied, n_unpaired

# ------------------------
# TEST

def update_params(p, g,alpha=0.1):
    inner_sgd_fn = lambda g, params: (params - alpha*g)
    return tree_multimap(inner_sgd_fn, g, p)

def main_test():
    h_x = H_X
    h_xy = H_XY
    params = (h_x, h_xy)

    atom_types = ['C', 'C', 'C', 'C']

    conectivity_matrix = jnp.array([[0, 1, 0, 0],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0]],dtype=int)

    molec = myMolecule('test',atom_types,conectivity_matrix,1.)
      
    # test single molecule
    v,g = value_and_grad(f_homo_lumo_gap,has_aux=True,)(params,molec)
    print('HOMO-LUMO')
    homo_lumo_val, _ = v
    print(homo_lumo_val)
    print('GRAD HOMO-LUMO')
    print(g)


if __name__ == "__main__":

    main_test()    

