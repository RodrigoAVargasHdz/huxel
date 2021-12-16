import jax
import jax.numpy as jnp

import chex


@chex.dataclass
class myMolecule:
    """
    Basic class for individual molecule
    """

    id: str
    smiles: str
    atom_types: chex.ArrayNumpy
    conectivity_matrix: chex.ArrayNumpy
    homo_lumo_grap_ref: chex.Array
    dm: chex.Array


# if __name__ == "__main__":

# atom_types = ["C", "C", "C", "C"]

# conectivity_matrix = jnp.array(
#     [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=int
# )
# homo_lumo_grap_ref = 1.0

# molec = myMolecule("caca", atom_types, conectivity_matrix, 2.0)
# molecs = [molec, molec]
# print(molecs[0].homo_lumo_grap_ref)
