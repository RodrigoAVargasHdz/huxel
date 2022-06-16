import jax
import jax.numpy as jnp
from typing import Any

import chex
from huxel.parameters import Bohr_to_AA

class myMolecule:
    """
    Basic class for individual molecule
    """

    def __init__(
        self,
        id: int,
        smiles: str,
        atom_types: list,
        conectivity_matrix: Any = jnp.ones(1),
        homo_lumo_grap_ref: float = 1.0,
        polarizability_ref: float = 1.0,
        xyz: Any = jnp.ones((2,3)),
        dm: Any = None
    ):
        self.id = id
        self.smiles = smiles
        self.atom_types = atom_types
        self.conectivity_matrix = conectivity_matrix
        self.homo_lumo_grap_ref = homo_lumo_grap_ref
        self.polarizability_ref = polarizability_ref
        self.xyz = xyz
        self.dm = dm

    def get_dm(self):
        z = self.xyz[:, None] - self.xyz[None, :]
        self.dm = jnp.linalg.norm(z, axis=2)  # compute the bond length

    def get_dm_AA_to_Bohr(self):
        z = self.xyz[:, None] - self.xyz[None, :]
        dm = jnp.linalg.norm(z, axis=2)  # compute the bond length      
        self.dm = jnp.divide(dm, Bohr_to_AA) # Bohr -> AA

    def get_xyz_AA_to_Bohr(self):    
        self.xyz_Bohr = jnp.divide(self.xyz, Bohr_to_AA) # Bohr -> AA
        


# if __name__ == "__main__":

# atom_types = ["C", "C", "C", "C"]

# conectivity_matrix = jnp.array(
#     [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=int
# )
# homo_lumo_grap_ref = 1.0

# molec = myMolecule("caca", atom_types, conectivity_matrix, 2.0)
# molecs = [molec, molec]
# print(molecs[0].homo_lumo_grap_ref)
