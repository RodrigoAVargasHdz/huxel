import jax.numpy as jnp
from typing import Any

from huxel.parameters import Bohr_to_AA


class myMolecule:
    """
    Basic class for individual molecule
    """

    def __init__(
        self,
        id0: int,
        smiles: str,
        atom_types: list,
        connectivity_matrix: Any = jnp.ones(1),
        homo_lumo_grap_ref: float = 1.0,
        polarizability_ref: float = 1.0,
        xyz: Any = jnp.ones((2, 3)),
        dm: Any = None
    ):
        """_summary_

        Args:
            id (int): identification
            smiles (str): SMILE
            atom_types (list): list of atoms
            connectivity_matrix (Any, optional): connectivity matrix. Defaults to jnp.ones(1).
            homo_lumo_grap_ref (float, optional): True HOMO-LUMO gap. Defaults to 1.0.
            polarizability_ref (float, optional): True polarizability. Defaults to 1.0.
            xyz (Any, optional): XYZ matrix. Defaults to jnp.ones((2, 3)).
            dm (Any, optional): Distance matrix. Defaults to None.
        """
        self.id = id0
        self.smiles = smiles
        self.atom_types = atom_types
        self.connectivity_matrix = connectivity_matrix
        self.homo_lumo_grap_ref = homo_lumo_grap_ref
        self.polarizability_ref = polarizability_ref
        self.xyz = xyz
        self.dm = dm

    def get_dm(self):
        """distance matrix
        """
        z = self.xyz[:, None] - self.xyz[None, :]
        self.dm = jnp.linalg.norm(z, axis=2)  # compute the bond length

    def get_dm_AA_to_Bohr(self):
        """distance matrix in Armstrong
        """
        z = self.xyz[:, None] - self.xyz[None, :]
        dm = jnp.linalg.norm(z, axis=2)  # compute the bond length
        self.dm = jnp.divide(dm, Bohr_to_AA)  # AA --> Bohr

    def get_xyz_AA_to_Bohr(self):
        """XYZ matrix in Bohr
        """
        self.xyz_Bohr = jnp.divide(self.xyz, Bohr_to_AA)  # AA --> Bohr


# if __name__ == "__main__":

# atom_types = ["C", "C", "C", "C"]

# connectivity_matrix = jnp.array(
#     [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=int
# )
# homo_lumo_grap_ref = 1.0

# molec = myMolecule("caca", atom_types, connectivity_matrix, 2.0)
# molecs = [molec, molec]
# print(molecs[0].homo_lumo_grap_ref)
