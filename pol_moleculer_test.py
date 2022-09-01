import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

all_smiles = [
    "c1ccccc1",
    "C1=CC=C2C=CC=CC2=C1",
    "C1=CC=C2C=C3C=CC=CC3=CC2=C1",
    "C12=CC=CC=C1C=C3C(C=C(C=CC=C4)C4=C3)=C2",
    "C12=CC=CC=C1C=C3C(C=C(C=C(C=CC=C4)C4=C5)C5=C3)=C2",
    "C12=CC=CC=C1C=C3C(C=C(C=C(C=C(C=CC=C4)C4=C5)C5=C6)C6=C3)=C2",
]

R = {}
for i,m in enumerate(all_smiles):
    mol = Chem.MolFromSmiles(m)
    # Chem.AllChem.Compute2DCoords(mol, bondLength=1.4)
    # input_data, mask = process_rdkit_mol(mol)
    # xyz = input_data.coordinates
    Chem.AllChem.EmbedMolecule(mol)
    xyz_w_atoms = Chem.MolToXYZBlock(mol)
    xyz_w_atoms = list(xyz_w_atoms.split("\n"))
    xyz = []

    dm = Chem.GetAdjacencyMatrix(mol)

    for l in xyz_w_atoms[1:]:
        l = list(l.split(" "))
        xyz_i = []
        if l[0] != 'H':
            for li in l:
                if li != "":
                    xyz_i.append(li)
        if len(xyz_i) > 0:  
            xyz.append(xyz_i)
    print(i,m)
    # print(xyz)
    r = {'smile':m, 'xyz':xyz, 'dm': dm}
    # print({f'l_{i}':r})
    R[i] = r 
    print(R)

print(R)
a_file = open("molecule_test.pkl", "wb")
pickle.dump(R, a_file)
a_file.close()
        

