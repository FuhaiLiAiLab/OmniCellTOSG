import requests
from urllib.parse import quote
from rdkit import Chem
import torch

def pubchem_name_to_smiles(name: str) -> str:
    # Prefer IsomericSMILES (keeps stereochemistry), then fall back
    props = "IsomericSMILES,CanonicalSMILES,ConnectivitySMILES"
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/property/{props}/JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()["PropertyTable"]["Properties"][0]

    smiles = d.get("IsomericSMILES") or d.get("CanonicalSMILES") or d.get("ConnectivitySMILES")
    if not smiles:
        raise ValueError(f"No SMILES found for {name}. Returned keys: {list(d.keys())}")
    return smiles

def rdkit_to_pyg_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES:\n{smiles}")

    atom_feats = []
    for a in mol.GetAtoms():
        atom_feats.append([
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetIsAromatic()),
            a.GetTotalNumHs(),
        ])
    x = torch.tensor(atom_feats, dtype=torch.float)

    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index

name = "Etoposide"
smiles = pubchem_name_to_smiles(name)
print("SMILES:", smiles)
x, edge_index = rdkit_to_pyg_graph(smiles)
print("x shape:", x.shape)
print("All atom features:\n", x)
print("edge_index shape:", edge_index.shape)
print("All edges:\n", edge_index)
