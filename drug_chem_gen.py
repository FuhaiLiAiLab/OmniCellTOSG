import requests
from urllib.parse import quote
from rdkit import Chem
import torch
import json

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

drug_name_list = sorted(["Etoposide", "Geminticine", "Methotrexate", "SN-38", "Paclitaxel"])

# Generate dictionary for multiple drugs
drug_dict = {}
for drug_name in drug_name_list:
    try:
        smiles = pubchem_name_to_smiles(drug_name)
        x, edge_index = rdkit_to_pyg_graph(smiles)
        drug_dict[drug_name] = (smiles, x, edge_index)
        print(f"✓ Successfully processed {drug_name}")
    except Exception as e:
        print(f"✗ Failed to process {drug_name}: {e}")

print(f"\nSuccessfully processed {len(drug_dict)}/{len(drug_name_list)} drugs")
print(f"Drug names in dictionary: {list(drug_dict.keys())}")

# Convert to JSON-serializable format
drug_dict_json = {}
for drug_name, (smiles, x, edge_index) in drug_dict.items():
    drug_dict_json[drug_name] = {
        "smiles": smiles,
        "atom_features": x.tolist(),
        "edge_index": edge_index.tolist()
    }

# Save to JSON file
with open("drug_database.json", "w") as f:
    json.dump(drug_dict_json, f, indent=2)

print(f"\n✓ Saved drug database to drug_database.json")