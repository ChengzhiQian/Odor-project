# src/utils/visualize_graph.py
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem

from src.datasets.mol_dataset import (
    load_molecule_dataset,
    smiles_to_pyg_graph_deepchem,
    decode_atom_features_deepchem,
)

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "data" / "example"
OUT_DIR.mkdir(exist_ok=True)


def visualize_graph_only(idx: int = 0):
    """
    Visualize the “graph structure” of the idxth molecule in train_dataset:
    - Node labels: Atomic number: Element symbol (e.g., 0:C)
    - Edges: Drawn according to PyG's edge_index / RDKit key connectivity.
    """
    LABEL_COLS = None
    csv_path = DATA_PROCESSED_DIR / "train_dataset.csv"
    data_list = load_molecule_dataset(
        csv_path=str(csv_path),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_pyg_graph_deepchem,
    )

    data = data_list[idx]
    smiles = data.smiles
    print(f"Visualizing sample #{idx}, SMILES = {smiles}")
    print("x shape:", data.x.shape)
    print("edge_index shape:", data.edge_index.shape)

    # Use RDKit to parse it to obtain the atomic symbols.
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    # Building a graph using NetworkX
    G = nx.Graph()
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        G.add_node(i, symbol=atom.GetSymbol())

    # Use RDKit keys to add edges (you can also use data.edge_index)
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        G.add_edge(u, v)

    # Draw the picture
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(G, seed=42)
    labels = {i: f"{i}:{G.nodes[i]['symbol']}" for i in G.nodes}
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=400,
        font_size=8,
    )
    plt.title(f"Graph view (idx={idx})")

    out_path = OUT_DIR / f"mol_{idx}_graph.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"图结构已保存到: {out_path}")

def inspect_features(idx: int = 0):
    LABEL_COLS = None
    csv_path = DATA_PROCESSED_DIR / "train_dataset.csv"
    data_list = load_molecule_dataset(
        csv_path=str(csv_path),
        smiles_col="nonStereoSMILES",
        label_cols=LABEL_COLS,
        smiles_to_graph_fn=smiles_to_pyg_graph_deepchem,
    )

    data = data_list[idx]
    mol = Chem.MolFromSmiles(data.smiles)

    print(f"Sample #{idx}, SMILES = {data.smiles}")
    print("x shape (num_atoms, feat_dim):", data.x.shape)
    print("edge_index shape:", data.edge_index.shape)

    for i, atom in enumerate(mol.GetAtoms()):
        print(f"\nAtom {i} ({atom.GetSymbol()}):")
        print("  feature dim:", data.x[i].numel())
        print("  feature[:10]:", data.x[i][:31].tolist())

def inspect_decoded(idx: int = 0):
    csv_path = DATA_PROCESSED_DIR / "train_dataset.csv"
    data_list = load_molecule_dataset(
        csv_path=str(csv_path),
        smiles_col="nonStereoSMILES",
        label_cols=None,
        smiles_to_graph_fn=smiles_to_pyg_graph_deepchem,
    )

    data = data_list[idx]
    mol = Chem.MolFromSmiles(data.smiles)

    print(f"Sample #{idx}, SMILES = {data.smiles}")
    for i, atom in enumerate(mol.GetAtoms()):
        d = decode_atom_features_deepchem(data.x[i])
        print(f"\nAtom {i} ({atom.GetSymbol()}):")
        for k, v in d.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    #Draw the picture of atom
    #visualize_graph_only(idx=0)

    #Show the features
    #inspect_features(idx=0)
    inspect_decoded(0)
