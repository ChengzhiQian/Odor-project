# src/datasets/mol_dataset.py
import csv
from typing import List, Optional, Callable

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType, ChiralType, BondStereo

import numpy as np
from collections import deque

# =========================
# General one-hot encoding & feature space
# =========================

# Deepchem
ALLOWABLE_ATOMS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
ALLOWABLE_DEGREES = [0, 1, 2, 3, 4, 5]
ALLOWABLE_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
ALLOWABLE_HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
ALLOWABLE_BOND_TYPES = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]



# Chemprop
CHEMPROP_ATOMS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I', 'other']
CHEMPROP_DEGREES = [0, 1, 2, 3, 4, 5]
CHEMPROP_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHEMPROP_NUM_HS = [0, 1, 2, 3, 4]

CHEMPROP_CHIRAL_TAGS = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
]

CHEMPROP_HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    'other'
]

CHEMPROP_BOND_STEREO = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
    'other'
]


def one_hot_encoding(x, allowable_set):
    """if it's not in allowable_set，reflect it to 'other'。"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

# =========================
# DeepChem-style SMILES → Graph
# =========================

# DeepChem-style Atom Features
def atom_to_feature_vector_deepchem(atom: Chem.rdchem.Atom) -> List[float]:
    """
    DeepChem/MoleculeNet style atom features：
    - Atom type 原子类型
    - Atom degree(number of adjacent atoms) 原子度（邻接原子个数）
    - Formal charge 形式电荷
    - Hybridization state 杂化态
    - Aromatic 芳香性
    - Whether in a ring 是否在环
    """
    atom_symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hyb = atom.GetHybridization()
    aromatic = atom.GetIsAromatic()
    in_ring = atom.IsInRing()

    features: List[float] = []
    # Atom type one-hot
    features += one_hot_encoding(atom_symbol, ALLOWABLE_ATOMS + ['other'])
    # Degree one-hot
    features += one_hot_encoding(degree, ALLOWABLE_DEGREES + ['other'])
    # Formal charge one-hot
    features += one_hot_encoding(formal_charge, ALLOWABLE_FORMAL_CHARGES + ['other'])
    # Hybridization one-hot
    features += one_hot_encoding(hyb, ALLOWABLE_HYBRIDIZATIONS + ['other'])
    # Aromatic & Whether in a ring
    features.append(int(aromatic))
    features.append(int(in_ring))

    return features

# DeepChem-style Bond Features
def bond_to_feature_vector_deepchem(bond: Chem.rdchem.Bond) -> List[float]:
    """
    DeepChem-style bond features：
    - Bond type 键类型
    - Conjugated 共轭
    - Whether in a ring 是否在环
    """
    bt = bond.GetBondType()
    is_conj = bond.GetIsConjugated()
    in_ring = bond.IsInRing()

    features: List[float] = []
    features += one_hot_encoding(bt, ALLOWABLE_BOND_TYPES + ['other'])
    features.append(int(is_conj))
    features.append(int(in_ring))

    return features

# Show the Deepchem-style feature details
def decode_atom_features_deepchem(feat) -> dict:
    """
    Decode DeepChem/MoleculeNet-style atomic feature vectors, returning a readable dict.
    feat: 1D tensor or list whose length is 31
    """
    if isinstance(feat, torch.Tensor):
        v = feat.detach().cpu().numpy()
    else:
        v = feat
    v = list(v)

    # 1) Atom type
    atom_slice = v[0:10]
    atom_idx = int(max(range(len(atom_slice)), key=lambda i: atom_slice[i]))
    atom_type = (ALLOWABLE_ATOMS + ['other'])[atom_idx]

    # 2) Degree
    degree_slice = v[10:17]
    degree_idx = int(max(range(len(degree_slice)), key=lambda i: degree_slice[i]))
    degree_val = (ALLOWABLE_DEGREES + ['other'])[degree_idx]

    # 3) Formal charge
    charge_slice = v[17:23]
    charge_idx = int(max(range(len(charge_slice)), key=lambda i: charge_slice[i]))
    formal_charge = (ALLOWABLE_FORMAL_CHARGES + ['other'])[charge_idx]

    # 4) Hybridization
    hybrid_slice = v[23:29]
    hybrid_idx = int(max(range(len(hybrid_slice)), key=lambda i: hybrid_slice[i]))
    hybrid_val = (ALLOWABLE_HYBRIDIZATIONS + ['other'])[hybrid_idx]

    if hybrid_val == 'other':
        hybrid_str = 'other'
    else:
        # HybridizationType.SP -> 'SP'
        hybrid_str = str(hybrid_val).split('.')[-1]

    # 5) aromatic & whether in a ring
    aromatic = bool(round(v[29]))
    in_ring = bool(round(v[30]))

    return {
        "element": atom_type,
        "degree": degree_val,
        "formal_charge": formal_charge,
        "hybrid": hybrid_str,
        "aromatic": aromatic,
        "in_ring": in_ring,
    }

def decode_bond_features_deepchem(feat) -> dict:
    """
    Decode DeepChem/MoleculeNet-style atomic feature vectors, returning a readable dict.
    - bond_type: SINGLE / DOUBLE / TRIPLE / AROMATIC / other
    - conjugated: 是否共轭
    - in_ring: 是否在环内
    """
    if isinstance(feat, torch.Tensor):
        v = feat.detach().cpu().numpy()
    else:
        v = feat
    v = list(v)

    # 1) bond type one-hot
    # ALLOWABLE_BOND_TYPES + ['other'] 共有 5 个
    bond_slice = v[0:5]
    bond_idx = int(max(range(len(bond_slice)), key=lambda i: bond_slice[i]))
    bond_type_val = (ALLOWABLE_BOND_TYPES + ['other'])[bond_idx]
    if bond_type_val == 'other':
        bond_type_str = 'other'
    else:
        # BondType.SINGLE -> 'SINGLE'
        bond_type_str = str(bond_type_val).split('.')[-1]

    # 2) 是否共轭
    conjugated = bool(round(v[5])) if len(v) > 5 else False
    # 3) 是否在环内
    in_ring = bool(round(v[6])) if len(v) > 6 else False

    return {
        "bond_type": bond_type_str,
        "conjugated": conjugated,
        "in_ring": in_ring,
    }

def inspect_data_deepchem(data: Data, max_edges: int = 50):
    """
    Print DeepChem Style Graph's readable information(Atom + Edge)
    Parameters：
      - data: PyG's Data(from smiles_to_pyg_graph_deepchem)
      - max_edges: avoid it being too long, the total number of edges is 2*max_edges
    """
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)  # double-direction edge
    feat_dim = data.x.size(1)

    print("===== DeepChem Graph Inspect =====")
    if hasattr(data, "smiles"):
        print("SMILES:", data.smiles)
    print(f"Node number: {num_nodes}")
    print(f"Edge number: {num_edges}")
    print(f"Atom feature dimension: {feat_dim}")

    if hasattr(data, "y"):
        print("the shape of label y:", tuple(data.y.shape))

    # ------- Atom -------
    print("\n--- Nodes (Atoms) ---")
    for i in range(num_nodes):
        atom_feat = data.x[i]
        info = decode_atom_features_deepchem(atom_feat)
        print(f"Atom {i}: {info}")

    # ------- Edge -------
    if data.edge_index is None or data.edge_index.numel() == 0:
        print("\n[WARN] There is no edge in the graph(Maybe only one atom?)")
        return

    print("\n--- Edges (Bonds) ---")

    num_print = min(num_edges, max_edges)
    for e_idx in range(num_print):
        src = int(data.edge_index[0, e_idx])
        dst = int(data.edge_index[1, e_idx])

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(0) > e_idx:
            bond_feat = data.edge_attr[e_idx]
            binfo = decode_bond_features_deepchem(bond_feat)
        else:
            binfo = {}

        print(f"Edge {e_idx}: {src} -> {dst}, {binfo}")

    if num_edges > num_print:
        print(f"... ({num_edges - num_print} remaining directed edges have not been printed.)")

# Deepchem-style SMILE->Graph
def smiles_to_pyg_graph_deepchem(smiles: str) -> Optional[Data]:
    """
    Convert SMILES strings into PyG data in the style of DeepChem/MoleculeNet:
    - x: [num_atoms, atom_feat_dim]
    - edge_index: [2, num_edges*2] 无向图（双向有向边）
    - edge_attr: [num_edges*2, bond_feat_dim]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # build atom feature matrix 构建节点特征矩阵
    atom_features_list: List[List[float]] = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector_deepchem(atom))
    x = torch.tensor(atom_features_list, dtype=torch.float)

    # edge features + edge_index 初始化边索引和边特征矩阵
    edge_indices = []
    edge_features_list: List[List[float]] = []

    # Iterate through all keys to construct bidirectional edges in an undirected graph.
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_to_feature_vector_deepchem(bond)

        # undirected graph：i->j, j->i
        edge_indices.append((i, j))
        edge_features_list.append(bf)
        edge_indices.append((j, i))
        edge_features_list.append(bf)

    # Handling the extreme case of “no bonds”
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_to_feature_vector_deepchem(
            Chem.MolFromSmiles('CC').GetBonds()[0]
        ))), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# =========================
# Chemprop / D-MPNN SMILES → Graph
# =========================

# Chemprop-style Atom Features
def atom_to_feature_vector_chemprop(atom: Chem.rdchem.Atom) -> List[float]:
    """
    Chemprop / D-MPNN Style Atom Characteristic：
    - Atom type 原子类型（B,C,N,O,F,Si,P,S,Cl,Br,I,other）
    - Atom degree 原子度（0-5）
    - Formal charge 形式电荷（-2,-1,0,1,2）
    - Chiral label 手性标签（null / CW / CCW / other）
    - Total number of hydrogen atoms 总氢原子数（0-4）
    - Hybridization 杂化态（sp,sp2,sp3,sp3d,sp3d2,other）
    - Aromatic 芳香性（bool）
    - Atom quality 原子质量（real quality /100）
    """
    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    chiral_tag = atom.GetChiralTag()
    num_hs = atom.GetTotalNumHs()
    hyb = atom.GetHybridization()
    aromatic = atom.GetIsAromatic()
    mass = atom.GetMass()

    feats: List[float] = []

    # Atom type one-hot
    feats += one_hot_encoding(symbol, CHEMPROP_ATOMS)
    # Atom degree one-hot
    feats += one_hot_encoding(degree, CHEMPROP_DEGREES + ['other'])
    # Formal charge one-hot
    feats += one_hot_encoding(formal_charge, CHEMPROP_FORMAL_CHARGES + ['other'])
    # Chiral label one-hot
    feats += one_hot_encoding(chiral_tag, CHEMPROP_CHIRAL_TAGS)
    # Total number of hydrogen atoms one-hot
    feats += one_hot_encoding(num_hs, CHEMPROP_NUM_HS + ['other'])
    # Hybridization one-hot
    feats += one_hot_encoding(hyb, CHEMPROP_HYBRIDIZATIONS)
    # Aromatic
    feats.append(int(aromatic))
    # Atom quality
    feats.append(mass * 0.01)

    return feats

# Chemprop-style Bond Features
def bond_to_feature_vector_chemprop(bond: Chem.rdchem.Bond) -> List[float]:
    """
    Chemprop Style edge characteristic：
    - Bond type 键类型（单/双/三/芳香/other）
    - Conjugated 共轭（bool）
    - Whether in a ring 是否在环（bool）
    - Three-dimensional information 立体化信息 stereo（none/any/Z/E/cis/trans/other）
    """
    bt = bond.GetBondType()
    stereo = bond.GetStereo()
    is_conj = bond.GetIsConjugated()
    in_ring = bond.IsInRing()

    feats: List[float] = []
    feats += one_hot_encoding(bt, ALLOWABLE_BOND_TYPES + ['other'])
    feats.append(int(is_conj))
    feats.append(int(in_ring))
    feats += one_hot_encoding(stereo, CHEMPROP_BOND_STEREO)

    return feats

# Show the Chemprop-style feature details
def decode_atom_features_chemprop(feat) -> dict:
    """
    Decode Chemprop / D-MPNN Style's atomic characteristic vectors，return a readable dict。
    feat: 1D tensor or list whose length is around 43
    """
    if isinstance(feat, torch.Tensor):
        v = feat.detach().cpu().numpy()
    else:
        v = feat
    v = list(v)

    idx = 0

    # 1) Atom type
    n_atom = len(CHEMPROP_ATOMS)
    atom_slice = v[idx: idx + n_atom]
    idx += n_atom
    atom_idx = int(max(range(len(atom_slice)), key=lambda i: atom_slice[i]))
    atom_type = CHEMPROP_ATOMS[atom_idx]

    # 2) Degree
    n_degree = len(CHEMPROP_DEGREES) + 1  # + 'other'
    degree_slice = v[idx: idx + n_degree]
    idx += n_degree
    degree_idx = int(max(range(len(degree_slice)), key=lambda i: degree_slice[i]))
    degree_val = (CHEMPROP_DEGREES + ['other'])[degree_idx]

    # 3) Formal charge
    n_charge = len(CHEMPROP_FORMAL_CHARGES) + 1  # + 'other'
    charge_slice = v[idx: idx + n_charge]
    idx += n_charge
    charge_idx = int(max(range(len(charge_slice)), key=lambda i: charge_slice[i]))
    formal_charge = (CHEMPROP_FORMAL_CHARGES + ['other'])[charge_idx]

    # 4) Chiral tag
    n_chiral = len(CHEMPROP_CHIRAL_TAGS)
    chiral_slice = v[idx: idx + n_chiral]
    idx += n_chiral
    chiral_idx = int(max(range(len(chiral_slice)), key=lambda i: chiral_slice[i]))
    chiral_val = CHEMPROP_CHIRAL_TAGS[chiral_idx]
    # change to string
    if isinstance(chiral_val, str):
        chiral_str = chiral_val
    else:
        chiral_str = str(chiral_val).split('.')[-1]  # ChiralType.CHI_XXX -> 'CHI_XXX'

    # 5) Total number of H
    n_hs = len(CHEMPROP_NUM_HS) + 1  # + 'other'
    num_h_slice = v[idx: idx + n_hs]
    idx += n_hs
    num_h_idx = int(max(range(len(num_h_slice)), key=lambda i: num_h_slice[i]))
    num_h_val = (CHEMPROP_NUM_HS + ['other'])[num_h_idx]

    # 6) Hybridization
    n_hyb = len(CHEMPROP_HYBRIDIZATIONS)
    hybrid_slice = v[idx: idx + n_hyb]
    idx += n_hyb
    hybrid_idx = int(max(range(len(hybrid_slice)), key=lambda i: hybrid_slice[i]))
    hybrid_val = CHEMPROP_HYBRIDIZATIONS[hybrid_idx]
    if isinstance(hybrid_val, str):
        hybrid_str = hybrid_val
    else:
        hybrid_str = str(hybrid_val).split('.')[-1]  # HybridizationType.SP2 -> 'SP2'

    # 7) Aromatic (bool)
    aromatic = bool(round(v[idx]))
    idx += 1

    # 8) Atomic mass (scaled)
    mass_scaled = float(v[idx]) if idx < len(v) else 0.0
    mass_approx = mass_scaled * 100.0  # 对应 atom_to_feature_vector_chemprop 里的 mass*0.01

    return {
        "element": atom_type,
        "degree": degree_val,
        "formal_charge": formal_charge,
        "chiral_tag": chiral_str,
        "num_H": num_h_val,
        "hybrid": hybrid_str,
        "aromatic": aromatic,
        "mass_scaled": mass_scaled,
        "mass_approx": mass_approx,
    }

def decode_bond_features_chemprop(feat) -> dict:
    """
    Decode Chemprop Style's atomic characteristic vectors：
    - bond_type: SINGLE / DOUBLE / TRIPLE / AROMATIC / other
    - conjugated: 是否共轭
    - in_ring: 是否在环
    - stereo: STEREONONE / STEREOZ / STEREOE / CIS / TRANS / any / other
    """
    if isinstance(feat, torch.Tensor):
        v = feat.detach().cpu().numpy()
    else:
        v = feat
    v = list(v)

    idx = 0

    # 1) bond type
    n_bt = len(ALLOWABLE_BOND_TYPES) + 1  # + 'other'
    bt_slice = v[idx: idx + n_bt]
    idx += n_bt
    bt_idx = int(max(range(len(bt_slice)), key=lambda i: bt_slice[i]))
    bt_val = (ALLOWABLE_BOND_TYPES + ['other'])[bt_idx]
    if bt_val == 'other':
        bond_type_str = 'other'
    else:
        bond_type_str = str(bt_val).split('.')[-1]  # BondType.SINGLE -> 'SINGLE'

    # 2) conjugated
    conjugated = bool(round(v[idx])) if idx < len(v) else False
    idx += 1

    # 3) in ring
    in_ring = bool(round(v[idx])) if idx < len(v) else False
    idx += 1

    # 4) stereo
    n_st = len(CHEMPROP_BOND_STEREO)
    stereo_slice = v[idx: idx + n_st]
    if len(stereo_slice) == n_st:
        st_idx = int(max(range(len(stereo_slice)), key=lambda i: stereo_slice[i]))
        st_val = CHEMPROP_BOND_STEREO[st_idx]
        if isinstance(st_val, str):
            stereo_str = st_val
        else:
            stereo_str = str(st_val).split('.')[-1]  # BondStereo.STEREOZ -> 'STEREOZ'
    else:
        stereo_str = "unknown"

    return {
        "bond_type": bond_type_str,
        "conjugated": conjugated,
        "in_ring": in_ring,
        "stereo": stereo_str,
    }

def inspect_data_chemprop(data: Data, max_edges: int = 50):
    """
    Print Chemprop Style Graph's readable information(Atom + Edge)
    data: from smiles_to_pyg_graph_chemprop
    max_edges: avoid it being too long, the total number of edges is 2*max_edges
    """
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    feat_dim = data.x.size(1)

    print("===== Chemprop Graph Inspect =====")
    if hasattr(data, "smiles"):
        print("SMILES:", data.smiles)
    print(f"Node number: {num_nodes}")
    print(f"Edge number: {num_edges}")
    print(f"Atom feature dimension: {feat_dim}")

    if hasattr(data, "y"):
        print("the shape of label y:", tuple(data.y.shape))

    # ------- Atom -------
    print("\n--- Nodes (Atoms) ---")
    for i in range(num_nodes):
        atom_feat = data.x[i]
        info = decode_atom_features_chemprop(atom_feat)
        print(f"Atom {i}: {info}")

    # ------- Edge -------
    if data.edge_index is None or data.edge_index.numel() == 0:
        print("\n[WARN] There is no edge in the graph(Maybe only one atom?)")
        return

    print("\n--- Edges (Bonds) ---")
    num_print = min(num_edges, max_edges)
    for e_idx in range(num_print):
        src = int(data.edge_index[0, e_idx])
        dst = int(data.edge_index[1, e_idx])

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(0) > e_idx:
            bond_feat = data.edge_attr[e_idx]
            binfo = decode_bond_features_chemprop(bond_feat)
        else:
            binfo = {}

        print(f"Edge {e_idx}: {src} -> {dst}, {binfo}")

    if num_edges > num_print:
        print(f"... ({num_edges - num_print} remaining directed edges have not been printed.)")

def smiles_to_pyg_graph_chemprop(smiles: str) -> Optional[Data]:
    """
    Chemprop / D-MPNN Style SMILES→Graph：
    - Using Chemprop-style atom & bond features
    - The graph structure remains “each key split into directed edges i->j and j->i.”
      D-MPNN performs message passing along these directed edges within the model.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features: List[List[float]] = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector_chemprop(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge
    edge_indices = []
    edge_features: List[List[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_to_feature_vector_chemprop(bond)

        # i -> j
        edge_indices.append((i, j))
        edge_features.append(bf)
        # j -> i
        edge_indices.append((j, i))
        edge_features.append(bf)

    if len(edge_indices) == 0:
        # in case it has no keys
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # A “dummy” bond is needed to compute the feature dimension.
        dummy_mol = Chem.MolFromSmiles("CC")
        dummy_bond = list(dummy_mol.GetBonds())[0]
        dummy_dim = len(bond_to_feature_vector_chemprop(dummy_bond))
        edge_attr = torch.empty((0, dummy_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# =========================
# Grover SMILES → Graph
# =========================

def compute_shortest_path_matrix(num_atoms: int, bonds) -> np.ndarray:
    """
    Calculate the shortest path distance matrix (N x N) for the undirected graph based on the RDKit bond list.。
    bonds: mol.GetBonds()
    """
    N = num_atoms
    adj = [[] for _ in range(N)]
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    dist = np.full((N, N), fill_value=np.inf, dtype=np.float32)
    for i in range(N):
        # BFS from i
        dq = deque()
        dq.append(i)
        dist[i, i] = 0
        visited = {i}
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    dist[i, v] = dist[i, u] + 1
                    dq.append(v)

    # Typically, the molecule is connected, so there won't be an inf; to be safe, set inf to a large number.
    dist[np.isinf(dist)] = 1e3
    return dist


def compute_laplacian_pos_enc(spatial_adj: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Compute the first k eigenvectors of the Laplacian matrix as node position encoding (N x k).
    Input:
      spatial_adj: (N x N) adjacency Matrix (0/1)
    Return:
      pe: (N x k)，If N < k, only N columns are returned.
    """
    N = spatial_adj.shape[0]
    if N == 0:
        return np.zeros((0, k), dtype=np.float32)

    # Degree Matrix D
    deg = spatial_adj.sum(axis=1)
    D = np.diag(deg)

    # Non-normalized Laplace L = D - A
    L = D - spatial_adj

    # Feature decomposition
    # L is symmetric and positive semidefinite; take the eigenvectors corresponding to the first k smallest eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(L)  # eigvecs: N x N

    # Sort by eigenvalues in ascending order
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]

    # First, finalize the output, then make changes later.
    pe = np.zeros((N, k), dtype=np.float32)

    k_eff = min(k, N)
    pe = eigvecs[:, :k_eff]  # N x k_eff

    return pe

def decode_atom_features_groverlite(feat) -> dict:
    """
    GroverLite's atom feature is the same as Chemprop
    """
    return decode_atom_features_chemprop(feat)


def decode_bond_features_groverlite(feat) -> dict:
    """
    GroverLite's bond feature is the same as Chemprop
    """
    return decode_bond_features_chemprop(feat)

def inspect_data_groverlite(data: Data, max_edges: int = 50):
    """
    Print the readable information for an entire GroverLite-style diagram (nodes + edges + position-encoded shapes)。
    data: From smiles_to_pyg_graph_groverlite
    """
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    feat_dim = data.x.size(1)

    print("===== GroverLite Graph Inspect =====")
    if hasattr(data, "smiles"):
        print("SMILES:", data.smiles)
    print(f"Node number: {num_nodes}")
    print(f"Edge number: {num_edges}")
    print(f"Atom feature dimension: {feat_dim}")

    if hasattr(data, "y"):
        print("The shape of label y:", tuple(data.y.shape))

    # Location Encoding Information
    if hasattr(data, "lap_eigvec"):
        print("lap_eigvec shape:", tuple(data.lap_eigvec.shape))
    if hasattr(data, "spatial_pos"):
        print("spatial_pos shape:", tuple(data.spatial_pos.shape))

    # ------- Atom -------
    print("\n--- Nodes (Atoms) ---")
    for i in range(num_nodes):
        atom_feat = data.x[i]
        info = decode_atom_features_groverlite(atom_feat)
        print(f"Atom {i}: {info}")

    # ------- Edge -------
    if data.edge_index is None or data.edge_index.numel() == 0:
        print("\n[WARN] There is no edge in the graph (Maybe only one edge).")
        return

    print("\n--- Edges (Bonds) ---")
    num_print = min(num_edges, max_edges)
    for e_idx in range(num_print):
        src = int(data.edge_index[0, e_idx])
        dst = int(data.edge_index[1, e_idx])

        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.size(0) > e_idx:
            bond_feat = data.edge_attr[e_idx]
            binfo = decode_bond_features_groverlite(bond_feat)
        else:
            binfo = {}

        print(f"Edge {e_idx}: {src} -> {dst}, {binfo}")

    if num_edges > num_print:
        print(f"... ({num_edges - num_print} remaining directed edges have not been printed.)")


def smiles_to_pyg_graph_groverlite(smiles: str) -> Optional[Data]:
    """
    GROVER Lite Style SMILES → Graph：
    - Node features: Atom features in Chemprop style
    - Edge features: Bond features in Chemprop style (directed edges)
    - Additional graph position encoding:
        * spatial_pos: Shortest path distance matrix (N x N)
        * lap_eigvec: First k eigenvectors from Laplacian (N x k)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()

    # ---------- Atom features ----------
    atom_features: List[List[float]] = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector_chemprop(atom))
    x = torch.tensor(atom_features, dtype=torch.float)  # [N, F_atom]

    # ---------- Edge features ----------
    edge_indices = []
    edge_features: List[List[float]] = []
    bonds = list(mol.GetBonds())

    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_to_feature_vector_chemprop(bond)

        # i -> j
        edge_indices.append((i, j))
        edge_features.append(bf)
        # j -> i
        edge_indices.append((j, i))
        edge_features.append(bf)

    if len(edge_indices) == 0:
        # in case it has no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        dummy_mol = Chem.MolFromSmiles("CC")
        dummy_bond = list(dummy_mol.GetBonds())[0]
        dummy_dim = len(bond_to_feature_vector_chemprop(dummy_bond))
        edge_attr = torch.empty((0, dummy_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # ---------- Image Position Encoding ----------
    # 1) Shortest Path Distance Matrix spatial_pos
    sp_dist = compute_shortest_path_matrix(num_atoms, bonds)  # N x N

    # 2) Adjacency Matrix used to Laplacian
    adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # 3) Laplace Position Encoding (using the first k eigenvectors)
    lap_pe = compute_laplacian_pos_enc(adj, k=8)  # N x k_eff

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Stored additionally in Data for use by the Graph Transformer
    # data.spatial_pos = torch.tensor(sp_dist, dtype=torch.float)  # [N, N]
    data.lap_eigvec = torch.tensor(lap_pe, dtype=torch.float)    # [N, k_eff]

    return data

# =========================
# General: CSV → List[Data]
# =========================

def load_molecule_dataset(
    csv_path: str,
    smiles_col: str,
    label_cols: Optional[List[str]] = None,
    smiles_to_graph_fn: Callable[[str], Optional[Data]] = smiles_to_pyg_graph_deepchem,
    grover_fp_path: Optional[str] = None,
) -> List[Data]:
    """
    Read data from a CSV file and convert SMILES into PyG Data using the specified smiles_to_graph_fn.

    Parameters:
    - csv_path: e.g., "data/processed/train_dataset.csv"
    - smiles_col: e.g., "nonStereoSMILES"
    - label_cols: List of label columns; defaults to all columns except `smiles_col` and 'descriptors' if `None`
    - smiles_to_graph_fn: SMILES→Data function (can be deepchem / chemprop / groverlite)
    - grover_fp_path: optional path to a .npz file containing GROVER fingerprints
                      with key "fps" (or实际 key), shape [N, D], N = number of rows in csv.

    Return:
    - data_list: List[Data], where each Data contains data.y, data.smiles,
      and optionally data.grover_fp if grover_fp_path is provided.
    """
    data_list: List[Data] = []

    # ------- 1. 如果有 GROVER fingerprint，就先读进来 -------
    if grover_fp_path is not None:
        fp_npz = np.load(grover_fp_path)
        # 根据你真实的 npz 里的 key 来，这里你说是 "fps"
        grover_fp_all = fp_npz["fps"]          # [N, D]
        grover_fp_all = torch.from_numpy(grover_fp_all).float()
    else:
        grover_fp_all = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames

        if all_cols is None:
            raise ValueError(f"CSV {csv_path} has no header, pls check the file.")

        if label_cols is None:
            # 默认：除了 SMILES 列和描述文本列，其余都当标签
            exclude = {smiles_col, "descriptors"}
            label_cols = [c for c in all_cols if c not in exclude]

        row_idx = 0  # 对应 grover_fp_all 的行索引

        for row in reader:
            smiles = row[smiles_col]
            data = smiles_to_graph_fn(smiles)
            if data is None:
                # RDKit 解析失败，fingerprint 这行也要跳过，保持对齐
                row_idx += 1
                continue

            # ------- 2. 读取标签 -------
            labels = []
            for col in label_cols:
                val = row[col]
                if val == "" or val is None:
                    labels.append(float("nan"))
                else:
                    labels.append(float(val))
            y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)  # [1, L]

            data.y = y
            data.smiles = smiles

            # ------- 3. 如果存在 GROVER fingerprint，就挂在 data 上 -------
            if grover_fp_all is not None:
                if row_idx >= grover_fp_all.size(0):
                    raise ValueError(
                        f"GROVER fp rows ({grover_fp_all.size(0)}) < CSV rows, "
                        f"check {grover_fp_path} vs {csv_path}"
                    )
                fp_vec = grover_fp_all[row_idx]          # [D]
                # ★ 关键：加一维，变成 [1, D]，这样 PyG DataLoader 会拼成 [B, D]
                data.grover_fp = fp_vec.unsqueeze(0)     # [1, D]

            data_list.append(data)
            row_idx += 1

    return data_list


