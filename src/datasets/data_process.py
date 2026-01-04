import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def smiles_to_scaffold(smiles: str) -> str:
    """Murcko scaffold 作为 chemotype。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return "NOSCAFFOLD"
    return Chem.MolToSmiles(scaf, isomericSmiles=False)


def split_dataset_chemotype_stratified(
    input_csv_path,
    smiles_col="nonStereoSMILES",
    output_dir="../../data/processed",
    train_ratio=0.80,
    valid_ratio=0.10,
    test_ratio=0.10,
    random_state=42,
    rare_thresh=3,   # chemotype 出现次数 < rare_thresh 的全部合并为 RARE
):
    # 1) ratio check
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"找不到 smiles_col={smiles_col}，你现在列名有：{list(df.columns)[:20]} ...")

    # 2) build chemotype
    df = df.copy()
    df["chemotype"] = df[smiles_col].astype(str).apply(smiles_to_scaffold)

    # 3) merge rare chemotypes to make stratification feasible
    counts = df["chemotype"].value_counts()
    df.loc[df["chemotype"].map(counts) < rare_thresh, "chemotype"] = "RARE"

    y = df["chemotype"].values
    idx = df.index.values

    # 4) stratified split: train vs temp
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - train_ratio),
        random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(idx, y))
    train_df = df.iloc[train_idx].drop(columns=["chemotype"])
    temp_df = df.iloc[temp_idx].copy()

    # ✅ 关键修复：temp 里有些类会变成 1 个样本，必须再合并一次
    temp_counts = temp_df["chemotype"].value_counts()
    temp_df.loc[temp_df["chemotype"].map(temp_counts) < 2, "chemotype"] = "RARE"

    # 5) stratified split: valid vs test inside temp
    valid_ratio_in_temp = valid_ratio / (valid_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - valid_ratio_in_temp),
        random_state=random_state
    )
    temp_y = temp_df["chemotype"].values
    temp_i = temp_df.index.values
    valid_sub_idx, test_sub_idx = next(sss2.split(temp_i, temp_y))

    valid_df = temp_df.iloc[valid_sub_idx].drop(columns=["chemotype"])
    test_df = temp_df.iloc[test_sub_idx].drop(columns=["chemotype"])

    # 6) save
    train_df.to_csv(Path(output_dir) / "train_dataset.csv", index=False)
    valid_df.to_csv(Path(output_dir) / "valid_dataset.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test_dataset.csv", index=False)

    print("Chemotype-stratified split done!")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test:  {len(test_df)}")

    # 7) quick sanity: check overlap (防数据泄漏)
    train_set = set(train_df[smiles_col].astype(str))
    valid_set = set(valid_df[smiles_col].astype(str))
    test_set  = set(test_df[smiles_col].astype(str))
    print("Overlap train∩valid:", len(train_set & valid_set))
    print("Overlap train∩test :", len(train_set & test_set))
    print("Overlap valid∩test :", len(valid_set & test_set))


if __name__ == "__main__":
    split_dataset_chemotype_stratified(
        input_csv_path="../../data/raw/curated_GS_LF_merged_4983.csv",
        smiles_col="nonStereoSMILES",
        output_dir="../../data/processed",
        train_ratio=0.80,
        valid_ratio=0.10,
        test_ratio=0.10,
        random_state=42,
        rare_thresh=3,
    )
