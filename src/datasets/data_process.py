import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def smiles_to_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return "NOSCAFFOLD"
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

def detect_label_cols(df: pd.DataFrame, smiles_col: str, extra_nonlabel_cols=None):
    """
    默认：除了 smiles_col 以外的“数值型列”都当作标签列（适合 0/1 多标签）。
    如果你还有 id/name 等非标签列，请在 extra_nonlabel_cols 里排除。
    """
    extra_nonlabel_cols = set(extra_nonlabel_cols or [])
    cand = [c for c in df.columns if c != smiles_col and c not in extra_nonlabel_cols]
    label_cols = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
    if not label_cols:
        raise ValueError("未检测到数值型标签列，请手动指定或排除非标签列。")
    return label_cols


def split_dataset_chemotype_stratified(
    input_csv_path,
    smiles_col="nonStereoSMILES",
    output_dir="../../data/processed",
    train_ratio=0.80,
    valid_ratio=0.10,
    test_ratio=0.10,
    random_state=42,
    rare_thresh=3,   # chemotype 出现次数 < rare_thresh 的全部合并为 RARE
    n_bins=10
):
    # 1) ratio check
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal to 1")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"Can't find smiles_col={smiles_col}，now we have:{list(df.columns)[:20]} ...")

    # 2) build chemotype
    df = df.copy()

    # df["chemotype"] = df[smiles_col].astype(str).apply(smiles_to_scaffold)
    #
    # # 3) merge rare chemotypes to make stratification feasible
    # counts = df["chemotype"].value_counts()
    # df.loc[df["chemotype"].map(counts) < rare_thresh, "chemotype"] = "RARE"

    candidate_cols = [c for c in df.columns if c != smiles_col]
    label_cols = []
    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            label_cols.append(c)

    if len(label_cols) == 0:
        raise ValueError("No numeric label columns detected. Please specify label columns manually.")
    n_pos = (df[label_cols].fillna(0).values > 0).sum(axis=1)
    # 分桶：保证每桶都有样本
    df["chemotype"] = pd.qcut(n_pos, q=min(n_bins, len(df)), duplicates="drop").astype(str)

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

    # 7) quick sanity: check overlap
    train_set = set(train_df[smiles_col].astype(str))
    valid_set = set(valid_df[smiles_col].astype(str))
    test_set  = set(test_df[smiles_col].astype(str))
    print("Overlap train∩valid:", len(train_set & valid_set))
    print("Overlap train∩test :", len(train_set & test_set))
    print("Overlap valid∩test :", len(valid_set & test_set))


def iterative_stratified_split_once(
    input_csv_path,
    smiles_col="nonStereoSMILES",
    output_dir="../../data/processed",
    train_ratio=0.80,
    valid_ratio=0.10,
    test_ratio=0.10,
    random_state=42,
    extra_nonlabel_cols=None,  # 例如 ["id", "name"]，没有就 None
):
    # 1) ratio check
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal to 1")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Can't find smiles_col={smiles_col}，now we have:{list(df.columns)[:20]} ...")

    df = df.copy()

    # 2) label cols + multilabel matrix Y
    label_cols = detect_label_cols(df, smiles_col, extra_nonlabel_cols)
    Y = (df[label_cols].fillna(0).values > 0).astype(int)

    idx = np.arange(len(df))

    # 3) first split: train vs temp (iterative stratified)
    sss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - train_ratio), random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(idx, Y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_Y = Y[temp_idx]

    # 4) second split inside temp: valid vs test (iterative stratified)
    valid_ratio_in_temp = valid_ratio / (valid_ratio + test_ratio)
    sss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - valid_ratio_in_temp), random_state=random_state
    )
    temp_i = np.arange(len(temp_df))
    valid_sub_idx, test_sub_idx = next(sss2.split(temp_i, temp_Y))

    valid_df = temp_df.iloc[valid_sub_idx].reset_index(drop=True)
    test_df  = temp_df.iloc[test_sub_idx].reset_index(drop=True)

    # 5) save
    train_df.to_csv(Path(output_dir) / "train_dataset.csv", index=False)
    valid_df.to_csv(Path(output_dir) / "valid_dataset.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test_dataset.csv", index=False)

    print("Iterative stratified split done!")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test:  {len(test_df)}")

    # 6) quick sanity: overlap by SMILES string
    train_set = set(train_df[smiles_col].astype(str))
    valid_set = set(valid_df[smiles_col].astype(str))
    test_set  = set(test_df[smiles_col].astype(str))
    print("Overlap train∩valid:", len(train_set & valid_set))
    print("Overlap train∩test :", len(train_set & test_set))
    print("Overlap valid∩test :", len(valid_set & test_set))

    # 7) extra sanity: how many labels have 0 positives in valid/test
    valid_zero = int((valid_df[label_cols].fillna(0).sum(axis=0) == 0).sum())
    test_zero  = int((test_df[label_cols].fillna(0).sum(axis=0) == 0).sum())
    print(f"Valid labels with 0 positive: {valid_zero}/{len(label_cols)}")
    print(f"Test  labels with 0 positive: {test_zero}/{len(label_cols)}")


if __name__ == "__main__":
    iterative_stratified_split_once(
        input_csv_path="../../data/raw/curated_GS_LF_merged_4983.csv",
        smiles_col="nonStereoSMILES",
        output_dir="../../data/processed",
        train_ratio=0.80,
        valid_ratio=0.10,
        test_ratio=0.10,
        random_state=42,
        extra_nonlabel_cols=None,

    #     split_dataset_chemotype_stratified(
    #         input_csv_path="../../data/raw/curated_GS_LF_merged_4983.csv",
    #         smiles_col="nonStereoSMILES",
    #         output_dir="../../data/processed",
    #         train_ratio=0.80,
    #         valid_ratio=0.10,
    #         test_ratio=0.10,
    #         random_state=42,
    #         rare_thresh=3,
    )

