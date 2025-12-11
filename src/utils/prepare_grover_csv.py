# src/utils/prepare_grover_csv.py
from pathlib import Path
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    for split in ["train", "valid", "test"]:
        src = DATA_PROCESSED_DIR / f"{split}_dataset.csv"
        if not src.exists():
            print(f"[WARN] {src} 不存在，跳过")
            continue

        df = pd.read_csv(src)

        # 把 nonStereoSMILES 改名成 smiles
        if "smiles" in df.columns:
            smiles_series = df["smiles"]
        else:
            if "nonStereoSMILES" not in df.columns:
                raise ValueError(f"{src} 里既没有 'smiles' 也没有 'nonStereoSMILES'")
            smiles_series = df["nonStereoSMILES"]
            smiles_series = smiles_series.rename("smiles")

        # 只保留一列：smiles
        out_df = smiles_series.to_frame()   # 一列的 DataFrame

        dst = DATA_PROCESSED_DIR / f"{split}_for_grover.csv"
        out_df.to_csv(dst, index=False, encoding="utf-8")
        print(f"[OK] 写出 {dst}，shape = {out_df.shape}")


if __name__ == "__main__":
    main()
