import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(
    input_csv_path,
    output_dir="../../data/processed",
    train_ratio=0.70,
    valid_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    split train, valid, test and save as CSV files

    Args:
        input_csv_path (str): Path to the original CSV file
        output_dir (str): Directory for storing output files
        train_ratio (float): ratio of train set
        valid_ratio (float): ratio of valid set
        test_ratio (float): ratio of test set
        random_state (int): random state seed
    """

    # ensure that the ratio is legal
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio 必须等于 1")

    # create output table
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read the data
    df = pd.read_csv(input_csv_path)

    # split train & temp(valid+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_state,
        shuffle=True
    )

    # split valid / test
    valid_size = valid_ratio / (valid_ratio + test_ratio)

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - valid_size),
        random_state=random_state,
        shuffle=True
    )

    train_df.to_csv(Path(output_dir) / "train_dataset.csv", index=False)
    valid_df.to_csv(Path(output_dir) / "valid_dataset.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test_dataset.csv", index=False)

    print("The data has been split！")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test:  {len(test_df)}")


if __name__ == "__main__":
    split_dataset(
        input_csv_path="../../data/raw/curated_GS_LF_merged_4983.csv",
        output_dir="../../data/processed",
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
