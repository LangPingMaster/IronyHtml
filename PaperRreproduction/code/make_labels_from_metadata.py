"""
make_labels_from_metadata.py
將 metadata 的 sarcasm label 欄位轉成 labels.npy。

適用情境：
1. 你有 MCSD metadata.csv，裡面有 label 欄位。
2. 你有自己整理的 CSV，欄位為 id, text, label。

label 欄位規則：
    sarcastic / true / 1 -> 1
    non-sarcastic / false / 0 -> 0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common_utils import ensure_dir


def normalize_label(x) -> int:
    """把不同寫法的標籤轉成 0/1。"""
    s = str(x).strip().lower()
    if s in ["1", "true", "sarcastic", "sarcasm", "yes"]:
        return 1
    if s in ["0", "false", "not sarcastic", "non-sarcastic", "no"]:
        return 0
    raise ValueError(f"無法辨識 label：{x}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="metadata CSV 路徑")
    parser.add_argument("--label_col", default="label", help="標籤欄位名稱")
    parser.add_argument("--output", required=True, help="輸出的 labels.npy 路徑")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    labels = df[args.label_col].apply(normalize_label).to_numpy(dtype=np.int64)
    ensure_dir(str(Path(args.output).parent))
    np.save(args.output, labels)
    print(f"已輸出 labels：{args.output}, shape={labels.shape}")


if __name__ == "__main__":
    main()
