"""
reproduce_mustard.py
Castro et al.（2019）MUStARD 重現入口。

資料來源：
    https://github.com/soujanyaporia/MUStARD

建議做法：
1. 先 clone 官方 repository。
2. 依 README 下載 pre-extracted BERT / visual features。
3. 或使用本 ZIP 的 extract_* 程式重新抽 features。
4. 將整理好的 text.npy / audio.npy / visual.npy / labels.npy 放到 features/mustard/。
5. 執行本程式跑 SVM。
"""

import argparse
from pathlib import Path
import numpy as np
from common_utils import ensure_dir
from train_svm_fusion import run_all_modalities


def load_precomputed_features(feature_dir: str) -> tuple[dict, np.ndarray]:
    """讀取預先抽好的 MUStARD features。

    必要檔案：
        labels.npy
    可選檔案：
        text.npy, audio.npy, visual.npy
    """
    feature_path = Path(feature_dir)
    feature_dict = {}
    for name in ["text", "audio", "visual"]:
        fp = feature_path / f"{name}.npy"
        if fp.exists():
            feature_dict[name] = np.load(fp)
            print(f"已讀取 {name} feature：{fp}")
        else:
            print(f"找不到 {name} feature：{fp}，略過此模態")

    labels_path = feature_path / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            "找不到 labels.npy。請先依 sarcasm_data.json 整理 labels，或使用官方資料產生。"
        )
    y = np.load(labels_path)
    return feature_dict, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/mustard/sarcasm_data.json", help="MUStARD 標註檔路徑，保留給檢查用")
    parser.add_argument("--feature_dir", default="features/mustard", help="存放 .npy features 的資料夾")
    parser.add_argument("--output", default="results/mustard_results.csv", help="結果輸出 CSV")
    args = parser.parse_args()

    feature_dict, y = load_precomputed_features(args.feature_dir)
    df = run_all_modalities(feature_dict, y, average="weighted")

    ensure_dir(str(Path(args.output).parent))
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(df)
    print(f"結果已輸出：{args.output}")


if __name__ == "__main__":
    main()
