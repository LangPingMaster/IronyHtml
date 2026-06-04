"""
generate_demo_features.py
產生假的 demo features，用來測試 train_svm_fusion.py 與 reproduce_* 主流程是否能正常執行。

注意：這些是假資料，不能代表任何論文結果。
用途只是讓你在尚未下載原始資料集之前，先確認程式環境可以跑。
"""

import argparse
from pathlib import Path
import numpy as np
from common_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="features/demo", help="輸出資料夾")
    parser.add_argument("--n", type=int, default=120, help="樣本數")
    args = parser.parse_args()

    out = ensure_dir(args.out_dir)
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=args.n)

    # 讓 label 稍微影響 feature，避免 SVM 完全隨機。
    text = rng.normal(size=(args.n, 64)) + labels[:, None] * 0.15
    audio = rng.normal(size=(args.n, 32)) + labels[:, None] * 0.20
    visual = rng.normal(size=(args.n, 48)) + labels[:, None] * 0.10

    np.save(out / "text.npy", text)
    np.save(out / "audio.npy", audio)
    np.save(out / "visual.npy", visual)
    np.save(out / "labels.npy", labels)
    print(f"已產生 demo features 到：{out}")


if __name__ == "__main__":
    main()
