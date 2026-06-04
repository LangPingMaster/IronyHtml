"""
reproduce_mcsd.py
Gao et al.（2025）MCSD 重現入口。

資料來源：
    https://research.rug.nl/en/datasets/mcsd-10-multimodal-chinese-sarcasm-dataset/
    DOI: 10.34894/A0NLTQ
    GitHub Wiki: https://github.com/x-y-g/MCSD/wiki

使用方式：
1. 下載 MCSD metadata。
2. 依 start/end timestamps 切出影片與音訊。
3. 使用 extract_text_bert.py / extract_audio_wav2vec2.py / TimeSformer 抽 features。
4. 將 text.npy / audio.npy / visual.npy / labels.npy 放到 features/mcsd/。
5. 執行本程式。
"""

import argparse
from pathlib import Path
import numpy as np
from common_utils import ensure_dir
from train_svm_fusion import run_all_modalities


def load_features(feature_dir: str) -> tuple[dict, np.ndarray]:
    """讀取 MCSD 預抽特徵。"""
    feature_path = Path(feature_dir)
    feature_dict = {}
    for name in ["text", "audio", "visual"]:
        fp = feature_path / f"{name}.npy"
        if fp.exists():
            feature_dict[name] = np.load(fp)
            print(f"已讀取 {name} feature：{fp}")
        else:
            print(f"找不到 {name} feature：{fp}")

    y_path = feature_path / "labels.npy"
    if not y_path.exists():
        raise FileNotFoundError("找不到 labels.npy，請先由 MCSD metadata 產生。")
    return feature_dict, np.load(y_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/mcsd/metadata.csv", help="MCSD metadata CSV")
    parser.add_argument("--audio_dir", default="data/mcsd/audio", help="音訊資料夾")
    parser.add_argument("--video_dir", default="data/mcsd/videos", help="影片資料夾")
    parser.add_argument("--feature_dir", default="features/mcsd", help="features 資料夾")
    parser.add_argument("--output", default="results/mcsd_results.csv", help="結果 CSV")
    args = parser.parse_args()

    feature_dict, y = load_features(args.feature_dir)
    df = run_all_modalities(feature_dict, y, average="macro")
    ensure_dir(str(Path(args.output).parent))
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(df)
    print(f"結果已輸出：{args.output}")


if __name__ == "__main__":
    main()
