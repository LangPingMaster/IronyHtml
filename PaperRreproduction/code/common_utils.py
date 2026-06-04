"""
common_utils.py
共用工具函式。
本檔案提供三篇論文重現時會重複用到的功能：
1. 建立資料夾
2. 讀取 JSON / CSV
3. 儲存與讀取 numpy feature
4. 計算分類評估指標

注意：本程式不會自動下載有授權限制的影片或圖片資料。
請依各資料集官方網站下載後放入 data/ 目錄。
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def ensure_dir(path: str) -> Path:
    """建立資料夾並回傳 Path 物件。

    參數：
        path: 欲建立的資料夾路徑。
    回傳：
        Path 物件，方便後續組合檔名。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str) -> dict:
    """讀取 JSON 標註檔。

    MUStARD 的 sarcasm_data.json 可使用此函式讀取。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: str) -> pd.DataFrame:
    """讀取 CSV metadata。

    MCSD 若整理成 metadata.csv，可使用此函式讀取。
    """
    return pd.read_csv(path)


def save_feature(path: str, array: np.ndarray) -> None:
    """將 feature 儲存成 .npy 檔案。"""
    ensure_dir(str(Path(path).parent))
    np.save(path, array)


def load_feature(path: str) -> np.ndarray:
    """讀取 .npy feature 檔案。"""
    return np.load(path)


def evaluate_binary(y_true, y_pred, average="macro") -> dict:
    """計算二分類常用指標。

    參數：
        y_true: 真實標籤，0=非反諷，1=反諷。
        y_pred: 模型預測標籤。
        average: 'macro' 或 'weighted'。
    回傳：
        包含 accuracy、precision、recall、f1、confusion_matrix 的 dict。
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm,
    }
