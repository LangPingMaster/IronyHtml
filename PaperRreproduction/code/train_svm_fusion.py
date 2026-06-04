"""
train_svm_fusion.py
多模態特徵串接 + SVM 分類。
對應 Castro et al.（2019）與 Gao et al.（2025）的 baseline 思路。

輸入：
    features/text.npy
    features/audio.npy
    features/visual.npy
    labels.npy
輸出：
    Accuracy / Precision / Recall / F1
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common_utils import evaluate_binary


def concatenate_modalities(feature_dict: Dict[str, np.ndarray], modalities: List[str]) -> np.ndarray:
    """依指定模態名稱串接特徵。

    參數：
        feature_dict: 例如 {'text': text_array, 'audio': audio_array}
        modalities: 例如 ['text', 'audio', 'visual']

    回傳：
        串接後的 X 特徵矩陣。
    """
    selected = []
    for name in modalities:
        if name not in feature_dict:
            raise KeyError(f"feature_dict 中找不到模態：{name}")
        selected.append(feature_dict[name])
    return np.concatenate(selected, axis=1)


def train_svm_cv(X: np.ndarray, y: np.ndarray, average: str = "macro") -> dict:
    """使用 Stratified 5-fold CV + SVM 進行分類。

    參數：
        X: 特徵矩陣。
        y: 標籤向量，0=非反諷，1=反諷。
        average: 評估平均方式；MCSD 建議 macro，MUStARD 可用 weighted。

    回傳：
        評估結果 dict。
    """
    # 為了讓範例程式可以快速執行，這裡使用固定參數的 SVM。
    # 若要完全貼近 Gao et al.（2025）的 GridSearchCV，
    # 可以把 SVC(C=10, gamma=0.0001, kernel="rbf") 改成 GridSearchCV。
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=10, gamma=0.0001, kernel="rbf"))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    return evaluate_binary(y, y_pred, average=average)


def run_all_modalities(feature_dict: Dict[str, np.ndarray], y: np.ndarray, average="macro") -> pd.DataFrame:
    """自動跑常見模態組合。"""
    experiments = [
        ["text"], ["audio"], ["visual"],
        ["text", "audio"], ["text", "visual"], ["audio", "visual"],
        ["text", "audio", "visual"],
    ]
    rows = []
    for modalities in experiments:
        valid = all(m in feature_dict for m in modalities)
        if not valid:
            continue
        X = concatenate_modalities(feature_dict, modalities)
        result = train_svm_cv(X, y, average=average)
        result["modalities"] = "+".join(modalities)
        rows.append(result)
    return pd.DataFrame(rows)
