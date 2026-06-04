"""
extract_audio_librosa.py
使用 Librosa 抽取傳統聲學特徵。
對應 Castro et al.（2019）MUStARD 的 audio baseline 概念。

抽取特徵：
1. MFCC
2. MFCC delta
3. Mel-spectrogram
4. Spectral centroid

輸出：每個音檔一個固定長度向量。
"""

from pathlib import Path
from typing import List
import numpy as np
import librosa


def summarize_feature(matrix: np.ndarray) -> np.ndarray:
    """把 frame-level feature 轉成固定長度向量。

    做法：對時間軸取 mean 和 standard deviation，再串接。
    """
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1)
    return np.concatenate([mean, std], axis=0)


def extract_librosa_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """抽取單一音檔的 Librosa 特徵。

    參數：
        audio_path: 音訊檔路徑。
        sr: sampling rate。MUStARD 原文使用 22050 Hz。

    回傳：
        固定長度 numpy 向量。
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    feature_vector = np.concatenate([
        summarize_feature(mfcc),
        summarize_feature(mfcc_delta),
        summarize_feature(mel_db),
        summarize_feature(centroid),
    ])
    return feature_vector


def extract_many_audio_features(audio_paths: List[str]) -> np.ndarray:
    """批次抽取多個音檔特徵。"""
    features = []
    for path in audio_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"找不到音訊檔：{path}")
        features.append(extract_librosa_features(path))
    return np.vstack(features)


if __name__ == "__main__":
    print("請在其他主程式中呼叫 extract_many_audio_features(audio_paths)。")
