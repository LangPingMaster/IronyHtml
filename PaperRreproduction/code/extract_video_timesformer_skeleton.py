"""
extract_video_timesformer_skeleton.py
TimeSformer 影片特徵抽取骨架。

Gao et al.（2025）使用 TimeSformer 抽 MCSD 影片特徵。
官方 TimeSformer repository：
https://github.com/facebookresearch/TimeSformer

注意：TimeSformer 官方環境與權重可能需要額外安裝 fvcore、simplejson、pytorchvideo 等依賴。
本檔案提供研究流程骨架，不保證在未安裝官方依賴前可直接執行。
"""

from typing import List
import numpy as np


def extract_timesformer_embeddings(video_paths: List[str]) -> np.ndarray:
    """TimeSformer 特徵抽取骨架。

    你需要依官方 TimeSformer README 完成：
    1. clone facebookresearch/TimeSformer
    2. 安裝依賴
    3. 下載預訓練權重
    4. 把每個影片 resize 到 224x224，抽 8 frames
    5. 通過 TimeSformer encoder
    6. 對 frame embeddings 取平均

    回傳：
        shape=(樣本數, 768) 的影片向量。
    """
    raise NotImplementedError(
        "請先安裝 TimeSformer 官方 repository，然後依照本檔案註解接入模型。"
    )
