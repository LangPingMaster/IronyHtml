"""
extract_visual_resnet.py
使用 ResNet-152 抽影片 frame 視覺特徵。
對應 Castro et al.（2019）MUStARD 的 visual feature 方法。

流程：
video -> 抽 frame -> ResNet-152 -> pooling vector -> 對所有 frame 平均
"""

from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image


def sample_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """從影片中均勻抽取指定數量的 frame。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"影片沒有可讀 frame：{video_path}")

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def build_resnet152_feature_extractor(device: str):
    """建立 ResNet-152 特徵抽取器，移除最後分類層。"""
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()
    preprocess = weights.transforms()
    return feature_extractor, preprocess


def extract_resnet_video_embeddings(
    video_paths: List[str],
    num_frames: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """批次抽取影片 ResNet 視覺向量。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = build_resnet152_feature_extractor(device)
    all_vectors = []

    for video_path in video_paths:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"找不到影片：{video_path}")
        frames = sample_video_frames(video_path, num_frames=num_frames)
        frame_vectors = []
        for img in frames:
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x).flatten(1).squeeze(0)
            frame_vectors.append(feat.cpu().numpy())
        video_vec = np.mean(np.vstack(frame_vectors), axis=0)
        all_vectors.append(video_vec)

    return np.vstack(all_vectors)
