"""
extract_audio_wav2vec2.py
使用 Chinese Wav2Vec2 抽語音 embedding。
對應 Gao et al.（2025）MCSD 的 audio feature 方法。

模型建議：
jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
音訊要求：mono, 16kHz。
"""

from typing import List
import numpy as np
import torch
import librosa
from transformers import AutoProcessor, AutoModel


def extract_wav2vec2_embeddings(
    audio_paths: List[str],
    model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    device: str | None = None,
) -> np.ndarray:
    """抽取 Wav2Vec2 語音向量。

    參數：
        audio_paths: WAV 檔案路徑清單。
        model_name: Hugging Face Wav2Vec2 模型名稱。
        device: 'cuda' 或 'cpu'。

    回傳：
        shape=(樣本數, hidden_size) 的 numpy array。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_vectors = []
    for audio_path in audio_paths:
        # Gao et al. 使用 16kHz。這裡強制 resample 到 16000。
        speech, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state: (batch, frames, hidden_size)
            # 對 frames 平均，得到固定長度 utterance representation。
            vec = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        all_vectors.append(vec.cpu().numpy())

    return np.vstack(all_vectors)
