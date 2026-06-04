"""
extract_text_bert.py
使用 Hugging Face Transformer 模型抽取文字特徵。
可用於：
1. MUStARD：英文 BERT，例如 google-bert/bert-base-uncased
2. MCSD：中文 BERT，例如 google-bert/bert-base-chinese
3. SarcNet：多語模型，例如 FacebookAI/xlm-roberta-base

輸出：每句文字一個固定長度向量，通常取 [CLS] token。
"""

from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def extract_bert_cls_embeddings(
    texts: List[str],
    model_name: str = "google-bert/bert-base-chinese",
    batch_size: int = 8,
    max_length: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """抽取 BERT / XLM-R 的 CLS 文字向量。

    參數：
        texts: 文字清單，每一筆是一句 utterance 或 caption。
        model_name: Hugging Face 模型名稱。
        batch_size: 每批處理幾句文字。GPU 記憶體不足時請調小。
        max_length: 最大 token 長度，超過會截斷。
        device: 'cuda' 或 'cpu'。若 None，會自動判斷。

    回傳：
        numpy array，形狀為 (樣本數, hidden_size)。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # last_hidden_state: (batch, seq_len, hidden_size)
            # CLS token 通常位於第 0 個 token。
            cls_vectors = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_vectors.cpu().numpy())

    return np.vstack(all_embeddings)


if __name__ == "__main__":
    demo_texts = ["你真是太聰明了。", "今天的天氣很好。"]
    features = extract_bert_cls_embeddings(demo_texts)
    print("文字特徵形狀：", features.shape)
