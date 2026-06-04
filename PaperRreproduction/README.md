# 三篇多模態反諷辨識論文重現教學網站

本 ZIP 針對以下三篇論文：

1. Castro et al. (2019) MUStARD
2. Gao et al. (2025) MCSD 1.0
3. Yue et al. (2024) SarcNet

提供：

- 多網頁 HTML 教學
- 下拉式說明
- 流程圖與 CSS 動畫
- 資料來源與網站連結
- 開源模型連結
- Python 範例程式碼
- 中文註解與變數函式說明

## 開始使用

請先打開 `index.html`。

## 安裝環境

```bash
conda create -n sarcasm-mm python=3.10
conda activate sarcasm-mm
pip install -r requirements.txt
```

Ubuntu 需安裝 FFmpeg：

```bash
sudo apt update
sudo apt install ffmpeg
```

## 資料集

本 ZIP 不包含原始資料集本體。請依 HTML 的「資料來源與網站」頁面自行下載或申請資料。

## 重要提醒

程式碼是研究重現框架。若你要嚴格重現原論文數值，請使用原論文指定的資料 split、特徵版本與模型參數。

## 測試假資料流程

```bash
python code/generate_demo_features.py --out_dir features/mustard
python code/reproduce_mustard.py --feature_dir features/mustard --output results/demo_mustard.csv
```


## 新增內容
- `pages/paper_summaries.html`：整理原始論文連結、PDF、資料集連結、研究重點、方法、研究成果、圖表與動畫下拉式說明。
- `pages/data_links.html`：補強三篇核心論文的可點擊原文與資料連結。
