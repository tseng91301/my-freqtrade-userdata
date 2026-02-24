"""
sequence_learning.py — LSTM + Static Feature 混合模型
======================================================
架構：
  ┌──────────────────────┐    ┌───────────────────────┐
  │  前 SEQ_LEN 根 K 線   │    │  訊號點靜態特徵（20維） │
  │  OHLCV + 衍生（11維） │    │  FVG/OB/BOS/蠟燭/脈絡  │
  └──────────┬───────────┘    └───────────┬───────────┘
             │  LSTM / GRU                │  FC Layer
             │  (學走勢模式)               │  (學結構品質)
             └──────────────┬─────────────┘
                        Concat + Fusion FC
                            │
                      有效(1) / 無效(0)

流程：
  1. 從 data_builder 取得/建構資料集
  2. Walk-Forward 時序交叉驗證
  3. 每折：訓練 + Early Stopping + 評估
  4. 訓練全資料最終模型，儲存 .pt
  5. 輸出評估圖表
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)

# ── 本地模組 ──
from data_builder import build_dataset, load_dataset, DATASET_FILE, SCALER_FILE

# ═══════════════════════════════════════════════════════
# ① 參數
# ═══════════════════════════════════════════════════════
SEQ_LEN         = 50       # 與 data_builder 一致
HIDDEN_SIZE     = 128      # LSTM 隱藏層大小
NUM_LAYERS      = 2        # LSTM 層數
DROPOUT         = 0.3      # Dropout 比率
STATIC_HIDDEN   = 64       # 靜態特徵分支隱藏層大小
FUSION_HIDDEN   = 64       # 融合層大小

BATCH_SIZE      = 64
MAX_EPOCHS      = 100
LR              = 1e-3
PATIENCE        = 10       # Early Stopping patience
N_CV_SPLITS     = 5
CV_GAP          = 5        # Walk-Forward gap（防 look-ahead）

OUTPUT_DIR      = "./feature_learning_output/"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}smc_lstm_model.pt"
REPORT_PATH     = f"{OUTPUT_DIR}smc_lstm_report.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  SMC 訊號有效性 — LSTM 混合模型")
print("=" * 60)
print(f"  Device: {DEVICE}")

# ═══════════════════════════════════════════════════════
# ② 資料集載入
# ═══════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(DATASET_FILE) and os.path.exists(SCALER_FILE):
    print("\n📂 偵測到已建構資料集，直接載入...")
    X_seq, X_static, y, seq_scaler, static_scaler, static_feat_cols, signal_indices = load_dataset()
else:
    print("\n🔧 資料集不存在，開始建構...")
    X_seq, X_static, y, seq_scaler, static_scaler, static_feat_cols, signal_indices = build_dataset()

N          = len(y)
F_seq      = X_seq.shape[2]
F_static   = X_static.shape[1]
print(f"\n   樣本數：{N}  序列特徵：{F_seq}  靜態特徵：{F_static}")
print(f"   有效訊號：{y.sum()} ({y.mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════
# ③ PyTorch Dataset
# ═══════════════════════════════════════════════════════
class SMCDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq    = torch.tensor(X_seq,    dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y        = torch.tensor(y,         dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


# ═══════════════════════════════════════════════════════
# ④ 模型定義
# ═══════════════════════════════════════════════════════
class SMCHybridModel(nn.Module):
    """
    雙輸入混合模型：
      - 序列分支：GRU（比 LSTM 稍快，效果相近）
      - 靜態分支：FC
      - 融合層：Concat → FC → 分類
    """
    def __init__(self, f_seq, f_static,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 static_hidden=STATIC_HIDDEN, fusion_hidden=FUSION_HIDDEN,
                 dropout=DROPOUT):
        super().__init__()

        # 序列分支：GRU
        self.gru = nn.GRU(
            input_size=f_seq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.seq_norm = nn.LayerNorm(hidden_size)
        self.seq_drop = nn.Dropout(dropout)

        # 靜態分支：FC
        self.static_branch = nn.Sequential(
            nn.Linear(f_static, static_hidden),
            nn.BatchNorm1d(static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(),
        )

        # 融合層
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + static_hidden, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2),
        )

    def forward(self, x_seq, x_static):
        # 序列：取最後一個時間步的隱藏狀態
        out, _ = self.gru(x_seq)            # (B, S, H)
        seq_feat = self.seq_drop(self.seq_norm(out[:, -1, :]))  # (B, H)

        # 靜態特徵
        static_feat = self.static_branch(x_static)  # (B, static_hidden)

        # 融合 → 分類
        combined = torch.cat([seq_feat, static_feat], dim=1)
        return self.fusion(combined)         # (B, 2)


def build_model():
    return SMCHybridModel(F_seq, F_static).to(DEVICE)


# ═══════════════════════════════════════════════════════
# ⑤ 訓練單折
# ═══════════════════════════════════════════════════════
def train_one_fold(
    X_seq_tr, X_sta_tr, y_tr,
    X_seq_val, X_sta_val, y_val,
    verbose=True,
):
    """訓練一折，含 Early Stopping，回傳最佳模型與訓練記錄。"""
    # 類別不平衡：計算 pos_weight
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    train_ds = SMCDataset(X_seq_tr, X_sta_tr, y_tr)
    val_ds   = SMCDataset(X_seq_val, X_sta_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = build_model()
    # BCEWithLogitsLoss 等價的 CrossEntropyLoss + pos_weight 調整
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight.item()]).to(DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for xb_seq, xb_sta, yb in train_dl:
            xb_seq, xb_sta, yb = xb_seq.to(DEVICE), xb_sta.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb_seq, xb_sta)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_seq, xb_sta, yb in val_dl:
                xb_seq, xb_sta, yb = xb_seq.to(DEVICE), xb_sta.to(DEVICE), yb.to(DEVICE)
                logits = model(xb_seq, xb_sta)
                val_loss += criterion(logits, yb).item() * len(yb)
        val_loss /= len(val_ds)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"     Epoch {epoch:>3d} | train={tr_loss:.4f}  val={val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    print(f"     Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history


# ═══════════════════════════════════════════════════════
# ⑥ 推論
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, X_seq_arr, X_sta_arr):
    """回傳 (preds, probas)。"""
    model.eval()
    ds = SMCDataset(X_seq_arr, X_sta_arr, np.zeros(len(X_seq_arr), dtype=np.int64))
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    preds_list, probas_list = [], []
    for xb_seq, xb_sta, _ in dl:
        logits = model(xb_seq.to(DEVICE), xb_sta.to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        preds_list.append(preds)
        probas_list.append(probs)
    return np.concatenate(preds_list), np.concatenate(probas_list)


# ═══════════════════════════════════════════════════════
# ⑦ Walk-Forward 交叉驗證
# ═══════════════════════════════════════════════════════
print(f"\n🔁 Walk-Forward CV（{N_CV_SPLITS} 折，gap={CV_GAP}）...")

n        = N
fold_sz  = n // (N_CV_SPLITS + 1)
cv_results   = []
oof_preds    = np.full(n, np.nan)
oof_probas   = np.full(n, np.nan)
fold_histories = []

for fold in range(N_CV_SPLITS):
    train_end  = fold_sz * (fold + 1)
    test_start = train_end + CV_GAP
    test_end   = test_start + fold_sz if fold < N_CV_SPLITS - 1 else n

    if test_start >= test_end or train_end < 50:
        continue

    print(f"\n  ── Fold {fold+1} | train=[0,{train_end})  test=[{test_start},{test_end}) ──")

    model, hist = train_one_fold(
        X_seq[:train_end],    X_static[:train_end],    y[:train_end],
        X_seq[test_start:test_end], X_static[test_start:test_end], y[test_start:test_end],
        verbose=True,
    )
    fold_histories.append(hist)

    y_pred, y_proba = predict(model, X_seq[test_start:test_end], X_static[test_start:test_end])
    y_test = y[test_start:test_end]

    oof_preds[test_start:test_end]  = y_pred
    oof_probas[test_start:test_end] = y_proba

    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    result = {
        "fold":      fold + 1,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc":       auc,
    }
    print(f"   Prec={result['precision']:.3f}  Recall={result['recall']:.3f}"
          f"  F1={result['f1']:.3f}  AUC={result['auc']:.3f}")
    cv_results.append(result)

cv_df = pd.DataFrame(cv_results)
print("\n── Walk-Forward 平均指標 ──")
print(cv_df[["precision", "recall", "f1", "auc"]].mean().to_string())

# ═══════════════════════════════════════════════════════
# ⑧ 全資料訓練最終模型
# ═══════════════════════════════════════════════════════
print("\n🏋️  訓練最終模型（全資料）...")

# 使用最後一折模型作為驗證集（看收斂性）
val_start = int(n * 0.9)
final_model, final_hist = train_one_fold(
    X_seq[:val_start],   X_static[:val_start],   y[:val_start],
    X_seq[val_start:],   X_static[val_start:],   y[val_start:],
    verbose=True,
)

torch.save({
    "model_state":   final_model.state_dict(),
    "f_seq":         F_seq,
    "f_static":      F_static,
    "static_feat_cols": static_feat_cols,
    "seq_len":       SEQ_LEN,
    "config": {
        "hidden_size":    HIDDEN_SIZE,
        "num_layers":     NUM_LAYERS,
        "dropout":        DROPOUT,
        "static_hidden":  STATIC_HIDDEN,
        "fusion_hidden":  FUSION_HIDDEN,
    }
}, MODEL_SAVE_PATH)
print(f"💾 模型已儲存：{MODEL_SAVE_PATH}")

# ═══════════════════════════════════════════════════════
# ⑨ 視覺化
# ═══════════════════════════════════════════════════════
print("\n📈 產生評估圖表...")

DARK_BG  = "#0e1117"
CARD_BG  = "#1a1d2e"
ACCENT   = "#00d4aa"
ACCENT2  = "#ff6b6b"
ACCENT3  = "#f5a623"
TEXT_COL = "#e0e0e0"
GRID_COL = "#2a2d3e"

def style_ax(ax, title=""):
    ax.set_facecolor(CARD_BG)
    ax.spines[["top", "right", "left", "bottom"]].set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)

fig = plt.figure(figsize=(18, 16), facecolor=DARK_BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                         top=0.92, bottom=0.07)

# ── (A) Loss 曲線（最終模型）──
ax_loss = fig.add_subplot(gs[0, :])
style_ax(ax_loss, "Final Model Loss Curve (Train vs Val)")
ax_loss.plot(final_hist["train_loss"], color=ACCENT,  lw=1.8, label="Train Loss")
ax_loss.plot(final_hist["val_loss"],   color=ACCENT2, lw=1.8, label="Val Loss")
ax_loss.set_xlabel("Epoch", color=TEXT_COL)
ax_loss.set_ylabel("CrossEntropy Loss", color=TEXT_COL)
ax_loss.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, fontsize=10)

# ── (B) OOF 混淆矩陣 ──
ax_cm = fig.add_subplot(gs[1, 0])
style_ax(ax_cm, "OOF Confusion Matrix (Out-of-Fold)")
valid_mask = ~np.isnan(oof_preds)
if valid_mask.sum() > 0:
    cm   = confusion_matrix(y[valid_mask], oof_preds[valid_mask].astype(int))
    cmap = LinearSegmentedColormap.from_list("teal", [DARK_BG, ACCENT])
    ax_cm.imshow(cm, cmap=cmap, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha="center", va="center",
                       color="white", fontsize=14, fontweight="bold")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred: Invalid", "Pred: Valid"],   color=TEXT_COL, fontsize=9)
    ax_cm.set_yticklabels(["True: Invalid", "True: Valid"],   color=TEXT_COL, fontsize=9)

# ── (C) Walk-Forward 各折指標 ──
ax_cv = fig.add_subplot(gs[1, 1])
style_ax(ax_cv, "Walk-Forward CV Performance per Fold")
if len(cv_df) > 0:
    x = cv_df["fold"].values
    w = 0.22
    ax_cv.bar(x - w, cv_df["precision"], w, label="Precision", color=ACCENT,  alpha=0.85)
    ax_cv.bar(x,     cv_df["recall"],    w, label="Recall",    color=ACCENT2, alpha=0.85)
    ax_cv.bar(x + w, cv_df["f1"],        w, label="F1",        color=ACCENT3, alpha=0.85)
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels([f"Fold {i}" for i in x], color=TEXT_COL)
    ax_cv.set_ylim(0, 1.1)
    ax_cv.set_ylabel("Score", color=TEXT_COL)
    ax_cv.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, fontsize=9)

# ── (D) OOF 機率分布 ──
ax_dist = fig.add_subplot(gs[2, 0])
style_ax(ax_dist, "OOF Predicted Probability Distribution")
if valid_mask.sum() > 0:
    bins = np.linspace(0, 1, 25)
    probs_pos = oof_probas[valid_mask][y[valid_mask] == 1]
    probs_neg = oof_probas[valid_mask][y[valid_mask] == 0]
    ax_dist.hist(probs_neg, bins=bins, alpha=0.6, color=ACCENT2, label="Invalid Signal")
    ax_dist.hist(probs_pos, bins=bins, alpha=0.6, color=ACCENT,  label="Valid Signal")
    ax_dist.axvline(0.5, color="white", linestyle="--", lw=1, alpha=0.7, label="Threshold 0.5")
    ax_dist.set_xlabel("Predicted Probability (Valid)", color=TEXT_COL)
    ax_dist.set_ylabel("Signal Count", color=TEXT_COL)
    ax_dist.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, fontsize=9)

# ── (E) 摘要統計 ──
ax_sum = fig.add_subplot(gs[2, 1])
style_ax(ax_sum, "Model Summary")
ax_sum.axis("off")

total    = int(N)
valid_n  = int(y.sum())
mean_p   = cv_df["precision"].mean() if len(cv_df) else float("nan")
mean_r   = cv_df["recall"].mean()    if len(cv_df) else float("nan")
mean_f1  = cv_df["f1"].mean()        if len(cv_df) else float("nan")
mean_auc = cv_df["auc"].mean()       if len(cv_df) else float("nan")

rows = [
    ("Architecture",     "GRU + Static FC Fusion"),
    ("Seq Length",        f"{SEQ_LEN} bars"),
    ("Seq Features",      f"{F_seq}  (OHLCV + derived)"),
    ("Static Features",   f"{F_static}  (SMC context)"),
    ("Total Signals",     f"{total:,}"),
    ("  Valid (1)",       f"{valid_n:,}  ({valid_n/total*100:.1f}%)"),
    ("  Invalid (0)",     f"{total-valid_n:,}  ({(total-valid_n)/total*100:.1f}%)"),
    ("─────────────",     "──────────────"),
    ("CV Folds",          f"{N_CV_SPLITS}  (gap={CV_GAP})"),
    ("Mean Precision",    f"{mean_p:.3f}"),
    ("Mean Recall",       f"{mean_r:.3f}"),
    ("Mean F1",           f"{mean_f1:.3f}"),
    ("Mean AUC",          f"{mean_auc:.3f}"),
    ("Device",            str(DEVICE)),
]
for i, (k, v) in enumerate(rows):
    y_pos = 1.0 - i * 0.067
    ax_sum.text(0.02, y_pos, k, transform=ax_sum.transAxes,
                color=TEXT_COL, fontsize=9, va="top")
    ax_sum.text(0.55, y_pos, v, transform=ax_sum.transAxes,
                color=ACCENT if i >= 9 else TEXT_COL,
                fontsize=9, va="top", fontweight="bold" if i >= 9 else "normal")

fig.suptitle("SMC Signal Validity — LSTM Hybrid Model Evaluation",
             color=TEXT_COL, fontsize=15, fontweight="bold", y=0.97)
plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"   ✅ 圖表已儲存：{REPORT_PATH}")

# ═══════════════════════════════════════════════════════
# ⑩ 完整分類報告
# ═══════════════════════════════════════════════════════
if valid_mask.sum() > 0:
    print("\n📋 完整 OOF 分類報告：")
    print(classification_report(
        y[valid_mask],
        oof_preds[valid_mask].astype(int),
        target_names=["Invalid Signal", "Valid Signal"],
    ))

print("\n✅ 全部完成！")
print(f"   - 評估報告：{REPORT_PATH}")
print(f"   - 模型檔案：{MODEL_SAVE_PATH}")
