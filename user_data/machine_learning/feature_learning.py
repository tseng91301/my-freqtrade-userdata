"""
SMC 訊號有效性判斷 - 機器學習學習流程
=========================================
流程：
  1. 加載資料並計算所有 SMC 指標
  2. 特徵工程：從每個偵測到的 SMC 事件提取上下文特徵
  3. 標籤生成：使用「固定窗口盈虧比」判斷訊號有效性
  4. 模型訓練：XGBoost + Walk-Forward 交叉驗證
  5. 評估：特徵重要性 + 混淆矩陣 + Precision/Recall
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from smartmoneyconcepts import smc
import os

# ── 嘗試載入 XGBoost（優先），退回 scikit-learn 的 GradientBoosting ──
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGBOOST = False
    print("⚠️  XGBoost 未安裝，改用 scikit-learn GradientBoostingClassifier")
    print("   建議安裝：pip install xgboost")

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════
# ① 設定參數
# ═══════════════════════════════════════════════════════
FEATURE_STORING_DIR = "../data_20240101-/binance/futures/"
SYMBOL              = "BTC_USDT_USDT"
TIMEFRAME           = "1h"
DATA_FILE           = f"{FEATURE_STORING_DIR}{SYMBOL}-{TIMEFRAME}-futures.feather"

# 訊號相關
SWING_LENGTH      = 25    # Swing High/Low 偵測長度
MAX_LABEL_WINDOW  = 100   # 模擬最多往未來看幾根（防無限等待）
MIN_RR_FILTER     = 1.0   # 進場前先過濾：目標/止損距離 RR < 此值則跳過

# 訓練相關
N_CV_SPLITS       = 5     # Walk-Forward 折數
CV_GAP            = 5     # 訓練集 & 測試集間的隔離 gap（防 look-ahead）
MIN_SIGNAL_COUNT  = 30    # 每個方向最少需要的訊號數，否則跳過訓練

# 輸出
OUTPUT_DIR        = "./feature_learning_output/"

print("=" * 60)
print("  SMC 訊號有效性判斷 - ML 學習流程")
print("=" * 60)

# ═══════════════════════════════════════════════════════
# ② 加載資料
# ═══════════════════════════════════════════════════════
print(f"\n📂 讀取資料：{DATA_FILE}")
df = pd.read_feather(DATA_FILE)
df = df.reset_index(drop=True)
print(f"   總長度：{len(df)} 根 K 線")

# ─── 確保欄位名稱為小寫（smartmoneyconcepts 要求）───
df.columns = df.columns.str.lower()
required_cols = {"open", "high", "low", "close", "volume"}
assert required_cols.issubset(df.columns), \
    f"缺少必要欄位：{required_cols - set(df.columns)}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# ③ 計算 SMC 指標
# ═══════════════════════════════════════════════════════
print("\n📊 計算 SMC 指標...")

swing_hl  = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
fvg_data  = smc.fvg(df, join_consecutive=False)
ob_data   = smc.ob(df, swing_hl, close_mitigation=False)
bos_data  = smc.bos_choch(df, swing_hl, close_break=True)
liq_data  = smc.liquidity(df, swing_hl)

print("   ✅ Swing High/Low、FVG、Order Block、BOS/CHoCH、Liquidity 計算完成")

# ═══════════════════════════════════════════════════════
# ④ 輔助特徵：ATR、成交量均線
# ═══════════════════════════════════════════════════════
df["atr"]       = (df["high"] - df["low"]).rolling(14).mean()
df["vol_ma20"]  = df["volume"].rolling(20).mean()
df["body_size"] = (df["close"] - df["open"]).abs()
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
# 趨勢：過去 20 根的收盤走勢斜率（標準化）
df["trend_slope"] = (
    df["close"].rolling(20)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    / df["atr"]
)

# ═══════════════════════════════════════════════════════
# ⑤ 特徵提取函式
# ═══════════════════════════════════════════════════════

def extract_features_at(idx: int, direction: int) -> dict:
    """
    在第 idx 根 K 線提取所有特徵。
    direction: +1 (多頭訊號)，-1 (空頭訊號)
    """
    if idx < SWING_LENGTH + 20 or idx + MAX_LABEL_WINDOW >= len(df):
        return None

    row = df.iloc[idx]
    atr = row["atr"] if row["atr"] > 0 else 1e-8

    # —— FVG 特徵 ——
    # 尋找在 idx 之前最近的同方向 FVG
    fvg_col = fvg_data["FVG"]
    recent_fvg = fvg_col[:idx + 1]
    same_dir_fvg = recent_fvg[recent_fvg == direction]
    if len(same_dir_fvg) > 0:
        fvg_idx = same_dir_fvg.index[-1]
        fvg_size = (fvg_data["Top"][fvg_idx] - fvg_data["Bottom"][fvg_idx]) / atr
        fvg_distance = (idx - fvg_idx) / SWING_LENGTH
        fvg_mitigated = 0 if pd.isna(fvg_data["MitigatedIndex"][fvg_idx]) else 1
    else:
        fvg_size, fvg_distance, fvg_mitigated = 0.0, 99.0, 0

    # —— Order Block 特徵 ——
    ob_col = ob_data["OB"] if "OB" in ob_data.columns else pd.Series(0, index=df.index)
    recent_ob = ob_col[:idx + 1]
    same_dir_ob = recent_ob[recent_ob == direction]
    if len(same_dir_ob) > 0:
        ob_idx = same_dir_ob.index[-1]
        ob_top = ob_data["Top"][ob_idx] if "Top" in ob_data.columns else row["high"]
        ob_bot = ob_data["Bottom"][ob_idx] if "Bottom" in ob_data.columns else row["low"]
        ob_size = (ob_top - ob_bot) / atr
        ob_distance = (idx - ob_idx) / SWING_LENGTH
        ob_vol_ratio = (df["volume"].iloc[ob_idx] / df["vol_ma20"].iloc[ob_idx]
                        if df["vol_ma20"].iloc[ob_idx] > 0 else 1.0)
    else:
        ob_size, ob_distance, ob_vol_ratio = 0.0, 99.0, 1.0

    # —— BOS / CHoCH 特徵 ——
    bos_val_col = bos_data["BOS"] if "BOS" in bos_data.columns else pd.Series(0, index=df.index)
    choch_col   = bos_data["CHOCH"] if "CHOCH" in bos_data.columns else pd.Series(0, index=df.index)
    recent_bos_same  = (bos_val_col[:idx + 1] == direction).sum()
    recent_choch_same = (choch_col[:idx + 1] == direction).sum()

    # —— Swing 結構特徵 ——
    shl_col = swing_hl["HighLow"]
    recent_shl = shl_col[:idx + 1][shl_col[:idx + 1] != 0]
    swing_count = len(recent_shl.tail(6))                # 最近 6 個擺動點
    # 趨勢一致性：多頭訊號時，若最近趨勢斜率 > 0 則 +1
    trend_align = 1 if (direction * row.get("trend_slope", 0)) > 0 else 0

    # —— Liquidity 特徵 ——
    liq_col = liq_data["Liquidity"] if "Liquidity" in liq_data.columns else pd.Series(0, index=df.index)
    nearby_liq = (liq_col[max(0, idx - SWING_LENGTH): idx + 1] != 0).sum()

    # —— 蠟燭形態特徵 ——
    body_ratio  = row["body_size"] / (row["high"] - row["low"] + 1e-8)
    vol_ratio   = row["volume"] / row["vol_ma20"] if row["vol_ma20"] > 0 else 1.0
    upper_wick_ratio = row["upper_wick"] / atr
    lower_wick_ratio = row["lower_wick"] / atr
    # 訊號蠟燭方向是否與訊號方向一致
    candle_align = 1 if (row["close"] - row["open"]) * direction > 0 else 0

    # —— 市場脈絡 ——
    # 距離最近 Swing High/Low 的相對位置
    highs = df["high"].iloc[max(0, idx - SWING_LENGTH * 2): idx]
    lows  = df["low"].iloc[max(0, idx - SWING_LENGTH * 2): idx]
    dist_to_high = (highs.max() - row["close"]) / atr if len(highs) > 0 else 0.0
    dist_to_low  = (row["close"] - lows.min()) / atr if len(lows) > 0 else 0.0

    return {
        # FVG
        "fvg_present":      1 if fvg_size > 0 else 0,
        "fvg_size_atr":     fvg_size,
        "fvg_distance":     fvg_distance,
        "fvg_mitigated":    fvg_mitigated,
        # Order Block
        "ob_present":       1 if ob_size > 0 else 0,
        "ob_size_atr":      ob_size,
        "ob_distance":      ob_distance,
        "ob_vol_ratio":     ob_vol_ratio,
        # 結構
        "bos_count":        int(recent_bos_same),
        "choch_count":      int(recent_choch_same),
        "swing_count":      swing_count,
        "trend_align":      trend_align,
        # 流動性
        "nearby_liq_count": int(nearby_liq),
        # 蠟燭
        "body_ratio":       body_ratio,
        "vol_ratio":        vol_ratio,
        "upper_wick_atr":   upper_wick_ratio,
        "lower_wick_atr":   lower_wick_ratio,
        "candle_align":     candle_align,
        # 市場脈絡
        "dist_to_high_atr": dist_to_high,
        "dist_to_low_atr":  dist_to_low,
        "atr_val":          atr,
    }


# ═══════════════════════════════════════════════════════
# ⑥ 標籤生成：SMC 結構目標（前高/前低 vs OB 止損）
# ═══════════════════════════════════════════════════════

def find_swing_target(idx: int, direction: int) -> float:
    """
    在訊號點 idx 之前，找最近的「結構目標」。
    多頭(+1)：找 idx 之前、收盤價以上最近的擺盪高點（前高 = buy-side liquidity）
    空頭(-1)：找 idx 之前、收盤價以下最近的擺盪低點（前低 = sell-side liquidity）
    回傳目標價，找不到則回傳 np.nan。
    """
    entry = df["close"].iloc[idx]
    shl = swing_hl["HighLow"][:idx]           # idx 之前的擺盪標記

    if direction == 1:
        # 擺盪高點（HighLow == 1）且高於進場價
        highs_idx = shl[shl == 1].index
        candidates = df["high"].loc[highs_idx]
        above = candidates[candidates > entry]
        if len(above) == 0:
            return np.nan
        return float(above.iloc[-1])           # 最近的前高
    else:
        # 擺盪低點（HighLow == -1）且低於進場價
        lows_idx = shl[shl == -1].index
        candidates = df["low"].loc[lows_idx]
        below = candidates[candidates < entry]
        if len(below) == 0:
            return np.nan
        return float(below.iloc[-1])           # 最近的前低


def find_ob_stop(idx: int, direction: int) -> float:
    """
    取最近同方向 OB 的邊緣作為止損位。
    多頭(+1)：OB 底部（Bottom）下方 = 止損
    空頭(-1)：OB 頂部（Top）上方 = 止損
    找不到 OB 則以訊號前 SWING_LENGTH 根的極值作為備用止損。
    """
    ob_col = ob_data["OB"] if "OB" in ob_data.columns else pd.Series(0, index=df.index)
    recent_ob = ob_col[:idx + 1]
    same_dir_ob = recent_ob[recent_ob == direction]

    if len(same_dir_ob) > 0:
        ob_idx = same_dir_ob.index[-1]
        if direction == 1:
            return float(ob_data["Bottom"][ob_idx]) if "Bottom" in ob_data.columns \
                   else float(df["low"].iloc[ob_idx])
        else:
            return float(ob_data["Top"][ob_idx]) if "Top" in ob_data.columns \
                   else float(df["high"].iloc[ob_idx])

    # 備用：前 SWING_LENGTH 根的極值
    window = df.iloc[max(0, idx - SWING_LENGTH): idx]
    return float(window["low"].min()) if direction == 1 else float(window["high"].max())


def compute_label(idx: int, direction: int) -> int:
    """
    SMC 結構目標標籤：
      目標  = 最近前高（多頭）/ 前低（空頭）
      止損  = 最近同方向 OB 邊緣
      模擬  = 逐根 K 線往後掃，哪個先達到：
              目標先到 → label=1（有效）
              止損先到 → label=0（無效）
              MAX_LABEL_WINDOW 內都沒到 → label=0
      前置過濾：若 RR < MIN_RR_FILTER，直接跳過（回傳 (label, target, stop)，若跳過則回傳 (np.nan, np.nan, np.nan)）
    """
    if idx + 1 >= len(df):
        return np.nan, np.nan, np.nan

    entry  = df["close"].iloc[idx]
    target = find_swing_target(idx, direction)
    stop   = find_ob_stop(idx, direction)

    # 目標 / 止損找不到 → 跳過
    if np.isnan(target) or np.isnan(stop):
        return np.nan, np.nan, np.nan

    # 多頭：目標必須在進場上方，止損在進場下方
    if direction == 1:
        if target <= entry or stop >= entry:
            return np.nan, np.nan, np.nan
    else:
        if target >= entry or stop <= entry:
            return np.nan, np.nan, np.nan

    # RR 前置過濾
    reward = abs(target - entry)
    risk   = abs(entry - stop)
    if risk <= 0 or (reward / risk) < MIN_RR_FILTER:
        return np.nan, np.nan, np.nan

    # 逐根模擬
    end = min(idx + 1 + MAX_LABEL_WINDOW, len(df))
    for i in range(idx + 1, end):
        bar_high = df["high"].iloc[i]
        bar_low  = df["low"].iloc[i]
        if direction == 1:
            if bar_high >= target:
                return 1, target, stop   # 目標先到 → 有效
            if bar_low <= stop:
                return 0, target, stop   # 止損先到 → 無效
        else:
            if bar_low <= target:
                return 1, target, stop
            if bar_high >= stop:
                return 0, target, stop

    return 0, target, stop   # 逾時 → 無效


# ═══════════════════════════════════════════════════════
# ⑦ 收集所有訊號點（FVG + BOS 產生的交易候選）
# ═══════════════════════════════════════════════════════
print("\n🔍 收集 SMC 訊號點...")

records = []
seen = set()   # 防止同一 idx 重複加入

def _try_add(idx_val, dir_val):
    key = (idx_val, dir_val)
    if key in seen:
        return
    feats = extract_features_at(idx_val, dir_val)
    if feats is None:
        return
    label, target, stop = compute_label(idx_val, dir_val)
    if pd.isna(label):
        return
    seen.add(key)
    
    # 紀錄額外資訊
    date = df["date"].iloc[idx_val] if "date" in df.columns else df.index[idx_val]
    fvg_top = fvg_data["Top"][idx_val] if pd.notna(fvg_data["Top"][idx_val]) else np.nan
    fvg_bottom = fvg_data["Bottom"][idx_val] if pd.notna(fvg_data["Bottom"][idx_val]) else np.nan

    records.append({
        "idx":           idx_val,
        "date":          date,
        "direction":     dir_val,
        "label":         int(label),
        "fvg_top":       fvg_top,
        "fvg_bottom":    fvg_bottom,
        "target_price":  target,
        "stop_price":    stop,
        **feats
    })

# ① FVG 作為觸發點
fvg_col = fvg_data["FVG"]
for idx in fvg_col.dropna().index:
    _try_add(int(idx), int(fvg_col[idx]))

# ② BOS 作為觸發點
if "BOS" in bos_data.columns:
    bos_col = bos_data["BOS"]
    for idx in bos_col.dropna().index:
        if bos_col[idx] != 0:
            _try_add(int(idx), int(np.sign(bos_col[idx])))

# ③ CHoCH 作為觸發點
if "CHOCH" in bos_data.columns:
    choch_col = bos_data["CHOCH"]
    for idx in choch_col.dropna().index:
        if choch_col[idx] != 0:
            _try_add(int(idx), int(np.sign(choch_col[idx])))

df_signals = pd.DataFrame(records)
# 依訊號點位置排序（確保 Walk-Forward 時序正確）
df_signals = df_signals.sort_values("idx").reset_index(drop=True)

print(f"   收集到 {len(df_signals)} 個訊號點（FVG + BOS + CHoCH）")
print(f"   多頭訊號：{(df_signals['direction'] == 1).sum()}")
print(f"   空頭訊號：{(df_signals['direction'] == -1).sum()}")
print(f"   有效訊號（標籤=1）：{df_signals['label'].sum()} "
      f"({df_signals['label'].mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════
# ⑧ 模型訓練（Walk-Forward Cross-Validation）
# ═══════════════════════════════════════════════════════
FEATURE_COLS = [c for c in df_signals.columns
                if c not in ("idx", "date", "direction", "label", "atr_val", 
                             "fvg_top", "fvg_bottom", "target_price", "stop_price")]

def build_model():
    if USE_XGBOOST:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )


def walk_forward_cv(df_sig: pd.DataFrame, n_splits: int, gap: int):
    """
    Walk-Forward CV：時序分割 + gap 防止資訊洩漏。
    回傳：每折的指標、所有折的 OOF 預測。
    """
    n = len(df_sig)
    fold_size = n // (n_splits + 1)
    results = []
    oof_preds  = np.full(n, np.nan)
    oof_probas = np.full(n, np.nan)

    for fold in range(n_splits):
        train_end  = fold_size * (fold + 1)
        test_start = train_end + gap
        test_end   = test_start + fold_size if fold < n_splits - 1 else n

        if test_start >= test_end:
            continue

        X_train = df_sig[FEATURE_COLS].iloc[:train_end].values
        y_train = df_sig["label"].iloc[:train_end].values
        X_test  = df_sig[FEATURE_COLS].iloc[test_start:test_end].values
        y_test  = df_sig["label"].iloc[test_start:test_end].values

        if len(np.unique(y_train)) < 2:
            continue  # 跳過標籤只有一類的折

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = build_model()
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        oof_preds[test_start:test_end]  = y_pred
        oof_probas[test_start:test_end] = y_proba

        fold_result = {
            "fold":      fold + 1,
            "train_n":   train_end,
            "test_n":    test_end - test_start,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "auc":       roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
        }
        print(f"   Fold {fold+1}: Prec={fold_result['precision']:.3f}  "
              f"Recall={fold_result['recall']:.3f}  "
              f"F1={fold_result['f1']:.3f}  "
              f"AUC={fold_result['auc']:.3f}")
        results.append(fold_result)

    return results, oof_preds, oof_probas


print(f"\n🔁 Walk-Forward CV（{N_CV_SPLITS} 折，gap={CV_GAP}）...")
cv_results, oof_preds, oof_probas = walk_forward_cv(df_signals, N_CV_SPLITS, CV_GAP)
cv_df = pd.DataFrame(cv_results)
print("\n── 平均指標 ──")
print(cv_df[["precision", "recall", "f1", "auc"]].mean().to_string())

# ═══════════════════════════════════════════════════════
# ⑨ 全資料訓練最終模型（用於特徵重要性分析）
# ═══════════════════════════════════════════════════════
print("\n🏋️  訓練最終模型（全資料）...")
X_all = df_signals[FEATURE_COLS].values
y_all = df_signals["label"].values
scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X_all)

final_model = build_model()
final_model.fit(X_all_scaled, y_all)
print("   ✅ 完成")

# ═══════════════════════════════════════════════════════
# ⑩ 視覺化
# ═══════════════════════════════════════════════════════
print("\n📈 產生視覺化圖表...")

fig = plt.figure(figsize=(18, 16), facecolor="#0e1117")
gs  = gridspec.GridSpec(3, 2, figure=fig,
                         hspace=0.45, wspace=0.35,
                         top=0.92, bottom=0.07)

DARK_BG   = "#0e1117"
CARD_BG   = "#1a1d2e"
ACCENT    = "#00d4aa"
ACCENT2   = "#ff6b6b"
TEXT_COL  = "#e0e0e0"
GRID_COL  = "#2a2d3e"

def style_ax(ax, title=""):
    ax.set_facecolor(CARD_BG)
    ax.spines[["top", "right", "left", "bottom"]].set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)

# ── (A) 特徵重要性 ──
ax_fi = fig.add_subplot(gs[0, :])
style_ax(ax_fi, "Feature Importance")

if USE_XGBOOST:
    importances = final_model.feature_importances_
else:
    importances = final_model.feature_importances_

fi_series = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
colors = [ACCENT if v >= fi_series.quantile(0.7) else ACCENT2
          for v in fi_series.values]
bars = ax_fi.barh(fi_series.index, fi_series.values,
                  color=colors, alpha=0.85, height=0.7)
ax_fi.set_xlabel("Importance Score", color=TEXT_COL, fontsize=9)
ax_fi.tick_params(axis="y", labelsize=8)
for bar, val in zip(bars, fi_series.values):
    ax_fi.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
               f"{val:.3f}", va="center", ha="left",
               color=TEXT_COL, fontsize=7)

# ── (B) OOF 混淆矩陣 ──
ax_cm = fig.add_subplot(gs[1, 0])
style_ax(ax_cm, "OOF Confusion Matrix (Out-of-Fold)")

valid_mask = ~np.isnan(oof_preds)
if valid_mask.sum() > 0:
    cm = confusion_matrix(y_all[valid_mask], oof_preds[valid_mask].astype(int))
    cmap = LinearSegmentedColormap.from_list("teal", ["#0e1117", ACCENT])
    im = ax_cm.imshow(cm, cmap=cmap, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha="center", va="center",
                       color="white", fontsize=14, fontweight="bold")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred: Invalid", "Pred: Valid"], color=TEXT_COL, fontsize=9)
    ax_cm.set_yticklabels(["True: Invalid", "True: Valid"], color=TEXT_COL, fontsize=9)

# ── (C) Walk-Forward 各折指標 ──
ax_cv = fig.add_subplot(gs[1, 1])
style_ax(ax_cv, "Walk-Forward CV Performance per Fold")

if len(cv_df) > 0:
    x = cv_df["fold"].values
    width = 0.22
    ax_cv.bar(x - width, cv_df["precision"], width, label="Precision",
              color=ACCENT,  alpha=0.85)
    ax_cv.bar(x,         cv_df["recall"],    width, label="Recall",
              color=ACCENT2, alpha=0.85)
    ax_cv.bar(x + width, cv_df["f1"],        width, label="F1",
              color="#f5a623", alpha=0.85)
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels([f"Fold {i}" for i in x], color=TEXT_COL)
    ax_cv.set_ylim(0, 1.1)
    ax_cv.set_ylabel("Score", color=TEXT_COL)
    ax_cv.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, fontsize=9)

# ── (D) OOF 預測機率分布 ──
ax_dist = fig.add_subplot(gs[2, 0])
style_ax(ax_dist, "ML Predicted Probability Distribution (Valid vs Invalid)")

valid_idx = np.where(~np.isnan(oof_probas))[0]
if len(valid_idx) > 0:
    probs_pos = oof_probas[valid_idx][y_all[valid_idx] == 1]
    probs_neg = oof_probas[valid_idx][y_all[valid_idx] == 0]
    bins = np.linspace(0, 1, 25)
    ax_dist.hist(probs_neg, bins=bins, alpha=0.6, color=ACCENT2, label="Invalid Signal")
    ax_dist.hist(probs_pos, bins=bins, alpha=0.6, color=ACCENT,  label="Valid Signal")
    ax_dist.axvline(0.5, color="white", linestyle="--", lw=1, alpha=0.7, label="Threshold 0.5")
    ax_dist.set_xlabel("Predicted Probability (Valid)", color=TEXT_COL)
    ax_dist.set_ylabel("Signal Count", color=TEXT_COL)
    ax_dist.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, fontsize=9)

# ── (E) 標籤分布 & 統計摘要 ──
ax_sum = fig.add_subplot(gs[2, 1])
style_ax(ax_sum, "Signal Summary")
ax_sum.axis("off")

total   = len(df_signals)
valid_n = int(df_signals["label"].sum())
invalid_n = total - valid_n
mean_prec = cv_df["precision"].mean() if len(cv_df) > 0 else float("nan")
mean_rec  = cv_df["recall"].mean()    if len(cv_df) > 0 else float("nan")
mean_f1   = cv_df["f1"].mean()        if len(cv_df) > 0 else float("nan")
mean_auc  = cv_df["auc"].mean()       if len(cv_df) > 0 else float("nan")

summary_rows = [
    ("Timeframe",           f"{TIMEFRAME}"),
    ("Total Candles",       f"{len(df):,}"),
    ("Total FVG Signals",   f"{total:,}"),
    ("  Valid (label=1)",   f"{valid_n:,}  ({valid_n/total*100:.1f}%)"),
    ("  Invalid (label=0)", f"{invalid_n:,}  ({invalid_n/total*100:.1f}%)"),
    ("Label Method",        "SMC Structure Target"),
    ("Min RR Filter",       f">= {MIN_RR_FILTER}"),
    ("Max Label Window",    f"{MAX_LABEL_WINDOW} bars"),
    ("Target",              "Nearest Swing High/Low"),
    ("Stop Loss",           "OB Edge (or Swing Extreme)"),
    ("-------------",       "----------"),
    ("CV Folds",            f"{N_CV_SPLITS}"),
    ("Mean Precision",      f"{mean_prec:.3f}"),
    ("Mean Recall",         f"{mean_rec:.3f}"),
    ("Mean F1",             f"{mean_f1:.3f}"),
    ("Mean AUC",            f"{mean_auc:.3f}"),
    ("Model",               "XGBoost" if USE_XGBOOST else "GradientBoosting"),
]

for row_i, (k, v) in enumerate(summary_rows):
    y_pos = 1.0 - row_i * 0.067
    ax_sum.text(0.02, y_pos, k, transform=ax_sum.transAxes,
                color=TEXT_COL if not k.startswith("─") else GRID_COL,
                fontsize=9, va="top")
    ax_sum.text(0.55, y_pos, v, transform=ax_sum.transAxes,
                color=ACCENT if row_i > 8 else TEXT_COL,
                fontsize=9, va="top", fontweight="bold" if row_i > 8 else "normal")

# 標題
fig.suptitle("SMC Signal Validity — Machine Learning Evaluation Report",
             color=TEXT_COL, fontsize=15, fontweight="bold", y=0.97)

out_path = f"{OUTPUT_DIR}smc_ml_report.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"   ✅ 圖表已儲存：{out_path}")

# ═══════════════════════════════════════════════════════
# ⑪ 輸出完整分類報告
# ═══════════════════════════════════════════════════════
valid_mask = ~np.isnan(oof_preds)
if valid_mask.sum() > 0:
    print("\n📋 完整 OOF 分類報告：")
    print(classification_report(
        y_all[valid_mask],
        oof_preds[valid_mask].astype(int),
        target_names=["Invalid Signal", "Valid Signal"]
    ))

# ═══════════════════════════════════════════════════════
# ⑫ 儲存特徵資料供後續使用
# ═══════════════════════════════════════════════════════
signals_out = f"{OUTPUT_DIR}smc_signals_with_labels.csv"
df_signals.to_csv(signals_out, index=False)
print(f"💾 訊號特徵資料已儲存：{signals_out}")

print("\n✅ 全部完成！")
print(f"   - 報告圖表：{out_path}")
print(f"   - 特徵資料：{signals_out}")