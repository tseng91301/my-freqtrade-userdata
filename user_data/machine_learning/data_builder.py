"""
data_builder.py — SMC 序列資料建構模組
=========================================
功能：
  1. 讀取原始 OHLCV 資料（feather）
  2. 讀取 feature_learning.py 產生的訊號點 CSV（含靜態特徵 + 標籤）
  3. 為每個訊號點建構前 SEQ_LEN 根 K 線的序列特徵
  4. 標準化（序列 / 靜態 分開 scale）
  5. 儲存 .npz + scaler（供 sequence_learning.py 使用）

使用方式：
  python data_builder.py
  或 from data_builder import build_dataset, load_dataset
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# ═══════════════════════════════════════════════════════
# ① 參數
# ═══════════════════════════════════════════════════════
FEATURE_STORING_DIR = "../data_20240101-/binance/futures/"
SYMBOL              = "BTC_USDT_USDT"
TIMEFRAME           = "1h"
DATA_FILE           = f"{FEATURE_STORING_DIR}{SYMBOL}-{TIMEFRAME}-futures.feather"

SIGNALS_CSV         = "./feature_learning_output/smc_signals_with_labels.csv"
OUTPUT_DIR          = "./feature_learning_output/"
DATASET_FILE        = f"{OUTPUT_DIR}lstm_dataset.npz"
SCALER_FILE         = f"{OUTPUT_DIR}lstm_scalers.pkl"

# 序列長度：訊號點之前取多少根 K 線作為走勢輸入
SEQ_LEN = 50

# 序列原始特徵欄位（來自 OHLCV + 衍生）
SEQ_RAW_COLS = ["open", "high", "low", "close", "volume",
                "body_size", "upper_wick", "lower_wick",
                "atr", "vol_ma20", "trend_slope"]

# 靜態特徵：排除非特徵欄位
EXCLUDE_COLS = {"idx", "direction", "label", "atr_val"}


# ═══════════════════════════════════════════════════════
# ② 衍生欄位計算（對齊 feature_learning.py）
# ═══════════════════════════════════════════════════════
def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 OHLCV DataFrame 上計算衍生欄位。
    與 feature_learning.py 保持一致，以確保序列特徵和靜態特徵在同一基準下。
    """
    df = df.copy()
    df["atr"]         = (df["high"] - df["low"]).rolling(14).mean()
    df["vol_ma20"]    = df["volume"].rolling(20).mean()
    df["body_size"]   = (df["close"] - df["open"]).abs()
    df["upper_wick"]  = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]  = df[["open", "close"]].min(axis=1) - df["low"]
    df["trend_slope"] = (
        df["close"].rolling(20)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        / df["atr"].replace(0, np.nan)
    ).fillna(0.0)
    return df


# ═══════════════════════════════════════════════════════
# ③ 主建構函式
# ═══════════════════════════════════════════════════════
def build_dataset(
    data_file: str = DATA_FILE,
    signals_csv: str = SIGNALS_CSV,
    seq_len: int = SEQ_LEN,
    output_dir: str = OUTPUT_DIR,
    save: bool = True,
) -> tuple:
    """
    建構 LSTM 訓練資料集。

    回傳：
        X_seq    : np.ndarray, shape (N, seq_len, n_seq_features)
        X_static : np.ndarray, shape (N, n_static_features)
        y        : np.ndarray, shape (N,)
        seq_scaler    : fitted StandardScaler（用於序列）
        static_scaler : fitted StandardScaler（用於靜態特徵）
        feature_names : list of static feature names
        signal_indices: np.ndarray，每個樣本對應的 K 線 idx（供 walk-forward 分割使用）
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 讀取原始 K 線資料 ──
    print(f"📂 讀取 K 線資料：{data_file}")
    df_ohlcv = pd.read_feather(data_file).reset_index(drop=True)
    df_ohlcv.columns = df_ohlcv.columns.str.lower()
    df_ohlcv = add_derived_cols(df_ohlcv)
    print(f"   K 線總數：{len(df_ohlcv)}")

    # 確認序列特徵欄位都存在
    missing = [c for c in SEQ_RAW_COLS if c not in df_ohlcv.columns]
    if missing:
        raise ValueError(f"K 線資料缺少欄位：{missing}")

    # ── 讀取訊號 CSV ──
    print(f"📂 讀取訊號資料：{signals_csv}")
    df_sig = pd.read_csv(signals_csv)
    print(f"   訊號數量：{len(df_sig)}")

    static_feature_cols = [c for c in df_sig.columns if c not in EXCLUDE_COLS]

    # ── 建構序列 + 靜態特徵 ──
    print(f"🔧 建構序列特徵（seq_len={seq_len}）...")

    seq_list    = []
    static_list = []
    label_list  = []
    idx_list    = []
    skipped     = 0

    ohlcv_arr = df_ohlcv[SEQ_RAW_COLS].values  # 預先轉 numpy，加速

    for _, row in df_sig.iterrows():
        signal_idx = int(row["idx"])

        # 序列範圍：[signal_idx - seq_len, signal_idx)
        # 注意：不包含訊號點本身，防止 data leakage
        start = signal_idx - seq_len
        if start < 0:
            skipped += 1
            continue

        seq = ohlcv_arr[start:signal_idx]          # shape (seq_len, n_seq_features)
        if seq.shape[0] != seq_len or np.any(np.isnan(seq)):
            skipped += 1
            continue

        static = row[static_feature_cols].values.astype(np.float32)
        if np.any(np.isnan(static)):
            static = np.nan_to_num(static, nan=0.0)

        seq_list.append(seq)
        static_list.append(static)
        label_list.append(int(row["label"]))
        idx_list.append(signal_idx)

    print(f"   有效樣本：{len(seq_list)}  跳過：{skipped}")

    X_seq    = np.array(seq_list,    dtype=np.float32)   # (N, S, F_seq)
    X_static = np.array(static_list, dtype=np.float32)   # (N, F_static)
    y        = np.array(label_list,  dtype=np.int64)
    signal_indices = np.array(idx_list, dtype=np.int64)

    # ── 標準化 ──
    print("📐 標準化...")

    # 序列：將 (N, S, F) reshape 成 (N*S, F) → fit → reshape 回來
    N, S, F_seq = X_seq.shape
    seq_scaler = StandardScaler()
    X_seq_2d = X_seq.reshape(-1, F_seq)
    X_seq_scaled = seq_scaler.fit_transform(X_seq_2d).reshape(N, S, F_seq).astype(np.float32)

    # 靜態特徵
    static_scaler = StandardScaler()
    X_static_scaled = static_scaler.fit_transform(X_static).astype(np.float32)

    print(f"   序列 shape：{X_seq_scaled.shape}")
    print(f"   靜態 shape：{X_static_scaled.shape}")
    print(f"   標籤分布：Valid={y.sum()}  Invalid={(y==0).sum()}")

    # ── 儲存 ──
    if save:
        np.savez(
            DATASET_FILE,
            X_seq=X_seq_scaled,
            X_static=X_static_scaled,
            y=y,
            signal_indices=signal_indices,
        )
        with open(SCALER_FILE, "wb") as f:
            pickle.dump({
                "seq_scaler": seq_scaler,
                "static_scaler": static_scaler,
                "static_feature_cols": static_feature_cols,
                "seq_feature_cols": SEQ_RAW_COLS,
            }, f)
        print(f"💾 資料集已儲存：{DATASET_FILE}")
        print(f"💾 Scaler 已儲存：{SCALER_FILE}")

    return (X_seq_scaled, X_static_scaled, y,
            seq_scaler, static_scaler, static_feature_cols, signal_indices)


# ═══════════════════════════════════════════════════════
# ④ 讀取已儲存資料集
# ═══════════════════════════════════════════════════════
def load_dataset(
    dataset_file: str = DATASET_FILE,
    scaler_file: str = SCALER_FILE,
) -> tuple:
    """讀取已建構好的資料集（避免重複計算）。"""
    data = np.load(dataset_file)
    with open(scaler_file, "rb") as f:
        scalers = pickle.load(f)

    print(f"✅ 讀取資料集：{dataset_file}")
    print(f"   序列 shape：{data['X_seq'].shape}")
    print(f"   靜態 shape：{data['X_static'].shape}")
    print(f"   標籤分布：Valid={data['y'].sum()}  Invalid={(data['y']==0).sum()}")

    return (
        data["X_seq"],
        data["X_static"],
        data["y"],
        scalers["seq_scaler"],
        scalers["static_scaler"],
        scalers["static_feature_cols"],
        data["signal_indices"],
    )


# ═══════════════════════════════════════════════════════
# ⑤ 直接執行時建構並儲存
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  SMC LSTM 資料建構器")
    print("=" * 60)
    build_dataset()
    print("\n✅ 完成！可執行 sequence_learning.py 開始訓練。")
