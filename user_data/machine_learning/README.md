# Machine Learning — SMC 訊號有效性辨識

> 使用機器學習判斷 Smart Money Concepts (SMC) 偵測到的結構訊號是否「有效」，作為策略交易的二次過濾器。

---

## 目錄結構

```
machine_learning/
├── README.md                        # 本文件
├── feature_learning.py              # 主程式：特徵提取 → 標籤生成 → 模型訓練 → 視覺化
├── feature_learning_output/
│   ├── smc_ml_report.png            # 訓練評估報告圖（特徵重要性、混淆矩陣、CV 折線圖等）
│   └── smc_signals_with_labels.csv  # 每個訊號點的特徵向量 + 標籤（供後續再訓練使用）
└── smartmoneyconcepts/              # SMC 指標計算函式庫（本地版本）
    ├── __init__.py
    └── smc.py
```

---

## 目前實作：`feature_learning.py`

### 流程概覽

```
K 線資料 (.feather)
    │
    ▼
① SMC 指標計算
   Swing High/Low、FVG、Order Block、BOS/CHoCH、Liquidity
    │
    ▼
② 特徵工程（每個 FVG 訊號點）
   20 個上下文特徵（結構、蠟燭、市場脈絡）
    │
    ▼
③ 標籤生成（固定窗口盈虧比）
   未來 N 根內：最大浮盈 / 最大浮虧 ≥ RR_THRESHOLD → 有效(1)，否則無效(0)
    │
    ▼
④ Walk-Forward 交叉驗證（XGBoost）
   時序分割 + gap 防止 look-ahead bias
    │
    ▼
⑤ 評估輸出
   特徵重要性 / OOF 混淆矩陣 / CV 折線圖 / 機率分布
```

### 主要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `SYMBOL` | `BTC_USDT_USDT` | 交易對 |
| `TIMEFRAME` | `1h` | K 線時框 |
| `SWING_LENGTH` | `20` | Swing High/Low 偵測長度 |
| `FUTURE_WINDOW` | `15` | 標籤判斷的未來 K 線數 |
| `RR_THRESHOLD` | `1.0` | 有效訊號門檻（盈虧比） |
| `N_CV_SPLITS` | `5` | Walk-Forward 折數 |
| `CV_GAP` | `5` | 訓練/測試集隔離 gap（防 look-ahead） |

### 特徵列表（20 個）

| 類別 | 特徵 | 說明 |
|------|------|------|
| **FVG** | `fvg_present`, `fvg_size_atr`, `fvg_distance`, `fvg_mitigated` | Fair Value Gap 存在性、大小、距離、是否已緩解 |
| **Order Block** | `ob_present`, `ob_size_atr`, `ob_distance`, `ob_vol_ratio` | OB 存在性、大小、距離、成交量比率 |
| **結構** | `bos_count`, `choch_count`, `swing_count`, `trend_align` | BOS/CHoCH 次數、擺動點數量、趨勢一致性 |
| **流動性** | `nearby_liq_count` | 附近流動性點位數量 |
| **蠟燭形態** | `body_ratio`, `vol_ratio`, `upper_wick_atr`, `lower_wick_atr`, `candle_align` | 實體比、量比、上下影線、方向一致性 |
| **市場脈絡** | `dist_to_high_atr`, `dist_to_low_atr` | 距近期高低點的相對距離 |

### 模型

- **主要**：XGBoost（`n_estimators=200, max_depth=4, learning_rate=0.05`）
- **備用**：scikit-learn `GradientBoostingClassifier`（XGBoost 未安裝時自動切換）

---

## 執行方式

```bash
# 確保資料路徑正確（預設讀取 ../data_20240101-/binance/futures/）
cd /home/tseng/ft_userdata/user_data/machine_learning

# 安裝依賴
pip install xgboost scikit-learn pandas numpy matplotlib

# 執行訓練流程
python feature_learning.py
```

輸出結果會儲存至 `feature_learning_output/`：
- `smc_ml_report.png`：視覺化評估報告
- `smc_signals_with_labels.csv`：帶標籤的訊號特徵資料

---

## 未來願景：有效結構辨識學習模型

### 核心目標

> 訓練一個能夠「理解」市場微觀結構的模型，自動辨識哪些 SMC 訊號在當前市場環境下真正有效，並作為策略的智能過濾層。

### 短期規劃（近期擴充方向）

- [ ] **多觸發條件擴充**：目前僅以 FVG 作為候選進場點，未來加入 Order Block、BOS/CHoCH 重疊訊號作為觸發源
- [ ] **多時框特徵**：在 1h 訊號點上，加入 4h / 日線的趨勢方向作為上下文特徵（HTF Filter）
- [ ] **動態標籤**：根據 ATR 或結構高低點設定止損，取代固定 `FUTURE_WINDOW` 窗口
- [ ] **機率門檻調整**：根據 Precision/Recall 曲線動態選擇預測機率門檻（現為 0.5）

### 中期規劃（模型升級）

- [ ] **序列特徵建模**：每個訊號點前 N 根 K 線作為時序序列輸入（LSTM / Transformer），捕捉訂單流動態
- [ ] **訊號疊加評分**：建立多訊號疊加評分機制（FVG + OB + BOS + Liquidity 同時成立時加權），訓練一個評分→有效性的回歸/分類模型
- [ ] **線上學習 / 滾動訓練**：每月以最新資料重新訓練，偵測市場 regime 切換

### 長期願景（策略整合）

- [ ] **Freqtrade 整合**：將訓練好的模型存為 `.pkl`，在 Freqtrade 策略中即時推論，實現「SMC 偵測 + ML 過濾」的完整交易流程
- [ ] **市場狀態分類器**：獨立訓練一個 regime 分類模型（趨勢 / 震盪 / 高波動），條件式切換不同的 ML 過濾策略
- [ ] **強化學習探索**：以 RL 替代固定盈虧比標籤，讓模型自行學習最優進出場時機

### 技術架構藍圖

```
                    ┌─────────────────────────────────────┐
                    │         原始 K 線資料                │
                    │   (Binance Futures, 多時框)           │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       SMC 指標層                     │
                    │  FVG / OB / BOS / CHoCH / Liquidity  │
                    └──────────────┬──────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
    ┌──────────▼────────┐ ┌────────▼────────┐ ┌───────▼────────────┐
    │  特徵工程層        │ │  HTF 上下文層   │ │  序列特徵層 (未來)  │
    │  20 個靜態特徵     │ │  4h / 日線趨勢  │ │  LSTM / Transformer │
    └──────────┬────────┘ └────────┬────────┘ └───────┬────────────┘
               └───────────────────┼───────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       ML 分類模型                    │
                    │  XGBoost → 機率分數 (0~1)            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │     Freqtrade 策略整合               │
                    │  SMC 訊號 + ML 過濾 → 下單決策       │
                    └─────────────────────────────────────┘
```

---

## 資料來源

- K 線資料：`../data_20240101-/binance/futures/{SYMBOL}-{TIMEFRAME}-futures.feather`
- SMC 指標：`smartmoneyconcepts/smc.py`（本地版本，基於 [joshyattom/smartmoneyconcepts](https://github.com/joshyattom/smartmoneyconcepts)）
