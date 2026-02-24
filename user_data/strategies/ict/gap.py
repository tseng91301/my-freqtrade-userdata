import numpy as np
import pandas as pd

def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.2, basis: str = "close") -> pd.DataFrame:
    """
    ICT-style 3-candle FVG detector.
    df needs columns: open, high, low, close
    min_gap_pct: minimum gap height (%) to be considered valid
    basis: 'close' or 'open' or 'midgap'
    """

    high_1 = df["high"].shift(1)   # i-1
    low_1  = df["low"].shift(1)
    high_p1 = df["high"].shift(-1) # i+1
    low_p1  = df["low"].shift(-1)

    # Bullish FVG: low[i+1] > high[i-1]
    bull = low_p1 > high_1
    bull_low = high_1
    bull_high = low_p1
    bull_size = (bull_high - bull_low).clip(lower=0)

    # Bearish FVG: high[i+1] < low[i-1]
    bear = high_p1 < low_1
    bear_low = high_p1
    bear_high = low_1
    bear_size = (bear_high - bear_low).clip(lower=0)

    # choose basis price
    if basis == "close":
        base = df["close"]
    elif basis == "open":
        base = df["open"]
    elif basis == "midgap":
        # mid of whichever gap exists; fallback to close to avoid div0
        mid_bull = (bull_low + bull_high) / 2
        mid_bear = (bear_low + bear_high) / 2
        base = np.where(bull, mid_bull, np.where(bear, mid_bear, df["close"]))
        base = pd.Series(base, index=df.index)
    else:
        raise ValueError("basis must be one of: close, open, midgap")

    # gap pct
    bull_pct = (bull_size / base) * 100
    bear_pct = (bear_size / base) * 100

    # validity threshold
    bull_valid = bull & (bull_pct >= min_gap_pct)
    bear_valid = bear & (bear_pct >= min_gap_pct)

    out = df.copy()
    out["fvg_bull"] = bull_valid.astype(int)
    out["fvg_bear"] = bear_valid.astype(int)

    # store zone bounds (NaN if not valid)
    out["fvg_low"]  = np.nan
    out["fvg_high"] = np.nan
    out.loc[bull_valid, "fvg_low"] = bull_low[bull_valid]
    out.loc[bull_valid, "fvg_high"] = bull_high[bull_valid]
    out.loc[bear_valid, "fvg_low"] = bear_low[bear_valid]
    out.loc[bear_valid, "fvg_high"] = bear_high[bear_valid]
    
    # 判斷最高 / 最低點
    fvg_high_limit = np.where(bull_valid, high_p1, np.where(bear_valid, high_1, np.nan))
    fvg_low_limit = np.where(bear_valid, low_p1, np.where(bull_valid, low_1, np.nan))

    out["fvg_size"] = np.nan
    out.loc[bull_valid, "fvg_size"] = bull_size[bull_valid]
    out.loc[bear_valid, "fvg_size"] = bear_size[bear_valid]

    out["fvg_pct"] = np.nan
    out.loc[bull_valid, "fvg_pct"] = bull_pct[bull_valid]
    out.loc[bear_valid, "fvg_pct"] = bear_pct[bear_valid]
    
    # 儲存邊界以便進場判斷
    out["fvg_up_limit"] = fvg_high_limit
    out["fvg_down_limit"] = fvg_low_limit

    return out