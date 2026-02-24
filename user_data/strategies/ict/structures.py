import numpy as np
import pandas as pd
from pandas import DataFrame

def populate_market_structure(dataframe: DataFrame) -> DataFrame:
    # 初始化新增欄位為 NaN
    dataframe['bull_bos'] = np.nan
    dataframe['bear_bos'] = np.nan
    dataframe['bull_mss'] = np.nan
    dataframe['bear_mss'] = np.nan
    dataframe['bull_bos_trigger'] = False
    dataframe['bear_bos_trigger'] = False
    dataframe['bull_mss_trigger'] = False
    dataframe['bear_mss_trigger'] = False
    dataframe['trend'] = np.nan
    dataframe['prev_low'] = np.nan
    dataframe['prev_high'] = np.nan
    dataframe['support'] = np.nan
    dataframe['resistance'] = np.nan

    # 狀態變數
    # trend: 1 為看漲, -1 為看跌
    # high_price, low_price: 當前結構的最高/最低點
    # has_retraced: 是否發生過回調 (實體反向)
    
    # 初始化第一根 K 線
    first_open = dataframe.iloc[0]['open']
    first_close = dataframe.iloc[0]['close']
    current_trend = 1 if first_close >= first_open else -1
    
    dataframe.at[0, 'trend'] = current_trend
    prev_high = dataframe.iloc[0]['high']
    prev_high_index = 0
    dataframe.at[0, 'prev_high'] = prev_high
    dataframe.at[0, 'resistance'] = prev_high
    prev_low = dataframe.iloc[0]['low']
    prev_low_index = 0
    dataframe.at[0, 'prev_low'] = prev_low
    dataframe.at[0, 'support'] = prev_low
    curr_high = dataframe.iloc[0]['high']
    curr_high_index = 0
    curr_low = dataframe.iloc[0]['low']
    curr_low_index = 0
    has_retraced = False

    # 為了效能，我們使用 .itertuples() 或直接迭代索引
    for i in range(1, len(dataframe)):
        row = dataframe.iloc[i]
        close = row['close']
        open = row['open']
        high = row['high']
        low = row['low']

        if current_trend == 1:  # --- 看漲結構 ---
            # 1. 優先判斷 MSS (跌破前低 -> 轉看跌)
            if close < prev_low:
                # 標註從前低點到當前突破點的橫線
                dataframe.loc[prev_low_index:i, 'bear_mss'] = prev_low
                current_trend = -1
                prev_high = curr_high
                prev_high_index = curr_high_index
                curr_high, curr_high_index = high, i
                curr_low, curr_low_index = low, i
                has_retraced = False
                dataframe.at[i, 'trend'] = current_trend
                dataframe.at[i, 'prev_high'] = prev_high
                dataframe.at[i, 'prev_low'] = prev_low
                dataframe.at[i, 'resistance'] = prev_high
                dataframe.at[i, 'bear_mss_trigger'] = True
                continue
            
            dataframe.at[i, 'trend'] = current_trend
            # 2. 判斷回調
            if close < open:
                has_retraced = True

            # 3. 判斷 BOS (突破新高 -> 結構延續)
            if has_retraced and close > curr_high:
                dataframe.at[curr_high_index:i, 'bull_bos'] = curr_high
                prev_low = curr_low
                prev_low_index = curr_low_index
                dataframe.at[i, 'prev_low'] = prev_low
                dataframe.at[i, 'support'] = prev_low
                curr_low, curr_low_index = close, i
                curr_high, curr_high_index = high, i
                dataframe.at[i, 'prev_high'] = curr_high
                dataframe.at[i, 'bull_bos_trigger'] = True
                has_retraced = False
            else:
                # 4. 更新當前波段極端值
                if high > curr_high:
                    curr_high, curr_high_index = high, i
                    dataframe.at[i, 'prev_high'] = curr_high
                else:
                    dataframe.at[i, 'prev_high'] = dataframe.at[i - 1, 'prev_high']
                if low < curr_low:
                    curr_low, curr_low_index = low, i
                    dataframe.at[i, 'prev_low'] = curr_low
                else:
                    dataframe.at[i, 'prev_low'] = dataframe.at[i - 1, 'prev_low']
                dataframe.at[i, 'support'] = dataframe.at[i - 1, 'support']
                dataframe.at[i, 'resistance'] = dataframe.at[i - 1, 'resistance']

        else:  # --- 看跌結構 (current_trend == -1) ---
            # 1. 優先判斷 MSS (突破前高 -> 轉看漲)
            if close > prev_high:
                dataframe.loc[prev_high_index:i, 'bull_mss'] = prev_high
                current_trend = 1
                prev_low = curr_low
                prev_low_index = curr_low_index
                curr_high, curr_high_index = high, i
                curr_low, curr_low_index = low, i
                has_retraced = False
                dataframe.at[i, 'trend'] = current_trend
                dataframe.at[i, 'prev_high'] = prev_high
                dataframe.at[i, 'prev_low'] = prev_low
                dataframe.at[i, 'support'] = prev_low
                dataframe.at[i, 'bull_mss_trigger'] = True
                continue
            
            dataframe.at[i, 'trend'] = current_trend

            # 2. 判斷回調
            if close > open:
                has_retraced = True

            # 3. 判斷 BOS (跌破新低 -> 結構延續)
            if has_retraced and close < curr_low:
                dataframe.at[curr_low_index:i, 'bear_bos'] = curr_low
                prev_high = curr_high
                prev_high_index = curr_high_index
                dataframe.at[i, 'prev_high'] = prev_high
                dataframe.at[i, 'resistance'] = prev_high
                curr_high, curr_high_index = close, i
                curr_low, curr_low_index = low, i
                dataframe.at[i, 'prev_low'] = curr_low
                dataframe.at[i, 'bear_bos_trigger'] = True
                has_retraced = False
            else:
                # 4. 更新當前波段極端值
                if low < curr_low:
                    curr_low, curr_low_index = low, i
                    dataframe.at[i, 'prev_low'] = curr_low
                else:
                    dataframe.at[i, 'prev_low'] = dataframe.at[i - 1, 'prev_low']
                if high > curr_high:
                    curr_high, curr_high_index = high, i
                    dataframe.at[i, 'prev_high'] = curr_high
                else:
                    dataframe.at[i, 'prev_high'] = dataframe.at[i - 1, 'prev_high']
                dataframe.at[i, 'support'] = dataframe.at[i - 1, 'support']
                dataframe.at[i, 'resistance'] = dataframe.at[i - 1, 'resistance']


    return dataframe