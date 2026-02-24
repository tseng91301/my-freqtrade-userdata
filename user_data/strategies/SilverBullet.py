# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
    AnnotationType,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib

from ict.gap import detect_fvg


class SilverBullet(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }
    minimal_roi = {"0": 100} # 停用預設 ROI

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
             proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
             **kwargs) -> float:
        """
        自定義槓桿函數。
        """
        # 你可以直接回傳固定的槓桿倍數，例如 5 倍
        return 50.0

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    use_custom_stoploss = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                "entry_price": {"color": "blue", "style": "scatter"},
                "target_price_long": {"color": "#00ff00", "style": "line"},
                "target_price_short": {"color": "#00ff00", "style": "line"},
                "stoploss_price_long": {"color": "#ff0000", "style": "line"},
                "stoploss_price_short": {"color": "#ff0000", "style": "line"},
            },
            "subplots": {
                "Valid Time": {
                    "in_time_window": {"color": "green"},
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        # 這會告訴機器人抓取當前交易對的 15m 數據
        pairs = self.dp.current_whitelist()
        print("pairs: ", pairs)
        return [(pair, '15m') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # --- 時區處理 ---
        # 判斷是否已有時區資訊，並轉為紐約時間
        if dataframe['date'].dt.tz is None:
            ny_time = dataframe['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            ny_time = dataframe['date'].dt.tz_convert('America/New_York')

        ny_hour = ny_time.dt.hour

        # --- 建立 in_time_window 指標 ---
        # 使用邏輯判斷產生 0 或 1
        dataframe['in_time_window'] = (
            ((ny_hour >= 3) & (ny_hour < 4)) |   # London
            ((ny_hour >= 10) & (ny_hour < 11)) | # NY AM
            ((ny_hour >= 14) & (ny_hour < 15))   # NY PM
        ).astype(int)
        
        # 獲取 15m 數據
        inf_tf = '15m'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        # 定義「前高/前低」的範圍，例如找過去 20 根 15m K 棒的最高點
        # lookback 窗口可以根據你的 Silver Bullet 偏好調整
        informative['inf_high'] = informative['high'].rolling(window=20).max()
        informative['inf_low'] = informative['low'].rolling(window=20).min()

        # 合併到 5m 主表
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        
        # 呼叫你之前的 FVG 偵測
        dataframe = detect_fvg(dataframe, min_gap_pct=0.05)

        # --- 關鍵邏輯：計算目標價格 ---
        # 進場位
        dataframe['entry_price'] = np.nan
        # 計算 FVG 的高度 (Gap Height)
        fvg_height = dataframe['fvg_high'] - dataframe['fvg_low']

        # 多單進場位：fvg_low + 0.3 * 高度 (靠近缺口下沿)
        dataframe.loc[dataframe['fvg_bull'] == 1, 'entry_price'] = \
            dataframe['fvg_low'] + 0.6 * fvg_height

        # 空單進場位：fvg_high - 0.3 * 高度 (靠近缺口上沿)
        dataframe.loc[dataframe['fvg_bear'] == 1, 'entry_price'] = \
            dataframe['fvg_high'] - 0.6 * fvg_height

        # 多單止盈位 (Exit Long)：找 15m 前高，如果前高比 FVG 高點還低，就用 FVG 高點
        dataframe['target_price_long'] = np.where(
            dataframe['inf_high_15m'] > dataframe['fvg_up_limit'],
            dataframe['inf_high_15m'],
            dataframe['fvg_up_limit']
        )
        dataframe['target_price_long'] = np.where(
            dataframe['in_time_window'] == 1,
            dataframe['target_price_long'],
            np.nan
        )
        
        # 多單止損位 (SL)：基於 RR 1.5
        # Risk = (TP - Entry) / 1.5 -> SL = Entry - Risk
        dataframe['reward_long'] = (dataframe['target_price_long'] - dataframe['entry_price']).clip(lower=0)
        dataframe['stoploss_price_long'] = dataframe['entry_price'] - (dataframe['reward_long'] / 1.5)
        
        # 空單止盈位 (Exit Short)：找 15m 前低，如果前低比 FVG 低點還高，就用 FVG 低點
        dataframe['target_price_short'] = np.where(
            dataframe['inf_low_15m'] < dataframe['fvg_down_limit'],
            dataframe['inf_low_15m'],
            dataframe['fvg_down_limit']
        )
        dataframe['target_price_short'] = np.where(
            dataframe['in_time_window'] == 1,
            dataframe['target_price_short'],
            np.nan
        )
        
        # 多單止損位：計算盈虧比 1.5 的止損位
        # 空單止損位 (SL)：基於 RR 1.5
        # Risk = (Entry - TP) / 1.5 -> SL = Entry + Risk
        dataframe['reward_short'] = (dataframe['entry_price'] - dataframe['target_price_short']).clip(lower=0)
        dataframe['stoploss_price_short'] = dataframe['entry_price'] + (dataframe['reward_short'] / 1.5)
        
        dataframe['target_price_long'] = dataframe['target_price_long'].ffill()
        dataframe['target_price_short'] = dataframe['target_price_short'].ffill()
        dataframe['stoploss_price_long'] = dataframe['stoploss_price_long'].ffill()
        dataframe['stoploss_price_short'] = dataframe['stoploss_price_short'].ffill()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. 時區處理 (已修正 TypeError 問題)
        if dataframe['date'].dt.tz is None:
            ny_time = dataframe['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            ny_time = dataframe['date'].dt.tz_convert('America/New_York')

        ny_hour = ny_time.dt.hour

        # 2. 定義 Silver Bullet 三個交易時段 (UTC-5)
        in_time_window = (
            ((ny_hour >= 3) & (ny_hour < 4)) |   # London Open
            ((ny_hour >= 10) & (ny_hour < 11)) | # NY AM Session
            ((ny_hour >= 14) & (ny_hour < 15))   # NY PM Session
        )

        # 3. 多單進場條件
        dataframe.loc[
            (
                in_time_window &
                (dataframe['fvg_bull'] == 1) &           # 出現看漲 FVG
                (dataframe['volume'] > 0)                # 確保有成交量
            ),
            'enter_long'] = 1

        # 4. 空單進場條件
        dataframe.loc[
            (
                in_time_window &
                (dataframe['fvg_bear'] == 1) &           # 出現看跌 FVG
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                       entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # 讀取我們在 populate_indicators 算好的 entry_price
        if side == "long" and last_candle['entry_price'] > 0:
            return last_candle['entry_price']
        elif side == "short" and last_candle['entry_price'] > 0:
            return last_candle['entry_price']
        
        return proposed_rate # 如果沒抓到則使用建議價格
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                current_profit: float, **kwargs) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # 多單止盈判斷
        if trade.is_short is False:
            if last_candle['target_price_long'] > 0 and current_rate >= last_candle['target_price_long']:
                return "target_hit_long"

        # 空單止盈判斷
        if trade.is_short is True:
            if last_candle['target_price_short'] > 0 and current_rate <= last_candle['target_price_short']:
                return "target_hit_short"

        return None
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # 多單止損：計算當前價格相對於進場價的虧損比例
        if trade.is_short is False:
            if last_candle['stoploss_price_long'] > 0:
                # 返回的是相對於當前價格的負值百分比 (例如 -0.02 代表在當前價再跌 2% 止損)
                # 這裡我們直接計算目標止損價與進場價的距離
                sl_relative = (last_candle['stoploss_price_long'] - current_rate) / current_rate
                return sl_relative

        # 空單止損
        if trade.is_short is True:
            if last_candle['stoploss_price_short'] > 0:
                sl_relative = (current_rate - last_candle['stoploss_price_short']) / current_rate
                return sl_relative

        return -0.99 # 預設一個極大的止損值，避免被預設規則誤殺
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                        proposed_stake: float, min_stake: Optional[float], max_stake: float,
                        entry_tag: Optional[str], side: str, **kwargs) -> float:
    
        # 1. 設定你「每一筆交易」願意承擔的最大損失金額 (例如 100 USDT)
        risk_amount_per_trade = 10 

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # 2. 取得我們算好的進場價與止損價
        if side == "long":
            entry_price = last_candle['entry_price']
            sl_price = last_candle['stoploss_price_long']
        else:
            entry_price = last_candle['entry_price']
            sl_price = last_candle['stoploss_price_short']

        # 3. 如果價格異常，回傳預設值
        if entry_price == 0 or sl_price == 0 or entry_price == sl_price:
            return proposed_stake

        # 4. 計算止損百分比 (距離)
        # 例如：進場 100，止損 95，則 sl_pct = 0.05 (5%)
        sl_pct = abs(entry_price - sl_price) / entry_price

        # 5. 計算應該下單的總金額
        # 公式：下單金額 = 風險金額 / 止損百分比
        # 例如：100 / 0.05 = 2000 USDT (這樣賠 5% 正好是 100)
        new_stake = risk_amount_per_trade / sl_pct

        # 6. 安全檢查：確保不超過錢包可用餘額或交易所限制
        if new_stake > max_stake:
            return max_stake
        
        return new_stake
    
    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order', 
                        current_time: datetime, **kwargs) -> bool:
        # 取得紐約時間
        ny_time = current_time.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-5)))
        ny_hour = ny_time.hour

        # 如果當前時間已經不在 3, 10, 14 點 (代表窗口已過)
        # 且這筆單還沒成交 (is_open 為 True)
        if ny_hour not in [3, 10, 14]:
            return True  # 回傳 True 代表「立刻取消這筆掛單」
        
        return False  # 窗口內則繼續等待