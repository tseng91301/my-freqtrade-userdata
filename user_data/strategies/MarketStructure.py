"""
繪製結構趨勢的測試用策略

Todo: 做出一個可以劃出價格結構趨勢的程式，能夠針對價格的漲跌先畫出一個簡易的折線圖，並且從簡化過的折線圖去判斷BOS, MSS 等結構趨勢以及趨勢的 HH, HL, LH, LL

折線圖的繪製規則:
    Step 1:
        1. 使用每根 K 棒的收盤價作為繪製的參考點
        2. 價格如果是單邊行情，直接更新目標點到下一根 K 線的收盤價
        3. 如果有轉折的話，紀錄轉折點的 K 線 index 並記錄價格
    Step 2 從轉折點去找出結構趨勢:
        1.  
    Step 2 剔除無效的回調並且優化折線圖表:
        已上漲結構舉例
        1. 當價格出現轉折點時，看前一個及下一個轉折點的價格(假設沒有下一個轉折點則暫時視為有效轉折點)
        2. 回調深度 = abs(後一個轉折點 - 目前轉折點) / abs(前一個轉折點 - 目前轉折點)
        3. 設定一個最小值 (min 回調比例)，當回調深度有達到這個比例則視為有效回調，保留此轉折點，

"""

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

from ict.structures import populate_market_structure


class MarketStructure(IStrategy):
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
    timeframe = "15m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }
    minimal_roi = {"0": 100}
    
    unfilledtimeout = {
        "entry": 1, 
        "exit": 1,
        "unit": "candles"
    }
    
    # 1. 在 class 內確保掛單類型是 limit
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False
    }
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -100.0

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = None
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    use_custom_stoploss = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                # 結構趨勢線
                "zigzag": {"color": "#ff9900"},
                "bull_bos": {"color": "#00ff00", "style": "line"},
                "bear_bos": {"color": "#ff0000", "style": "line"},
                "bull_mss": {"color": "#00ffff", "style": "line"},
                "bear_mss": {"color": "#ff00ff", "style": "line"},
                "target_entry_price": {"color": "blue", "style": "scatter"},
                "target_stoploss_raw": {"color": "red", "style": "line"},
                "target_tp_raw": {"color": "green", "style": "line"},
            },
            "subplots": {
                "RSI": {"rsi": {"color": "red"}}
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
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Momentum Indicators
        # ------------------------------------
        
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ("live", "dry_run"):
                ob = self.dp.orderbook(metadata["pair"], 1)
                dataframe["best_bid"] = ob["bids"][0][0]
                dataframe["best_ask"] = ob["asks"][0][0]
        """
        dataframe = populate_market_structure(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. 計算進場相關價格 (使用 shift(1) 是因為訊號觸發在當前 K，參考的是當前的結構數值)
        # 止損點：當前的 prev_low
        dataframe['target_stoploss_raw'] = dataframe['support']
        # 止盈點：當前的 prev_high
        dataframe['target_tp_raw'] = dataframe['prev_high'].shift(1)
        
        # 計算進場掛單價：止損與止盈之間的 38% 位置 (Fibonacci 0.382 概念)
        # 公式：止損 + (止盈 - 止損) * 0.38
        dataframe['target_entry_price'] = dataframe['target_stoploss_raw'] + \
            (dataframe['target_tp_raw'] - dataframe['target_stoploss_raw']) * 0.38
            
        # 基礎條件：價格還沒跑超過 TP，且目標進場價存在
        base_cond = (
            (dataframe['volume'] > 0) &
            (dataframe['target_entry_price'].notna()) &
            (dataframe['close'] < dataframe['target_tp_raw'])
        )

        # 1. 初始進場：剛發生 BOS 的那一刻
        dataframe.loc[
            base_cond & (dataframe['bull_bos_trigger'] == True),
            ['enter_long', 'enter_tag']
        ] = (1, "smc_38_entry")

        # 2. 重新進場：BOS 已過，但價格推高導致舊單撤銷後的重掛
        # 這裡判斷 bull_bos_trigger 不是當下觸發，但結構依然有效
        dataframe.loc[
            base_cond &
            (dataframe['bull_bos_trigger'] == False) & 
            (dataframe['low'] > dataframe['target_stoploss_raw']),
            ['enter_long', 'enter_tag']
        ] = (1, "smc_38_reopen")

        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 當價格達到或超過目標止盈價時賣出
        # 注意：這裡使用 ffill() 讓目標價延續到部位關閉
        dataframe['target_tp_ffill'] = dataframe['target_tp_raw'].ffill()
        
        # dataframe.loc[
        #     (dataframe['high'] >= dataframe['target_tp_ffill']) &
        #     (dataframe['volume'] > 0) &
        #     (dataframe['enter_long'] != 1),
        #     ['exit_long', 'exit_tag']
        # ] = (1, "smc_38_profit")
        
        dataframe.loc[
            (dataframe['enter_long'] != 1),
            ['exit_long', 'exit_tag']
        ] = (1, "smc_38_profit")
        
        # dataframe.loc[
        #     (dataframe['low'] <= dataframe['target_stoploss_raw']) &
        #     (dataframe['volume'] > 0) &
        #     (dataframe['enter_long'] != 1),
        #     ['exit_long', 'exit_tag']
        # ] = (1, "smc_38_loss")
        
        return dataframe
    
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float, 
                       entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        # 只要是 SMC 相關標籤，都強制讀取最新的計算價格
        if entry_tag in ["smc_38_entry", "smc_38_reopen"]:
            if not np.isnan(last_candle['target_entry_price']):
                return last_candle['target_entry_price']
                
        return proposed_rate
    
    def custom_exit_price(self, pair: str, trade: 'Trade',
                      current_time: datetime, proposed_rate: float,
                      current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # 1. 確保 current_time 被轉換成與 dataframe 一致的格式 (通常是 UTC)
        # 2. 使用 .iloc[-1] 獲取當前這根 K 線，或者透過搜尋最接近的時間點
        # 在回測中，current_time 通常就是當前正在處理的這根 K 線的 date
        
        # 建議改用這個方式來獲取「當前最新的一根 K 線」
        current_candle = dataframe.loc[dataframe['date'] <= current_time].iloc[-1:]
        
        if not current_candle.empty:
            tp_price = current_candle.iloc[0]['target_tp_raw']
            return tp_price
                
        return proposed_rate

    # 精確止損價
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # 找到成交時的數據
        trade_candle = dataframe.loc[dataframe['date'] <= trade.open_date_utc].iloc[-1:]

        if not trade_candle.empty:
            sl_price = trade_candle.iloc[0]['target_stoploss_raw']
            if sl_price > 0:
                # 使用官方函數，第二個參數必須是 current_rate
                return stoploss_from_absolute(sl_price, current_rate, is_short=trade.is_short)
        return self.stoploss

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order', 
                        current_time: datetime, **kwargs) -> bool:
        # 獲取最新的分析數據
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        last_candle = dataframe.iloc[-1]
        
        # 獲取掛單時的目標價格（從 trade 的進場標籤或當前 dataframe 取得）
        # 假設我們在進場時已經將當時的 target_tp_raw 記錄下來
        
        # 失效條件 1：發生了新的 BOS (結構再次延續)
        if last_candle['bull_bos_trigger'] == True:
            # print(f"{pair} 發生新的 BOS，撤銷舊有的 SMC 掛單")
            return True

        # 失效條件 2：價格已經推高，導致原有的 38% 位置不再適用 (你之前提到的需求)
        if last_candle['prev_high'] > last_candle['target_tp_raw']:
            # print(f"{pair} 價格推高，結構更新，撤銷並準備重新掛單")
            return True

        # # 失效條件 3：價格已經觸及或超過原本的止盈點 (未成交先達標)
        # if last_candle['close'] >= last_candle['target_tp_raw']:
        #     print(f"{pair} 價格已達目標止盈點但未成交，訂單失效撤銷")
        #     return True
        
        return False
