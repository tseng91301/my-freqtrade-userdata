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


class OrderblockStrategy(IStrategy):
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
    
    # 讓 entry/exit 可以用 limit 价
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")
    
    # === FVG 參數（你可調）===
    fvg_min_gap_pct = DecimalParameter(0.01, 2.0, default=0.1, decimals=3, space="buy")
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # 每個 pair 最新等待中的 FVG
        self._active_fvg = {}       # pair -> dict
        # 觸發進場後，等待下單 callback 使用的資料
        self._pending_orders = {}   # pair -> dict
        self._trade_plan = {}   # trade_id -> dict(tp, sl, entry, side)

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
        
        fvgdf = detect_fvg(dataframe, min_gap_pct=float(self.fvg_min_gap_pct.value), basis="close")
        dataframe["fvg_bull"] = fvgdf["fvg_bull"]
        dataframe["fvg_bear"] = fvgdf["fvg_bear"]
        dataframe["fvg_low"] = fvgdf["fvg_low"]
        dataframe["fvg_high"] = fvgdf["fvg_high"]
        dataframe["fvg_pct"] = fvgdf["fvg_pct"]

        # FVG 第一根 K（i-1）低點：對應到「fvg_bull=1 的那根(中間K)」
        dataframe["fvg_first_low"] = dataframe["low"].shift(1)
        dataframe["fvg_first_high"] = dataframe["high"].shift(1)
        # dataframe["fvg_first_low"] = dataframe["low"]
        # dataframe["fvg_first_high"] = dataframe["high"]
        
        # ========= 畫圖用：bull / bear 的 entry / tp / sl 線 =========
        df = dataframe  # alias

        # --- Bullish lines ---
        bull_id = df["fvg_bull"].fillna(0).astype(int).cumsum()

        bull_entry_raw = np.where(df["fvg_bull"] == 1, df["fvg_high"], np.nan)
        bull_sl_raw    = np.where(df["fvg_bull"] == 1, df["fvg_first_low"], np.nan)

        df["bull_entry"] = pd.Series(bull_entry_raw, index=df.index).groupby(bull_id).ffill()
        df["bull_sl"]    = pd.Series(bull_sl_raw, index=df.index).groupby(bull_id).ffill()
        df["bull_tp"]    = df["high"].groupby(bull_id).cummax()
                
        bull_valid = bull_id > 0

        # 每個 group 的第一根（FVG 創建那根）
        bull_first = bull_valid & (bull_id != bull_id.shift(1))

        # K線價格曾經到達 fvg 上緣
        bull_touched = (
            bull_valid &
            (~bull_first) &
            (df["low"] <= df["bull_entry"]) &
            (df["high"] >= df["bull_entry"])
        )
        
        bull_touched_after = bull_touched.groupby(bull_id).cummax()
        bull_prev_touched = bull_touched_after.groupby(bull_id).shift(1, fill_value=False)
        bull_show = bull_valid & (~bull_prev_touched)
        
        # 確保 bull_show 是布林格式
        mask = bull_show == True

        # 使用 where：當條件為 True 時保留原值，否則設為 NaN
        df["bull_entry"] = df["bull_entry"].where(mask, np.nan)
        df["bull_tp"] = df["bull_tp"].where(mask, np.nan)
        df["bull_sl"] = df["bull_sl"].where(mask, np.nan)

        # --- Bearish lines ---
        bear_id = df["fvg_bear"].fillna(0).astype(int).cumsum()

        bear_entry_raw = np.where(df["fvg_bear"] == 1, df["fvg_low"], np.nan)
        bear_sl_raw    = np.where(df["fvg_bear"] == 1, df["fvg_first_high"], np.nan)

        df["bear_entry"] = pd.Series(bear_entry_raw, index=df.index).groupby(bear_id).ffill()
        df["bear_sl"]    = pd.Series(bear_sl_raw, index=df.index).groupby(bear_id).ffill()
        df["bear_tp"]    = df["low"].groupby(bear_id).cummin()
        
        bear_valid = bear_id > 0

        # 每個 group 的第一根（FVG 創建那根）
        bear_first = bear_valid & (bear_id != bear_id.shift(1))

        # K線價格曾經到達 fvg 上緣
        bear_touched = (
            bear_valid &
            (~bear_first) &
            (df["low"] <= df["bear_entry"]) &
            (df["high"] >= df["bear_entry"])
        )
        bear_touched_after = bear_touched.groupby(bear_id).cummax()
        bear_prev_touched = bear_touched_after.groupby(bear_id).shift(1, fill_value=False)
        bear_show = bear_valid & (~bear_prev_touched)
        
        # 確保 bear_show 是布林格式
        mask = bear_show == True

        # 使用 where：當條件為 True 時保留原值，否則設為 NaN
        df["bear_entry"] = df["bear_entry"].where(mask, np.nan)
        df["bear_tp"] = df["bear_tp"].where(mask, np.nan)
        df["bear_sl"] = df["bear_sl"].where(mask, np.nan)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df["enter_long"] = 0
        df["enter_short"] = 0

        # 1. 建立一個「不被抹除」的進場參考線（僅限內部邏輯使用）
        # 這樣即使 bull_entry 為了畫圖變成了 NaN，這裡依然有數值可以對比
        bull_id = df["fvg_bull"].fillna(0).astype(int).cumsum()
        temp_bull_entry = np.where(df["fvg_bull"] == 1, df["fvg_high"], np.nan)
        temp_bull_entry = pd.Series(temp_bull_entry).groupby(bull_id).ffill()

        bear_id = df["fvg_bear"].fillna(0).astype(int).cumsum()
        temp_bear_entry = np.where(df["fvg_bear"] == 1, df["fvg_low"], np.nan)
        temp_bear_entry = pd.Series(temp_bear_entry).groupby(bear_id).ffill()

        # 2. 判斷長單進場
        # 使用 temp_bull_entry (未抹除版)
        touched_long = (
            (bull_id > 0) &
            (df["fvg_bull"] != 1) & 
            (temp_bull_entry.notna()) &
            (df["low"] <= temp_bull_entry) &
            (df["high"] >= temp_bull_entry)
        )
        
        # 確保只在該 FVG 的「第一次」觸碰時進場
        first_touch_long = touched_long & (~touched_long.groupby(bull_id).shift(1, fill_value=False))
        df.loc[first_touch_long, "enter_long"] = 1

        # 3. 判斷短單進場
        touched_short = (
            (bear_id > 0) &
            (df["fvg_bear"] != 1) &
            (temp_bear_entry.notna()) &
            (df["low"] <= temp_bear_entry) &
            (df["high"] >= temp_bear_entry)
        )
        
        first_touch_short = touched_short & (~touched_short.groupby(bear_id).shift(1, fill_value=False))
        df.loc[first_touch_short, "enter_short"] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df["exit_long"] = 0
        df["exit_short"] = 0

        # 必須使用與 indicators 相同的 ID 邏輯
        bull_id = df["fvg_bull"].fillna(0).astype(int).cumsum()
        bear_id = df["fvg_bear"].fillna(0).astype(int).cumsum()

        # ===== Long Exit =====
        # 只有在進場那根才鎖定 TP/SL
        long_tp = np.where(df["enter_long"] == 1, df["bull_tp"], np.nan)
        long_sl = np.where(df["enter_long"] == 1, df["bull_sl"], np.nan)
        
        # 關鍵：ffill 必須限制在同一個 ID 內！否則會跨 FVG 污染
        df["long_tp_fixed"] = pd.Series(long_tp).groupby(bull_id).ffill()
        df["long_sl_fixed"] = pd.Series(long_sl).groupby(bull_id).ffill()

        # 判斷是否觸及（且確保當前有鎖定的數值）
        hit_tp_long = (df["long_tp_fixed"].notna()) & (df["high"] >= df["long_tp_fixed"])
        hit_sl_long = (df["long_sl_fixed"].notna()) & (df["low"] <= df["long_sl_fixed"])
        
        # 為了避免每一根都出現 Exit，我們只在「第一次觸及」時發出訊號
        df.loc[hit_tp_long | hit_sl_long, "exit_long"] = 1

        # ===== Short Exit =====
        short_tp = np.where(df["enter_short"] == 1, df["bear_tp"], np.nan)
        short_sl = np.where(df["enter_short"] == 1, df["bear_sl"], np.nan)
        
        df["short_tp_fixed"] = pd.Series(short_tp).groupby(bear_id).ffill()
        df["short_sl_fixed"] = pd.Series(short_sl).groupby(bear_id).ffill()

        hit_tp_short = (df["short_tp_fixed"].notna()) & (df["low"] <= df["short_tp_fixed"])
        hit_sl_short = (df["short_sl_fixed"].notna()) & (df["high"] >= df["short_sl_fixed"])
        
        df.loc[hit_tp_short | hit_sl_short, "exit_short"] = 1

        return df
    
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                       entry_tag: Optional[str], side: str, **kwargs) -> float:
        p = self._pending_orders.get(pair)
        if p and p.get("side") == side and "entry_price" in p:
            return float(p["entry_price"])
        return proposed_rate

    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        plan = self._trade_plan.get(trade.id)
        if plan and "sl_price" in plan:
            sl_abs = float(plan["sl_price"])
            return stoploss_from_absolute(trade.open_rate, sl_abs, is_short=trade.is_short)

        return self.stoploss


    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                        time_in_force: str, current_time: datetime, entry_tag: str,
                        side: str, **kwargs) -> bool:
        if side != "long":
            return True

        p = self._pending_orders.get(pair)
        if not p:
            return True

        # 進場確認後，把資料塞給 trade.user_data（後續 stoploss/exit 用）
        # 注意：confirm_trade_entry 沒有 trade 物件，因此這裡只回 True；
        # 真正寫入 trade.user_data 我們改用 `custom_trade_info` 更穩（見下）
        return True
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                current_rate: float, current_profit: float, **kwargs) -> Optional[str]:

        # 若這筆 trade 還沒綁定 plan，嘗試用 pending_orders 綁一次（只會成功一次）
        if trade.id not in self._trade_plan:
            p = self._pending_orders.get(pair)
            # pending 可能已經被下一筆覆蓋，所以做個基本檢查：side 要一致
            if p and p.get("side") == ("short" if trade.is_short else "long"):
                self._trade_plan[trade.id] = {
                    "tp_price": float(p["tp_price"]),
                    "sl_price": float(p["sl_price"]),
                    "entry_price": float(p["entry_price"]),
                    "side": p.get("side"),
                    "created_idx": str(p.get("created_idx")),
                }
                # 綁定成功就清掉 pending，避免污染下一筆
                self._pending_orders.pop(pair, None)

        plan = self._trade_plan.get(trade.id)
        if not plan:
            return None

        tp = float(plan["tp_price"])

        if self.dp:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df) > 0:
                last = df.iloc[-1]
                if not trade.is_short:
                    if float(last["high"]) >= tp:
                        return "fvg_tp"
                else:
                    if float(last["low"]) <= tp:
                        return "fvg_tp"

        return None


    def custom_exit_price(self, pair: str, trade: Trade, current_time: datetime,
                      proposed_rate: float, exit_reason: Optional[str] = None, **kwargs) -> float:
        # 有些版本不會傳 exit_reason，就從 kwargs 試著抓
        if exit_reason is None:
            exit_reason = kwargs.get("reason") or kwargs.get("exit_reason")

        if exit_reason == "fvg_tp":
            plan = self._trade_plan.get(trade.id)
            if plan and "tp_price" in plan:
                return float(plan["tp_price"])

        return proposed_rate

    plot_config = {
        "main_plot": {
            "fvg_low":  {"plotly": {"mode": "lines", "line": {"width": 1, "dash": "dot"}}},
            "fvg_high": {"plotly": {"mode": "lines", "line": {"width": 1, "dash": "dot"}}},

            # Bull (long) lines
            "bull_entry": {"plotly": {"mode": "lines", "line": {"width": 2}}},
            "bull_tp":    {"plotly": {"mode": "lines", "line": {"width": 1, "dash": "dash"}}},
            "bull_sl":    {"plotly": {"mode": "lines", "line": {"width": 1, "dash": "dot"}}},

            # Bear (short) lines
            "bear_entry": {"plotly": {"mode": "lines", "line": {"width": 2}}},
            "bear_tp":    {"plotly": {"mode": "lines", "line": {"width": 1, "dash": "dash"}}},
            "bear_sl":    {"plotly": {"mode": "lines", "line": {"width": 1, "dash": "dot"}}},
        },
        "subplots": {
            "FVG": {
                "fvg_pct":  {"plotly": {"mode": "lines"}},
                "fvg_bull": {"plotly": {"mode": "lines"}},
                "fvg_bear": {"plotly": {"mode": "lines"}},
            }
        }
    }