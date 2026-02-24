# 文件位置： ~/ft_bot/strategies/FvgTimeStrategy.py

from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

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
)

class Rectangle:
    # Static Method
    _rectangle_num = 0
    _rectangle_arr = []
    
    x_start: int
    x_end: int
    y_start: float
    y_end: float
    id: int
    __Name: str
    __Name_y_bottom: str
    __Name_y_top: str
    showName: bool = False
    fill_color = "rgba(0, 176, 246, 0.2)"
    border_color = "rgba(0, 176, 246, 1)"

    def __init__(self, xs: int, xe: int, ys: float, ye: float):
        """
        X: From left to right
        Y: From top to bottom
        """
        
        self.x_start = xs
        self.x_end = xe
        self.y_start = ys
        self.y_end = ye

        # Assign ID
        self.id = Rectangle._rectangle_num
        self.setName(str(self.id))
        Rectangle._rectangle_num += 1

        # Store object
        Rectangle._rectangle_arr.append(self)

    @staticmethod
    def get(id: int):
        """
        用全域 ID 取得 Rectangle 物件
        """
        if id < 0 or id >= Rectangle._rectangle_num:
            raise IndexError("Rectangle index out of range.")
        return Rectangle._rectangle_arr[id]

    @staticmethod
    def reset():
        """
        清空所有紀錄，常用於下次 Backtest 前重置
        """
        Rectangle._rectangle_arr.clear()
        Rectangle._rectangle_num = 0

    def export_dataframe(self, max_size: int):
        """
        回傳上下邊界各一個 pandas.Series，只有 rectangle 區間內有值，其餘 NaN。
        """
        s_top = pd.Series([np.nan] * max_size)
        s_bottom = pd.Series([np.nan] * max_size)

        xs = max(0, self.x_start)
        xe = min(max_size - 1, self.x_end)

        s_top.loc[xs:xe] = self.y_start
        s_bottom.loc[xs:xe] = self.y_end

        return s_top, s_bottom
    
    def setName(self, name: str):
        self.__Name = name
        self.__Name_y_top = f"{name}_top"
        self.__Name_y_bottom = f"{name}_bottom"
    
    def get_name(self):
        return self.__Name
    
    def get_y_names(self):
        return self.__Name_y_top, self.__Name_y_bottom    
    
    @staticmethod    
    def extend_plot_config(original_config: dict):
        if "main_plot" not in original_config:
            original_config["main_plot"] = {}

        for r in Rectangle._rectangle_arr:
            name_y_top, name_y_bottom = r.get_y_names()

            # 上邊界線
            original_config["main_plot"][name_y_top] = {
                "color": r.border_color,
                "plotly": {
                    "connectgaps": False,
                    "showlegend": False,
                    "name": "",           # 額外保險
                },
            }

            # 下邊界線 + 填色
            cfg = {
                "color": r.border_color,
                "fill_to": name_y_top,
                "fill_color": r.fill_color,
                "fill_label": "",
                "plotly": {
                    "connectgaps": False,
                    "showlegend": False,  # 關掉 legend
                    "name": "",           # 不給它名字
                },
            }

            if r.showName:
                # 注意：要改的是 plotly 裡的 showlegend
                cfg["plotly"]["showlegend"] = True
                cfg["plotly"]["name"] = r.get_name()   # 顯示你自己的名稱

            original_config["main_plot"][name_y_bottom] = cfg

        return original_config
        
def get_segments(s: pd.Series):
    """
    回傳 list of dict：
    [
        { "start": idx_start, "end": idx_end, "values": Series },
        ...
    ]
    """
    result = []
    in_seg = False
    start = None

    for i, v in s.items():
        if pd.notna(v):
            if not in_seg:
                start = i
                in_seg = True
        else:
            if in_seg:
                result.append({
                    "start": start,
                    "end": i - 1,
                    "values": s.loc[start:i-1]
                })
                in_seg = False

    if in_seg:
        result.append({
            "start": start,
            "end": s.index[-1],
            "values": s.loc[start:s.index[-1]]
        })

    return result

class Structure:
    initial_status: int # 1: bull, 0: bear 第一個結構是漲 or 跌
    data = [] # 看漲結構、看跌結構的資料 [(min_index, max_index, status)]
    lines = [] # 趨勢突破 & 跌破的水平線 [(min_index, max_index, price, status)]
    df: pd.DataFrame
    def __init__(self, df: pd.DataFrame):
        # 傳入原始圖表，包含 open, high, low, close
        self.df = df
        open = df["open"]
        close = df["close"]
        
        # 初始劃定議漲跌狀況
        d = close[0] - open[0]
        if d > 0.0:
            self.initial_status = 1
        else:
            self.initial_status = 0
        pass
    
    def __calculate_structure(self):
        """
        計算上漲/下跌結構（Market Structure）：

        定義：
        - status = 1：上漲趨勢
        - status = 0：下跌趨勢

        規則（依你描述）：
        1) MSS（反轉）
        - 上漲中：若 close < 結構最低點 -> MSS，上漲 -> 下跌；新下跌結構的「最高點」沿用舊上漲結構最高點
        - 下跌中：若 close > 結構最高點 -> MSS，下跌 -> 上漲；新上漲結構的「最低點」沿用舊下跌結構最低點（對稱）

        2) 回測（have_backtrace）
        - 上漲中：出現陰K（close < open） -> 表示回測開始
        - 下跌中：出現陽K（close > open） -> 表示回測開始

        3) BOS（延續）
        - 上漲中：若已回測，且某根K close > 結構最高點 -> BOS
            -> 結構最低點更新成「回測期間的最低點」
        - 下跌中：若已回測，且某根K close < 結構最低點 -> BOS
            -> 結構最高點更新成「回測期間的最高點」

        輸出：
        - self.data  : [(start_i, end_i, trend)]  以 MSS 分段的趨勢段落
        - self.lines : [(level_i, break_i, level_price, trend_after)]
                    用來畫「被突破的結構線」（BOS/MSS都會寫入）
                    trend_after: 事件後趨勢（BOS不變，MSS會切換）
        - self.events: 更完整事件資訊（方便除錯/維護；你不需要也可以不使用）
        """

        # 容器初始化（避免重複呼叫時累積）
        self.data = []
        self.lines = []
        self.events = []

        df = self.df.reset_index(drop=True)
        n = len(df)
        if n == 0:
            return

        # 用 numpy array 讀值比較快且乾淨
        op = df["open"].to_numpy()
        cl = df["close"].to_numpy()
        hi = df["high"].to_numpy()
        lo = df["low"].to_numpy()

        UP = 1
        DOWN = 0

        status = int(self.initial_status)  # 1=up, 0=down
        start_i = 0  # MSS 分段的起點

        # 結構（已確認的 swing）高低點
        low_i = 0
        low_v = float(lo[0])
        high_i = 0
        high_v = float(hi[0])

        # 回測狀態
        have_backtrace = False
        pb_low_i, pb_low_v = None, None    # 上漲回測期間：最低點
        pb_high_i, pb_high_v = None, None  # 下跌回測期間：最高點

        for i in range(1, n):
            o = float(op[i])
            c = float(cl[i])
            h = float(hi[i])
            l = float(lo[i])

            # =========================================================
            # 上漲趨勢（UP）
            # =========================================================
            if status == UP:
                # -------- MSS：上漲 -> 下跌（收盤跌破結構低點）--------
                if c < low_v:
                    end_i = i - 1
                    if end_i >= start_i:
                        self.data.append((start_i, end_i, UP))

                    # 畫出「被跌破的結構低點」水平線
                    self.lines.append((low_i, i, low_v, DOWN))
                    self.events.append({
                        "type": "MSS",
                        "idx": i,
                        "from": UP,
                        "to": DOWN,
                        "break_level": low_v,
                        "break_level_i": low_i,
                        "carry_high": high_v,
                        "carry_high_i": high_i,
                    })

                    # 反轉：進入下跌
                    status = DOWN
                    start_i = i

                    # 依規則：新下跌結構的最高點沿用舊結構最高點（high_i/high_v 不動）
                    # 新下跌結構的最低點從當前K開始
                    low_i, low_v = i, l

                    # 重置回測狀態
                    have_backtrace = False
                    pb_low_i, pb_low_v = None, None
                    pb_high_i, pb_high_v = None, None
                    continue

                # -------- 回測：出現陰K（close < open）--------
                if (c < o) and (not have_backtrace):
                    have_backtrace = True
                    pb_low_i, pb_low_v = i, l

                # 回測期間：持續更新回測最低點（用 wick low）
                if have_backtrace:
                    if pb_low_v is None or l < pb_low_v:
                        pb_low_i, pb_low_v = i, l

                # -------- BOS：已回測 + 收盤突破結構高點（close > high_v）--------
                if have_backtrace and (c > high_v):
                    # 畫出「被突破的結構高點」水平線
                    self.lines.append((high_i, i, high_v, UP))
                    self.events.append({
                        "type": "BOS",
                        "idx": i,
                        "trend": UP,
                        "break_level": high_v,
                        "break_level_i": high_i,
                        "confirm_low": pb_low_v,
                        "confirm_low_i": pb_low_i,
                    })

                    # 依規則：BOS 後，結構最低點變成「回測期間最低點」
                    if pb_low_i is not None:
                        low_i, low_v = pb_low_i, float(pb_low_v)

                    # BOS 後進入新一段推進：重置回測狀態，並把結構高點起算點設為當前K
                    have_backtrace = False
                    pb_low_i, pb_low_v = None, None
                    high_i, high_v = i, h
                    continue

                # -------- 更新結構高點：只在「沒有回測」的推進段更新 --------
                if (not have_backtrace) and (h > high_v):
                    high_i, high_v = i, h

            # =========================================================
            # 下跌趨勢（DOWN）
            # =========================================================
            else:
                # -------- MSS：下跌 -> 上漲（收盤突破結構高點）--------
                if c > high_v:
                    end_i = i - 1
                    if end_i >= start_i:
                        self.data.append((start_i, end_i, DOWN))

                    # 畫出「被突破的結構高點」水平線
                    self.lines.append((high_i, i, high_v, UP))
                    self.events.append({
                        "type": "MSS",
                        "idx": i,
                        "from": DOWN,
                        "to": UP,
                        "break_level": high_v,
                        "break_level_i": high_i,
                        "carry_low": low_v,
                        "carry_low_i": low_i,
                    })

                    # 反轉：進入上漲
                    status = UP
                    start_i = i

                    # 依對稱規則：新上漲結構的最低點沿用舊結構最低點（low_i/low_v 不動）
                    # 新上漲結構的最高點從當前K開始
                    high_i, high_v = i, h

                    # 重置回測狀態
                    have_backtrace = False
                    pb_low_i, pb_low_v = None, None
                    pb_high_i, pb_high_v = None, None
                    continue

                # -------- 回測：出現陽K（close > open）--------
                if (c > o) and (not have_backtrace):
                    have_backtrace = True
                    pb_high_i, pb_high_v = i, h

                # 回測期間：持續更新回測最高點（用 wick high）
                if have_backtrace:
                    if pb_high_v is None or h > pb_high_v:
                        pb_high_i, pb_high_v = i, h

                # -------- BOS：已回測 + 收盤跌破結構低點（close < low_v）--------
                if have_backtrace and (c < low_v):
                    # 畫出「被突破的結構低點」水平線
                    self.lines.append((low_i, i, low_v, DOWN))
                    self.events.append({
                        "type": "BOS",
                        "idx": i,
                        "trend": DOWN,
                        "break_level": low_v,
                        "break_level_i": low_i,
                        "confirm_high": pb_high_v,
                        "confirm_high_i": pb_high_i,
                    })

                    # 依對稱規則：BOS 後，結構最高點變成「回測期間最高點」
                    if pb_high_i is not None:
                        high_i, high_v = pb_high_i, float(pb_high_v)

                    # BOS 後進入新一段下跌推進：重置回測狀態，並把結構低點起算點設為當前K
                    have_backtrace = False
                    pb_high_i, pb_high_v = None, None
                    low_i, low_v = i, l
                    continue

                # -------- 更新結構低點：只在「沒有回測」的推進段更新 --------
                if (not have_backtrace) and (l < low_v):
                    low_i, low_v = i, l

        # 收尾：把最後一段趨勢段落補進去（以 MSS 分段）
        if (n - 1) >= start_i:
            self.data.append((start_i, n - 1, status))


class LowAndHighStrategy(IStrategy):
    """
    時間區間 + FVG + 固定 R:R = 1.5 的簡化版策略骨架
    - 僅示範多空皆做的情況，你可以改成只做多或只做空
    """

    INTERFACE_VERSION = 3

    # === 基本設定 ===
    timeframe = "15m"           # 你可以改成 1m / 15m ...
    can_short = True           # 要不要做空，如果你只做多，可以改成 False

    minimal_roi = {"0": 100.0}  # 用不到 ROI（退出邏輯我們用 exit_trend 控），設一個很大的值
    stoploss = -0.99            # 真正停損位置由我們自己算 SL 判斷，這裡設超大避免干擾

    use_custom_stoploss = False  # 這裡示範用 exit_trend 做 TP/SL，不用 custom_stoploss

    startup_candle_count = 50   # 至少要一些歷史 K 線才有辦法找 local high/low

    def informative_pairs(self):
        """
        這支策略不需要額外 informative pair
        """
        return []

    # === 工具函式 1：判斷是否在指定的紐約時間區間 ===
    @staticmethod
    def _is_in_ny_session(df: DataFrame) -> pd.Series:
        """
        回傳一個 boolean Series，代表每根 K 是否在指定的紐約時間區間：
          03:00–04:00, 10:00–11:00, 14:00–15:00
        """

        # 先拿時間欄位
        if "date" in df.columns:
            dt = df["date"]
        else:
            dt = df.index.to_series()

        # 如果是 tz-naive，就先當作 UTC，再轉成紐約時間
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")

        ny_dt = dt.dt.tz_convert("America/New_York")
        hours = ny_dt.dt.hour

        # 你要的三個小時區間
        in_session = hours.isin([3, 10, 14])
        return in_session

    # === 工具函式 2：偵測 FVG，回傳 gap 上下界與中點 ===
    @staticmethod
    def _detect_fvg(df: DataFrame) -> DataFrame:
        """
        以三根 K 為一組，偵測多空 FVG
        多頭 FVG：low[i] > high[i-2]
        空頭 FVG：high[i] < low[i-2]
        在第 i 根 K 才標記 FVG（代表這個缺口「發展完」）
        """
        high = df["high"]
        low = df["low"]

        # 將 2 根之前的高/低 shift 回來
        high_2ago = high.shift(2)
        low_2ago = low.shift(2)

        bull_fvg = low > high_2ago     # 多頭 FVG
        bear_fvg = high < low_2ago     # 空頭 FVG

        # 多頭 FVG 的 gap 區間
        bull_gap_low = high_2ago
        bull_gap_high = low
        bull_mid = (bull_gap_low + bull_gap_high) / 2.0

        # 空頭 FVG 的 gap 區間
        bear_gap_high = low_2ago
        bear_gap_low = high
        bear_mid = (bear_gap_low + bear_gap_high) / 2.0

        df["bull_fvg"] = bull_fvg.astype(int)
        df["bull_gap_low"] = bull_gap_low
        df["bull_gap_high"] = bull_gap_high
        df["bull_mid"] = bull_mid

        df["bear_fvg"] = bear_fvg.astype(int)
        df["bear_gap_low"] = bear_gap_low
        df["bear_gap_high"] = bear_gap_high
        df["bear_mid"] = bear_mid

        return df

    # === 工具函式 3：找前一個 local high / low（簡化為 rolling high/low） ===
    @staticmethod
    def _prev_local_high_low(df: DataFrame, window: int = 10) -> DataFrame:
        """
        用 rolling 的方式找到前 N 根 K 的最高/最低收盤價，
        當作「前一個 local high / low」的近似。
        """
        close = df["close"]

        # shift(1) 確保是「之前」的高/低，不包含當前 K
        df["prev_local_high"] = close.shift(1).rolling(window).max()
        df["prev_local_low"] = close.shift(1).rolling(window).min()

        return df

    # === populate_indicators: 把所有會用到的東西先算好 ===
    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        
        Rectangle.reset()

        # 1. 時間區間 flag
        dataframe["in_session"] = self._is_in_ny_session(dataframe).astype(int)

        # 2. FVG
        dataframe = self._detect_fvg(dataframe)

        # 2.5 建立 FVG 區域欄位
        dataframe["bull_fvg_zone_low"] = np.where(
            dataframe["bull_fvg"] == 1,
            dataframe["bull_gap_low"],
            np.nan,
        )
        dataframe["bull_fvg_zone_high"] = np.where(
            dataframe["bull_fvg"] == 1,
            dataframe["bull_gap_high"],
            np.nan,
        )
        dataframe["bear_fvg_zone_low"] = np.where(
            dataframe["bear_fvg"] == 1,
            dataframe["bear_gap_low"],
            np.nan,
        )
        dataframe["bear_fvg_zone_high"] = np.where(
            dataframe["bear_fvg"] == 1,
            dataframe["bear_gap_high"],
            np.nan,
        )
            
        max_size = len(dataframe)
        for rect in Rectangle._rectangle_arr:
            e_h, e_l = rect.export_dataframe(max_size)
            e_h_name, e_l_name = rect.get_y_names()
            dataframe[e_h_name] = e_h
            dataframe[e_l_name] = e_l

        dataframe["min_price"] = dataframe["low"].rolling(12).min()
        dataframe["max_price"] = dataframe["high"].rolling(12).max()

        # 3. 前一個 local high / low
        dataframe = self._prev_local_high_low(dataframe, window=10)

        # 4. 用 FVG 的中點 + 前一個 local high/low 來預先計算 TP/SL
        #    這裡假設多單用 bull_mid ，空單用 bear_mid 進場
        #    RR = 1.5
        rr = 1.5

        # 多單 TP/SL
        entry_long = dataframe["bull_mid"]
        tp_long = dataframe["prev_local_high"]

        # 避免 NaN
        valid_long = entry_long.notna() & tp_long.notna() & (tp_long > entry_long)

        risk_long = tp_long - entry_long
        sl_long = entry_long - risk_long / rr

        dataframe["entry_long_price"] = np.where(valid_long, entry_long, np.nan)
        dataframe["tp_long"] = np.where(valid_long, tp_long, np.nan)
        dataframe["sl_long"] = np.where(valid_long, sl_long, np.nan)

        # 空單 TP/SL
        entry_short = dataframe["bear_mid"]
        tp_short = dataframe["prev_local_low"]

        valid_short = entry_short.notna() & tp_short.notna() & (tp_short < entry_short)

        risk_short = entry_short - tp_short
        sl_short = entry_short + risk_short / rr

        dataframe["entry_short_price"] = np.where(valid_short, entry_short, np.nan)
        dataframe["tp_short"] = np.where(valid_short, tp_short, np.nan)
        dataframe["sl_short"] = np.where(valid_short, sl_short, np.nan)

        # 5. 為了讓 exit_trend 也能讀到這些價位，我們 forward-fill
        dataframe[["entry_long_price", "tp_long", "sl_long",
                   "entry_short_price", "tp_short", "sl_short"]] = \
            dataframe[["entry_long_price", "tp_long", "sl_long",
                       "entry_short_price", "tp_short", "sl_short"]].ffill()
            
        # 將現有的長方形定義同步到 plot_config 中
        Rectangle.extend_plot_config(self.plot_config)
        
        return dataframe

    # === populate_entry_trend: 什麼時候進場 ===
    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:

        # 多單條件：
        # 1. 在指定紐約時間區間
        # 2. 當下這根 K 形成一個 bull_fvg（第三根 K）
        # 3. 有計算出 entry_long_price / tp / sl
        long_cond = (
            (dataframe["bull_fvg"] == 1) &
            dataframe["entry_long_price"].notna() &
            dataframe["tp_long"].notna() &
            dataframe["sl_long"].notna() & False
        )

        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[long_cond, "enter_tag"] = "bull_fvg_session"

        # 空單條件：
        short_cond = (
            (self.can_short) &
            (dataframe["bear_fvg"] == 1) &
            dataframe["entry_short_price"].notna() &
            dataframe["tp_short"].notna() &
            dataframe["sl_short"].notna() & False
        )

        dataframe.loc[short_cond, "enter_short"] = 1
        dataframe.loc[short_cond, "enter_tag"] = "bear_fvg_session"

        return dataframe

    # === populate_exit_trend: 價格碰到 TP 或 SL 就出場 ===
    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        close = dataframe["close"]

        # 多單：碰 TP 或跌破 SL
        long_tp_hit = close >= dataframe["tp_long"]
        long_sl_hit = close <= dataframe["sl_long"]

        long_exit = long_tp_hit | long_sl_hit

        dataframe.loc[long_exit, "exit_long"] = 1
        dataframe.loc[long_tp_hit, "exit_tag"] = "tp_long"
        dataframe.loc[long_sl_hit, "exit_tag"] = "sl_long"

        # 空單：碰 TP（往下）或漲破 SL
        short_tp_hit = close <= dataframe["tp_short"]
        short_sl_hit = close >= dataframe["sl_short"]

        short_exit = short_tp_hit | short_sl_hit

        dataframe.loc[short_exit, "exit_short"] = 1
        dataframe.loc[short_tp_hit, "exit_tag"] = "tp_short"
        dataframe.loc[short_sl_hit, "exit_tag"] = "sl_short"

        return dataframe

    plot_config = {
        "main_plot": {
            
            # FVG 區域
            "bull_fvg_zone_high": {
                "color": "orange",
            },
            "bull_fvg_zone_low": {"color": "orange"},
            "bear_fvg_zone_high": {"color": "orange"},
            "bear_fvg_zone_low": {"color": "orange"},

            # Entry / TP / SL 線
            "entry_long_price": {
                "color": "blue",
            },
            "tp_long": {"color": "green"},
            "sl_long": {"color": "red"},
            "entry_short_price": {"color": "blue"},
            "tp_short": {"color": "green"},
            "sl_short": {"color": "red"},
        },
    }
    
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }