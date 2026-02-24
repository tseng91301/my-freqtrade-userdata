import pandas as pd
import numpy as np

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