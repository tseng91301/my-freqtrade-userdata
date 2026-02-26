import os
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from smartmoneyconcepts import smc

def load_data(symbol, timeframe, start_date, end_date):
    """加載並過濾指定範圍的資料"""
    data_dir = "../data_20240101-/binance/futures/"
    file_path = f"{data_dir}{symbol}-{timeframe}-futures.feather"
    
    if not os.path.exists(file_path):
        # 嘗試相對當前目錄的路徑
        file_path = os.path.join(os.path.dirname(__file__), data_dir, f"{symbol}-{timeframe}-futures.feather")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到資料檔案: {file_path}")

    print(f"📂 讀取資料：{file_path}")
    df = pd.read_feather(file_path)
    df.columns = df.columns.str.lower()
    
    # 確保 date 欄位是 datetime 格式
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # 處理時區問題：如果資料是時區相關的，則也將過濾器轉為時區相關
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        if df['date'].dt.tz is not None:
            start_ts = start_ts.tz_localize(df['date'].dt.tz)
            end_ts = end_ts.tz_localize(df['date'].dt.tz)
            
        # 過濾日期範圍
        mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
        df = df.loc[mask].reset_index(drop=True)
    else:
        print("⚠️ 警告：資料中沒有 'date' 欄位，無法精確過濾日期範圍。")
        
    print(f"   總計：{len(df)} 根 K 線")
    return df

def calculate_smc(df):
    """使用 smartmoneyconcepts 計算所有指標"""
    print("📊 計算 SMC 指標...")
    SWING_LENGTH = 25
    
    swing_hl = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
    fvg_data = smc.fvg(df, join_consecutive=False)
    ob_data = smc.ob(df, swing_hl, close_mitigation=False)
    bos_choch_data = smc.bos_choch(df, swing_hl, close_break=True)
    liq_data = smc.liquidity(df, swing_hl)
    
    return {
        "swing_hl": swing_hl,
        "fvg": fvg_data,
        "ob": ob_data,
        "bos_choch": bos_choch_data,
        "liq": liq_data
    }

def create_plot(df, smc_results, symbol, timeframe, output_path):
    """創建 Plotly 互動式圖表"""
    print("📈 繪製圖表...")
    
    fig = go.Figure()

    # 1. 蠟燭圖
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))

    # 2. FVG (Fair Value Gap) - 紫色長方形
    fvg = smc_results['fvg']
    for idx, row in fvg.dropna(subset=['FVG']).iterrows():
        # 如果有 MitigatedIndex (且 > 0)，繪製到該索引，否則繪製到最後
        end_idx = row['MitigatedIndex'] if pd.notna(row['MitigatedIndex']) and row['MitigatedIndex'] > 0 else len(df) - 1
        
        fig.add_shape(
            type="rect",
            x0=idx, y0=row['Bottom'], x1=end_idx, y1=row['Top'],
            fillcolor="rgba(128, 0, 128, 0.3)",
            line=dict(width=0),
            name="FVG"
        )
        # 文字標記
        fig.add_annotation(
            x=idx, y=(row['Top'] + row['Bottom']) / 2,
            text="FVG", showarrow=False, font=dict(color="purple", size=8)
        )

    # 3. OB (Order Block) - 透明藍色長方形
    ob = smc_results['ob']
    for idx, row in ob.dropna(subset=['OB']).iterrows():
        end_idx = row['MitigatedIndex'] if pd.notna(row['MitigatedIndex']) and row['MitigatedIndex'] > 0 else len(df) - 1
        
        fig.add_shape(
            type="rect",
            x0=idx, y0=row['Bottom'], x1=end_idx, y1=row['Top'],
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(width=0),
            name="OB"
        )
        fig.add_annotation(
            x=idx, y=(row['Top'] + row['Bottom']) / 2,
            text="OB", showarrow=False, font=dict(color="blue", size=8)
        )

    # 4. BOS & CHoCH - 藍色水平線
    bc = smc_results['bos_choch']
    for idx, row in bc.dropna(subset=['Level']).iterrows():
        start_idx = idx
        end_idx = row['BrokenIndex'] if pd.notna(row['BrokenIndex']) else len(df) - 1
        level = row['Level']
        
        label_text = "BOS" if pd.notna(row['BOS']) else "CHoCH"
        
        fig.add_trace(go.Scatter(
            x=[start_idx, end_idx],
            y=[level, level],
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name=label_text,
            showlegend=False
        ))
        
        fig.add_annotation(
            x=start_idx, y=level,
            text=label_text, showarrow=True, arrowhead=1,
            font=dict(color="blue", size=10),
            ax=0, ay=-20
        )

    # 5. Liquidity - 黑色水平線
    liq = smc_results['liq']
    for idx, row in liq.dropna(subset=['Liquidity']).iterrows():
        start_idx = idx
        end_idx = row['End'] if pd.notna(row['End']) else len(df) - 1
        level = row['Level']
        
        fig.add_trace(go.Scatter(
            x=[start_idx, end_idx],
            y=[level, level],
            mode='lines',
            line=dict(color='black', width=1),
            name='Liq',
            showlegend=False
        ))
        
        fig.add_annotation(
            x=start_idx, y=level,
            text="Liq", showarrow=False, font=dict(color="black", size=8)
        )

    # 圖表佈局
    fig.update_layout(
        title=f"SMC Analysis: {symbol} ({timeframe})",
        xaxis_title="Time Index",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=800,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # 儲存
    fig.write_html(output_path)
    print(f"✅ 圖表已儲存：{output_path}")

def main():
    parser = argparse.ArgumentParser(description="SMC 互動式視覺化工具")
    parser.add_argument("--symbol", type=str, default="BTC_USDT_USDT", help="幣種名稱 (例如 BTC_USDT_USDT)")
    parser.add_argument("--timeframe", type=str, default="4h", help="時間框架 (例如 15m, 1h, 4h, 1d)")
    parser.add_argument("--start", type=str, default="2024-01-1", help="開始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-30", help="結束日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    output_dir = "output_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 載入資料
    try:
        df = load_data(args.symbol, args.timeframe, args.start, args.end)
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        return

    if df.empty:
        print("❌ 錯誤：指定的範圍內沒有資料。")
        return

    # 計算 SMC
    smc_results = calculate_smc(df)
    
    # 產出檔案路徑
    # 格式: [日期時間範圍]-幣種-時間框架.html
    time_range = f"{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    filename = f"{time_range}-{args.symbol}-{args.timeframe}.html"
    output_path = os.path.join(output_dir, filename)
    
    # 繪圖
    create_plot(df, smc_results, args.symbol, args.timeframe, output_path)

if __name__ == "__main__":
    main()
