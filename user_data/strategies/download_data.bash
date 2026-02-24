#!/bin/bash

# --- 設定區 ---
# 在這裡填寫你想要下載的幣種 (以空白分隔)
PAIRS=("BTC/USDT:USDT" "ETH/USDT:USDT" "SOL/USDT:USDT" "XRP/USDT:USDT" "BNB/USDT:USDT" "DOGE/USDT:USDT" "SUI/USDT:USDT" "XAUT/USDT:USDT")

# 在這裡設定時間範圍 (格式: YYYYMMDD-)
TIMERANGE="20260101-20260131"

# 指定交易所 (預設 binance)
EXCHANGE="binance"

# 指定時區清單
TIMEFRAMES=("1m" "5m" "15m" "1h" "4h" "1d")

# --- 執行區 ---

# 將陣列轉換為 Freqtrade 要求的空白分隔字串
PAIRS_STR="${PAIRS[*]}"

echo "-------------------------------------------------------"
echo "開始下載資料..."
echo "幣種: ${PAIRS_STR}"
echo "交易所: ${EXCHANGE}"
echo "時間範圍: ${TIMERANGE}"
echo "時區: ${TIMEFRAMES[*]}"
echo "-------------------------------------------------------"

# 遍歷所有時區進行下載
for TF in "${TIMEFRAMES[@]}"
do
    echo "[+] 正在下載 ${TF} 的資料..."
    
    # 執行 Freqtrade 下載指令
    # 如果你是使用 Docker，請將下面這行改成:
    # docker-compose run --rm freqtrade download-data --pairs ${PAIRS_STR} --exchange ${EXCHANGE} -t ${TF} --timerange ${TIMERANGE}
    
    docker compose run --rm freqtrade download-data \
        --pairs ${PAIRS_STR} \
        --exchange ${EXCHANGE} \
        -t ${TF} \
        --timerange ${TIMERANGE}
    
    echo "[OK] ${TF} 下載完成。"
done

echo "-------------------------------------------------------"
echo "所有任務已完成！資料儲存於 user_data/data/${EXCHANGE} 中。"
echo "-------------------------------------------------------"

sudo chown -R tseng:tseng ~/ft_userdata/*