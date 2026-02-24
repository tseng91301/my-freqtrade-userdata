docker compose run --rm freqtrade backtesting --strategy MyStrategy --timerange 20260201-20260202 --config user_data/config.json 
docker compose run --rm freqtrade plot-dataframe --strategy MyStrategy --timerange 20260201-20260202
sudo chown -R tseng:tseng ~/ft_userdata/*