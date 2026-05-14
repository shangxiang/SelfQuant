import os
import pandas as pd
from typing import Optional


class DataLoader:
    def __init__(self, config):
        self.cfg = config
        self.cache = {}
        # 在初始化时读取一次股票列表并缓存，避免每次 get_data 都读盘
        self.stock_df = pd.read_csv(config.stock_list_file)

        cal = pd.read_csv(config.calendar_file, dtype={'cal_date': str})
        all_dates = sorted(cal['cal_date'].dropna().str.strip().tolist())
        self.all_dates = [d for d in all_dates if d >= '20200101']

    def get_data(self, date_str: str) -> Optional[pd.DataFrame]:
        if date_str in self.cache:
            return self.cache[date_str]
        path = os.path.join(self.cfg.data_dir, f"{date_str}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        # 合并股票名称用于过滤 ST（name_y 来自 stock_list，section CSV 中已有 name_x）
        df = df.merge(self.stock_df[["ts_code", "name"]], on="ts_code", how='left')
        df = df[~df['name_y'].str.contains('ST', na=False)]
        self.cache[date_str] = df
        return df

    def get_trading_dates(self) -> list:
        return self.all_dates
