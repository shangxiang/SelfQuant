import os
import pandas as pd
from typing import Optional


class DataLoader:
    """
    截面数据加载器。

    负责按交易日读取 data/section/ 目录下的截面 CSV，过滤 ST 股，并缓存结果。
    stock_df 和交易日历在初始化时一次性读入，避免在高频调用 get_data() 时重复 I/O。
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : 任意带以下属性的配置对象
            stock_list_file : str  股票列表 CSV，需含 ts_code、name 列
            calendar_file   : str  交易日历 CSV，需含 cal_date 列（YYYYMMDD 字符串）
            data_dir        : str  截面数据目录，文件名为 YYYYMMDD.csv
        """
        self.cfg = config

        # key: date_str → DataFrame，避免同一日期重复读盘
        self.cache: dict = {}

        # 股票列表只读一次，后续 get_data 内合并时直接复用
        self.stock_df = pd.read_csv(config.stock_list_file)

        # 读取交易日历并转为有序字符串列表，只保留 2020 年以后（更早的截面数据不完整）
        cal = pd.read_csv(config.calendar_file, dtype={'cal_date': str})
        all_dates = sorted(cal['cal_date'].dropna().str.strip().tolist())
        self.all_dates: list[str] = [d for d in all_dates if d >= '20200101']

    def get_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        读取指定交易日的截面数据，过滤 ST 股后返回。
        若该日期无对应文件（停市、数据缺失），返回 None。

        截面文件与 stock_list 合并后，原文件中的 name 列变为 name_x，
        stock_list 中的 name 列变为 name_y，用 name_y 做 ST 过滤。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期
        """
        if date_str in self.cache:
            return self.cache[date_str]

        path = os.path.join(self.cfg.data_dir, f"{date_str}.csv")
        if not os.path.exists(path):
            return None

        df = pd.read_csv(path)

        # 合并 stock_list 的 name 列（name_y），用于识别 ST/退市股
        # 截面 CSV 自身已有 name 列（name_x），两者都保留
        df = df.merge(self.stock_df[["ts_code", "name"]], on="ts_code", how='inner')
        df = df[~df['name_y'].str.contains('ST', na=False)]

        self.cache[date_str] = df
        return df

    def get_trading_dates(self) -> list[str]:
        """返回完整交易日历列表（YYYYMMDD 字符串，2020-01-01 起）。"""
        return self.all_dates
