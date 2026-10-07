import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd

from data_store import DataStore


class DataLoader:
    """
    截面数据加载器。

    负责按交易日读取截面数据，过滤 ST 股，并缓存结果。
    stock_df 和交易日历在初始化时一次性读入，避免在高频调用 get_data() 时重复 I/O。

    底层读取的是 DuckDB + Parquet 分区存储（data/market/trade_date=YYYYMMDD/）。
    若分区存储为空（例如尚未迁移的历史环境），自动回退到旧的 data/section/ 目录，
    因此上层策略与回测引擎无需感知存储实现的变化。
    """

    # True = 只使用主板股票（SH 6xxxxx 非688、SZ 0xxxxx 非300/301，排除北交所）
    MAIN_BOARD_ONLY: bool = True

    # 缓存上限：回测会遍历上千个交易日，无上限缓存等于把全量截面塞进内存
    CACHE_SIZE: int = 120

    def __init__(self, config):
        """
        Parameters
        ----------
        config : 任意带以下属性的配置对象
            stock_list_file : str  股票列表 CSV，需含 ts_code、name 列
            calendar_file   : str  交易日历 CSV，需含 cal_date 列（YYYYMMDD 字符串）
            data_dir        : str  旧版截面目录，仅在分区存储为空时回退使用
        """
        self.cfg = config

        # key: date_str → DataFrame，避免同一日期重复读盘（有上限，先进先出）
        self.cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

        # 股票列表只读一次，后续 get_data 内合并时直接复用
        stock_df = pd.read_csv(config.stock_list_file)
        if self.MAIN_BOARD_ONLY:
            stock_df = stock_df[stock_df['ts_code'].apply(self._is_main_board)]
        self.stock_df = stock_df.reset_index(drop=True)

        # 读取交易日历并转为有序字符串列表，只保留 2020 年以后（更早的截面数据不完整）
        cal = pd.read_csv(config.calendar_file, dtype={'cal_date': str})
        all_dates = sorted(cal['cal_date'].dropna().str.strip().tolist())
        all_dates = [d for d in all_dates if d >= '20180101']

        self._store = DataStore('data')
        # 分区存储已有数据时，把交易日历收敛到实际有截面的日期，
        # 避免引擎在大量空白日期上空转
        store_dates = set(self._store.get_all_dates())
        if store_dates:
            all_dates = [d for d in all_dates if d in store_dates]
        self.all_dates: list = all_dates

        # 行业 / 市场 beta 特征（可选）
        # standardize 把因子的行业成分剔掉了，模型看不到行业；但目标是总收益时
        # 行业轮动是真实可赚的钱，因此在这里把行业/市场信息补回特征侧。
        self.market_feat = None
        if getattr(config, 'add_market_features', False):
            from strategy.market_features import MarketFeatureBuilder
            self.market_feat = MarketFeatureBuilder(self._store)
            self.market_feat.prepare(self.all_dates)
            if self.market_feat.is_ready():
                print(f"  行业/市场 beta 特征已就绪，覆盖 {len(self.market_feat.covered_dates())} 个交易日")
        self.market_feat_zscore = getattr(config, 'market_feature_zscore', True)

        # 按 config.factor_set 解析最终因子列（此时才知道数据里实际有哪些列）
        from strategy.factor_columns import apply_factor_set
        self.n_factors = apply_factor_set(config, self._store)
        # 分区存储已有数据时，把交易日历收敛到实际有截面的日期，
        # 避免引擎在大量空白日期上空转
        store_dates = set(self._store.get_all_dates())
        if store_dates:
            all_dates = [d for d in all_dates if d in store_dates]
        self.all_dates: list = all_dates

    @staticmethod
    def _is_main_board(ts_code: str) -> bool:
        """主板判断：SH 以6开头且非688科创板；SZ 以0开头且非300/301创业板。"""
        code, exchange = ts_code[:6], ts_code[-2:]
        if exchange == 'SH':
            return code.startswith('6') and not code.startswith('688')
        if exchange == 'SZ':
            return code.startswith('0') and not code.startswith('3')
        return False  # BJ 及其他交易所一律排除

    # ------------------------------------------------------------------ #
    #  底层读取
    # ------------------------------------------------------------------ #

    def _read_section(self, date_str: str) -> Optional[pd.DataFrame]:
        """优先读分区存储；为空时回退到旧版 data/section/ 目录。"""
        df = self._store.read_section(date_str)
        if df is not None and not df.empty:
            return df

        data_dir = getattr(self.cfg, 'data_dir', None)
        if not data_dir:
            return None
        parquet_path = os.path.join(data_dir, f"{date_str}.parquet")
        csv_path = os.path.join(data_dir, f"{date_str}.csv")
        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def get_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        读取指定交易日的截面数据，过滤 ST 股后返回。
        若该日期无对应数据（停市、数据缺失），返回 None。

        ST 过滤逻辑：
        - 优先使用 is_st 列（来自 namechange 数据），记录的是当时的 ST 状态
        - 如果 is_st 列不存在，则回退到使用 name 列（当前名称），但会给出警告

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期
        """
        if date_str in self.cache:
            self.cache.move_to_end(date_str)
            return self.cache[date_str]

        df = self._read_section(date_str)
        if df is None or df.empty:
            return None

        # 附加行业 / 市场 beta 特征
        if self.market_feat is not None:
            extra = self.market_feat.get(date_str)
            if extra is not None:
                df = df.merge(extra, on='ts_code', how='left')
                if self.market_feat_zscore:
                    cols = [c for c in extra.columns if c != 'ts_code']
                    vals = df[cols]
                    # 市场级特征在截面内恒定（std=0），不能做截面 z-score，
                    # 否则会被压成常数；它们已在 builder 里做过时序 z-score。
                    std = vals.std()
                    varying = [c for c in cols if std[c] > 0]
                    if varying:
                        v = vals[varying]
                        df[varying] = ((v - v.mean()) / std[varying]).fillna(0.0)

        # 合并 stock_list 的 name 列
        df = df.merge(self.stock_df[["ts_code", "name"]], on="ts_code", how='inner')

        # ST 过滤：优先使用 is_st 列（当时的 ST 状态）
        if getattr(self.cfg, 'filter_st', True):
            if 'is_st' in df.columns:
                df = df[df['is_st'] == 0]
            else:
                # 回退到使用当前名称（不推荐，会引入未来信息）
                import warnings
                if not hasattr(self, '_st_warning_shown'):
                    warnings.warn(
                        "is_st 列不存在，使用当前名称过滤 ST 股（会引入未来信息）。"
                        "建议运行 build_st_flag() 生成 is_st 列。",
                        UserWarning
                    )
                    self._st_warning_shown = True
                # name_y = merge 后来自 stock_list 的当前名称（name_x 是截面文件里的历史名称）
                name_col = 'name_y' if 'name_y' in df.columns else 'name'
                df = df[~df[name_col].str.contains('ST', na=False)]

        min_mv = getattr(self.cfg, 'min_mv', None)
        if min_mv is not None and 'total_mv' in df.columns:
            df = df[df['total_mv'].fillna(0) >= min_mv]

        self.cache[date_str] = df
        if len(self.cache) > self.CACHE_SIZE:
            self.cache.popitem(last=False)
        return df

    def get_window(self, start_date: str, end_date: str,
                   columns: Optional[list] = None) -> pd.DataFrame:
        """
        一次性读取日期区间的截面数据（走 DuckDB 分区裁剪，比逐日读取快）。
        返回包含 trade_date 列的长表。
        """
        return self._store.read_window(start_date, end_date, columns=columns)

    def get_trading_dates(self) -> list:
        """返回完整交易日历列表（YYYYMMDD 字符串，2020-01-01 起）。"""
        return self.all_dates

    def close(self) -> None:
        self.cache.clear()
        self._store.close()
