"""
行业 / 市场 Beta 特征（beta+alpha 总体最优的关键补充）。

为什么需要
----------
standardize 阶段对每个因子做了「行业 + 市值中性化」，取的是回归残差。
这带来一个副作用：模型**看不到股票属于哪个行业**，也就无法利用行业轮动（beta）。

但选股的目标如果是 top-K 的**总收益**（而不是纯 alpha），行业 beta 是真实可赚的钱：
一个行业整体上涨时，行业内平庸的股票也能涨。把这部分从特征里剔掉、却留在 label 里，
等于给模型塞了一块它永远学不会的噪声。

本模块把行业与市场层面的信息重新补回特征侧，使「原始 label + 保留 beta 的特征」
形成自洽的组合。所有特征在截面内对同行业股票取相同值，树模型与线性模型都能直接用。

用法
----
    builder = MarketFeatureBuilder(store)
    builder.prepare(dates)                    # 一次性预计算
    extra = builder.get('20250102')           # DataFrame[ts_code, ind_ret_5d, ...]

由 DataLoader 在 config.add_market_features = True 时自动附加。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence

# 计算所需的原始列（只读这几列，避免把 700 列全表读进内存）
SOURCE_COLS = ['ts_code', 'trade_date', 'industry',
               'pct_chg', 'total_mv', 'net_mf_amount', 'pe_ttm']

# 产出的特征名（供各策略 factor_cols 引用）
MARKET_FEATURE_COLS = [
    'ind_ret_1d',        # 行业当日等权收益
    'ind_ret_5d',        # 行业过去 5 日累计收益（行业动量）
    'ind_ret_20d',       # 行业过去 20 日累计收益（行业中期趋势）
    'ind_up_5d',         # 行业过去 5 日上涨占比（行业宽度）
    'ind_up_20d',
    'ind_flow_5d',       # 行业主力资金净流入 / 行业总市值
    'ind_flow_20d',
    'ind_mom_rank',      # 行业 20 日动量在当日所有行业中的分位（0~1）
    'ind_pe_z',          # 行业估值中位数相对自身历史的 z-score
    'mkt_ret_5d',        # 全市场过去 5 日收益
    'mkt_ret_20d',
    'mkt_up_1d',         # 全市场当日上涨占比（市场宽度）
    'mkt_up_5d',
]


class MarketFeatureBuilder:
    """
    预计算行业 / 市场层面的日度特征，按 (trade_date, ts_code) 供给模型。

    所有计算只依赖几列原始字段，用 DuckDB 的分区裁剪一次性读入，
    在 pandas 里做行业聚合与滚动，内存占用很小。
    """

    def __init__(self, store, windows: Sequence[int] = (5, 20)):
        self.store = store
        self.windows = list(windows)
        self._by_date: Dict[str, pd.DataFrame] = {}   # date -> DataFrame[ts_code, feats]
        self._ready = False

    # ------------------------------------------------------------------ #

    def prepare(self, dates: Optional[Sequence[str]] = None,
                industry_map: Optional[pd.DataFrame] = None) -> 'MarketFeatureBuilder':
        """
        预计算指定日期区间的行业 / 市场特征。

        Parameters
        ----------
        dates : list[str] | None
            需要覆盖的交易日；None 表示存储中全部日期。
        industry_map : DataFrame[ts_code, industry] | None
            行业归属表；None 时直接用截面里的 industry 列。
        """
        if not self.store.table_exists():
            self._ready = True
            return self

        all_dates = self.store.get_all_dates()
        if dates is not None:
            wanted = set(dates)
            all_dates = [d for d in all_dates if d in wanted]
        if not all_dates:
            self._ready = True
            return self

        # 多往前读一段，保证最早几天的滚动窗口也有值
        lookback = max(self.windows) * 3
        start_idx = 0
        if len(all_dates) > lookback:
            start_date = all_dates[0]
            pos = self.store.get_all_dates().index(start_date)
            start_idx = max(0, pos - lookback)
        from_date = self.store.get_all_dates()[start_idx]

        df = self.store.read_window(from_date, all_dates[-1], columns=SOURCE_COLS)
        if df is None or df.empty:
            self._ready = True
            return self

        if industry_map is not None:
            df = df.drop(columns=['industry'], errors='ignore')
            df = df.merge(industry_map, on='ts_code', how='left')

        if 'industry' not in df.columns:
            self._ready = True
            return self

        df['industry'] = df['industry'].fillna('__UNKNOWN__')
        df['ret'] = pd.to_numeric(df.get('pct_chg'), errors='coerce') / 100.0
        df['_up'] = (df['ret'] > 0).astype(float)
        df.loc[df['ret'].isna(), '_up'] = np.nan
        df['_mv'] = pd.to_numeric(df.get('total_mv'), errors='coerce')
        df['_nmf'] = pd.to_numeric(df.get('net_mf_amount'), errors='coerce')
        df['_pe'] = pd.to_numeric(df.get('pe_ttm'), errors='coerce')
        df = df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

        # ---- 行业日度聚合 ----
        ind = df.groupby(['trade_date', 'industry'], observed=True).agg(
            ind_ret_1d=('ret', 'mean'),
            _up=('_up', 'mean'),
            _nmf=('_nmf', 'sum'),
            _mv=('_mv', 'sum'),
            _pe=('_pe', 'median'),
        ).reset_index().sort_values(['industry', 'trade_date']).reset_index(drop=True)

        gp = ind.groupby('industry', sort=False)
        for w in self.windows:
            ind[f'ind_ret_{w}d'] = gp['ind_ret_1d'].transform(lambda s: s.rolling(w, min_periods=1).sum())
            ind[f'ind_up_{w}d'] = gp['_up'].transform(lambda s: s.rolling(w, min_periods=1).mean())
            ind[f'ind_nmf_{w}d'] = gp['_nmf'].transform(lambda s: s.rolling(w, min_periods=1).sum())
        ind['_mv_smooth'] = gp['_mv'].transform(lambda s: s.rolling(20, min_periods=1).mean())

        for w in self.windows:
            ind[f'ind_flow_{w}d'] = ind[f'ind_nmf_{w}d'] / ind['_mv_smooth'].replace(0, np.nan)

        # 行业估值相对自身历史的 z-score（60 日）
        pe_mean = gp['_pe'].transform(lambda s: s.rolling(60, min_periods=20).mean())
        pe_std = gp['_pe'].transform(lambda s: s.rolling(60, min_periods=20).std())
        ind['ind_pe_z'] = (ind['_pe'] - pe_mean) / pe_std.replace(0, np.nan)

        # 行业动量在当日所有行业中的分位
        main_w = max(self.windows)
        ind['ind_mom_rank'] = ind.groupby('trade_date')[
            f'ind_ret_{main_w}d'].rank(pct=True)

        # 前 7 个是滚动类行业特征，ind_mom_rank / ind_pe_z 单独算；
        # 顺序严格对齐 MARKET_FEATURE_COLS，方便下游按名取列
        ind = ind[['trade_date', 'industry']
                  + MARKET_FEATURE_COLS[:7] + ['ind_mom_rank', 'ind_pe_z']]

        # ---- 全市场聚合 ----
        mkt = df.groupby('trade_date').agg(
            mkt_ret_1d=('ret', 'mean'),
            _up=('_up', 'mean'),
        ).reset_index().sort_values('trade_date').reset_index(drop=True)
        for w in self.windows:
            mkt[f'mkt_ret_{w}d'] = mkt['mkt_ret_1d'].rolling(w, min_periods=1).sum()
            mkt[f'mkt_up_{w}d'] = mkt['_up'].rolling(w, min_periods=1).mean()
        mkt['mkt_up_1d'] = mkt['_up']

        # 市场级特征在截面内是常数，做截面 z-score 会被压成 0 或 NaN。
        # 改用「相对自身历史」的时序 z-score（60 日）：既保证量纲可比，
        # 语义上也正是想要的「当前市场是否异常强/弱」。
        for c in MARKET_FEATURE_COLS[9:]:
            mu = mkt[c].rolling(60, min_periods=20).mean()
            sd = mkt[c].rolling(60, min_periods=20).std().replace(0, np.nan)
            mkt[c] = (mkt[c] - mu) / sd

        # ---- 拼到个股上 ----
        stock = df[['ts_code', 'trade_date', 'industry']].copy()
        out = stock.merge(ind, on=['trade_date', 'industry'], how='left')
        out = out.drop(columns=['industry'])
        out = out.merge(mkt[['trade_date'] + MARKET_FEATURE_COLS[9:]],
                        on='trade_date', how='left')

        out = out.replace([np.inf, -np.inf], np.nan)
        self._by_date = {d: g.drop(columns=['trade_date']).reset_index(drop=True)
                         for d, g in out.groupby('trade_date')}
        self._ready = True
        return self

    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self._ready

    def covered_dates(self) -> List[str]:
        return sorted(self._by_date.keys())

    def get(self, trade_date: str) -> Optional[pd.DataFrame]:
        """返回 DataFrame[ts_code, <行业/市场特征>]；未覆盖返回 None。"""
        return self._by_date.get(str(trade_date))

    @staticmethod
    def columns() -> List[str]:
        return list(MARKET_FEATURE_COLS)
