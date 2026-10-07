"""
因子列解析：把「数据里实际存在的因子」自动配进策略。

背景
----
factor_manager 会算出 190 个原始因子（对应 226 个 _standard 列），
但各策略 Config 里的 factor_cols 是手工挑选的，长期落后于因子库的扩充
（比如 Alpha101 31 个、动量族、风险族、流动性族、质量成长族一度完全没被用到）。

本模块提供两种取值方式：
    factor_set='curated'  → 沿用 Config 里手工挑选的列表（默认不变）
    factor_set='all'      → 手工列表 ∪ 数据中发现的全部因子

风格（factor_style）：
    'raw'      → 原始因子列名（树模型用，LightGBM / Ranker）
    'standard' → <因子>_standard 列名（线性模型用，ElasticNet）
"""

import os
from typing import List, Optional

import pyarrow.parquet as pq

from data_store import DataStore
from strategy.market_features import MARKET_FEATURE_COLS

# 最终产物文件（用于区分「合并阶段就有的原始行情列」和「因子计算新增的列」）
FINAL_RESULT_PATH = 'data/final_result.parquet'

# 明确不是因子的列（价格、成交、标识、辅助）
BASE_EXCLUDE = {
    'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'pct_change',
    'vol', 'amount', 'adj_factor', 'close_hfq', 'open_hfq', 'high_hfq', 'low_hfq',
    'name', 'reason', 'industry', 'is_st', 'ts_code', 'trade_date',
    'float_values', 'total_mv', 'circ_mv',
}

_cache: dict = {}


def _merge_base_columns() -> set:
    """合并阶段就存在的原始列（行情 / 基本面快照 / 资金流…），不是因子。"""
    if 'base' in _cache:
        return _cache['base']
    cols = set()
    if os.path.exists(FINAL_RESULT_PATH):
        try:
            cols = set(pq.ParquetFile(FINAL_RESULT_PATH).schema_arrow.names)
        except Exception:
            cols = set()
    _cache['base'] = cols
    return cols


def _section_columns(store: DataStore) -> List[str]:
    """取最新一个截面的列名（只读 schema，开销极小）。"""
    key = f'sec::{store.market_path}'
    if key in _cache:
        return _cache[key]
    cols: List[str] = []
    if store.table_exists():
        pdir = store._partition_dir(store.get_latest_date())
        files = sorted(pdir.glob('*.parquet'))
        if files:
            cols = list(pq.ParquetFile(files[0]).schema_arrow.names)
    _cache[key] = cols
    return cols


def discover_factor_cols(store: DataStore, style: str = 'raw') -> List[str]:
    """
    从截面数据中识别出全部可用因子列。

    Parameters
    ----------
    store : DataStore
    style : 'raw' | 'standard'
        'raw'      → 返回原始因子列名
        'standard' → 返回 <因子>_standard 列名

    Returns
    -------
    list[str]  按截面中出现的顺序排列
    """
    cols = _section_columns(store)
    if not cols:
        return []

    if style == 'standard':
        bad = {f'{b}_standard' for b in BASE_EXCLUDE}
        out = [c for c in cols
               if c.endswith('_standard') and not c.startswith('label') and c not in bad]
    else:
        base = _merge_base_columns()
        out = []
        for c in cols:
            if c in base:                       # 合并阶段就有的原始行情/基本面列
                continue
            if c.endswith(('_neutral', '_standard')):   # 中性化/标准化的派生变体
                continue
            if c.startswith('label') or c.endswith('_hfq'):
                continue
            if c in BASE_EXCLUDE or c == 'industry':
                continue
            out.append(c)
    return out


def resolve_factor_cols(config, store: DataStore) -> List[str]:
    """
    按 config.factor_set 解析出最终参与建模的因子列。

    'curated' → 原样返回 config.factor_cols
    'all'     → config.factor_cols ∪ discover_factor_cols(...)，保持原有顺序

    行业/市场 beta 特征始终保留在最后。
    """
    curated = list(getattr(config, 'factor_cols', []))
    if getattr(config, 'factor_set', 'curated') != 'all':
        return curated

    style = getattr(config, 'factor_style', 'raw')
    discovered = discover_factor_cols(store, style)

    beta = [c for c in curated if c in MARKET_FEATURE_COLS]
    non_beta = [c for c in curated if c not in MARKET_FEATURE_COLS]

    seen, out = set(), []
    for c in non_beta + discovered:
        if c not in seen:
            seen.add(c)
            out.append(c)
    for c in beta:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def apply_factor_set(config, store: DataStore) -> int:
    """
    就地更新 config.factor_cols，返回最终列数。

    由 DataLoader 在初始化时调用（那时才知道数据里实际有哪些列）。
    """
    config.factor_cols = resolve_factor_cols(config, store)
    return len(config.factor_cols)
