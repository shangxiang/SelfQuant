import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

import pyarrow.parquet as pq

from data_store import DataStore, normalize_date, MARKET_DIRNAME, PART_COL

SERIES_PATH = "data/series/"
MARKET_PATH = Path("data") / MARKET_DIRNAME


def _series_files() -> list:
    """series 目录下的时序文件（Parquet 优先）。"""
    files = sorted(glob.glob(SERIES_PATH + "[0-9]*.parquet"))
    if files:
        return files
    return sorted(glob.glob(SERIES_PATH + "[0-9]*.csv"))


def _read_series_file(path: str, columns=None) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _series_columns() -> list:
    """取一份时序文件的列名，作为"截面应有哪些列"的参照。"""
    files = _series_files()
    if not files:
        return []
    return list(_read_series_file(files[0]).columns)


def _series_dates() -> list:
    """
    截面应覆盖的交易日列表。
    以交易日历为准（避免单只股票停牌导致的日期缺失），下界取首份时序文件的最早日期。
    """
    cal_path = "data/raw/trade_cal.csv"
    if not os.path.exists(cal_path):
        return []
    cal = pd.read_csv(cal_path, dtype={'cal_date': str})
    dates = sorted(cal[cal['is_open'] == 1]['cal_date'].dropna().tolist())
    files = _series_files()
    if not files:
        return dates
    first = _read_series_file(files[0], columns=['trade_date'])
    lo = first['trade_date'].astype(str).min()
    return [d for d in dates if d >= lo]


def _market_columns(date_str: str) -> list:
    """
    只读 parquet footer 取某交易日截面已有的列名（不加载数据）。

    增量更新时要判断「哪些分区已经标准化过」，若逐个 read_section 会把
    几十 GB 全读一遍；只读 footer 只需毫秒级，扫两千多个分区也就几秒。
    """
    pdir = MARKET_PATH / f'{PART_COL}={normalize_date(date_str)}'
    if not pdir.is_dir():
        return []
    files = sorted(pdir.glob('*.parquet'))
    if not files:
        return []
    try:
        return list(pq.ParquetFile(files[0]).schema_arrow.names)
    except Exception:
        return []


def _is_standardized(date_str: str) -> bool:
    """
    判断某交易日的分区是否已经标准化过。

    判据用「分区里是否存在 *_standard 列」而不是「是否存在全部 *_standard 列」：
    各交易日的列数本来就不一样（早期 589 列、近期 724 列，因子有预热期），
    且 label 类列在最新几天必然全为 NaN 而不会被标准化，
    用固定列集做判据会把所有历史日期误判成未完成，进而触发全量重建。
    而由 series 直接派生的分区一定不含任何 _standard 列，两者区分度足够。
    """
    return any(c.endswith('_standard') for c in _market_columns(date_str))


def pending_section_dates(needed: set = None) -> list:
    """
    返回还需要（重新）构建 + 标准化的交易日：
      1. series 里有、market 里还没有的日期（新增交易日）
      2. market 里已有、但还没做标准化的日期（上次跑到一半留下的半成品）

    Parameters
    ----------
    needed : set  兼容旧签名，已不使用（判据改为分区自洽检查）。
    """
    store = DataStore()
    try:
        have = set(store.get_all_dates())
    finally:
        store.close()

    series_dates = set(_series_dates())
    todo = set(d for d in series_dates if d not in have)
    for d in have:
        if not _is_standardized(d):
            todo.add(d)
    return sorted(todo)


def series_to_section(colume_list=None, incremental: bool = False,
                      main_board_only: bool = True, only_dates=None,
                      chunk_size: int = 200) -> list:
    """
    将时序层 data/series/ 转置为截面层 data/market/trade_date=YYYYMMDD/。

    走 DataStore 的流式构建：按股票分块读取 → 按交易日切分 → 写入对应分区，
    内存只保留"一块股票"的数据。旧实现一次性 concat 全部时序文件，
    在 677 万行 × 数百列的规模下必然 OOM。

    Parameters
    ----------
    incremental : bool
        True 时先检查已有截面是否缺列；列完整则只补新交易日，缺列则全量重建。
    only_dates : list[str] | None
        只构建指定的交易日（用于分段验证 / 补算）。

    Returns
    -------
    list[str]  本次实际构建的交易日（升序）。增量且无需更新时返回空列表，
               调用方可据此决定后续标准化要处理哪些日期。
    """
    store = DataStore()
    try:
        targets = None
        if only_dates is not None:
            targets = [normalize_date(d) for d in only_dates]
        elif incremental:
            existing = set(store.get_all_dates())
            if existing:
                ref_cols = _series_columns()
                sample_cols = _market_columns(sorted(existing)[-1])
                missing = [c for c in ref_cols if c not in sample_cols]
                if missing:
                    print(f"  截面缺少 {len(missing)} 列，全量重建")
                else:
                    targets = pending_section_dates()
                    if not targets:
                        print("  截面已是最新，跳过。")
                        return []
                    print(f"  增量构建 {len(targets)} 个新交易日"
                          f"（{targets[0]} ~ {targets[-1]}）")

        store.build_sections_from_series(
            chunk_size=chunk_size,
            main_board_only=main_board_only,
            only_dates=targets,
            overwrite=True,
        )
        if targets is None:
            return [normalize_date(d) for d in _series_dates()]
        return list(targets)
    finally:
        store.close()



# 需要做「缩尾 → 行业市值中性化 → z-score 标准化」的因子清单。
# 提到模块级是为了让增量调度（pending_section_dates）能复用同一份判定标准。
SHOULD_NEUTRALIZE = [
    'label_1', 'label_3',
    'label', 'label_10', 'label_25',
    'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
    'total_share', 'float_share', 'free_share',
    'vol', 'amount_x', 'amount_y',
    'turnover_rate_x', 'turnover_rate_f', 'turnover_rate_y', 'volume_ratio',
    'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye',
    'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
    'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
    'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
    'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
    'net_mf_vol', 'net_mf_amount',
    'l_sell', 'l_buy', 'l_amount', 'net_amount', 'net_rate', 'amount_rate',
    'positive_flow', 'negative_flow', 'mfi',
    'raw_force_index', 'force_index_smoothed', 'force_index', 'vwap',
    'mtm_margin_balance_change', 'big_order_ratio', 'lhb_strength_5d',
    'volatility_20d', 'reversal_5d', 'high_low_spread',
    'gross_margin', 'debt_ratio', 'roe_ttm',
    'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
    'asset_growth_yoy', 'value_factor', 'cma_factor', 'momentum_12_1',
    'smb_mom', 'smb_squared_mom', 'hml_rmw', 'smb_hml', 'vol_mom',
    # 中短期动量 / 技术形态因子
    'ret_10d', 'ret_20d', 'ret_60d',
    'dist_52w_high', 'close_ma20_ratio',
    'up_day_ratio_20', 'vol_price_corr_20d', 'adx',
    # 高频痕迹因子
    'turnover_amplitude_ratio', 'long_shadow_freq',
    'doji_freq', 'intraday_drawdown', 'gap_vs_range_ratio',
    # 多项式形状因子
    'poly_close_a1', 'poly_close_a2',
    'poly_vol_a1', 'poly_vol_a2',
    # 时序差分因子
    'K_chg_5d', 'K_chg_10d',
    'D_chg_5d', 'D_chg_10d',
    'J_chg_5d', 'J_chg_10d',
    'rsi_chg_5d', 'rsi_chg_10d',
    'macd_chg_5d', 'macd_chg_10d',
    'adx_chg_5d', 'adx_chg_10d',
    'volatility_20d_chg_5d', 'volatility_20d_chg_10d',
    'turnover_rate_x_chg_5d', 'turnover_rate_x_chg_10d',
    'reversal_5d_chg_5d', 'reversal_5d_chg_10d',
    'momentum_12_1_chg_5d', 'momentum_12_1_chg_10d',
    'rzye_chg_5d', 'rzye_chg_10d',
    # Alpha101因子
    'alpha101_1', 'alpha101_2', 'alpha101_3', 'alpha101_4', 'alpha101_5',
    'alpha101_6', 'alpha101_7', 'alpha101_8', 'alpha101_9', 'alpha101_10',
    'alpha101_11', 'alpha101_12', 'alpha101_13', 'alpha101_14', 'alpha101_15',
    'alpha101_16', 'alpha101_17', 'alpha101_18', 'alpha101_19', 'alpha101_20',
    'alpha101_22', 'alpha101_23', 'alpha101_25', 'alpha101_33', 'alpha101_34',
    'alpha101_41', 'alpha101_52', 'alpha101_53', 'alpha101_54', 'alpha101_57',
    'alpha101_101',
    # Size因子
    'size', 'float_size',
    # Value因子
    'earnings_to_price', 'book_to_market', 'ocf_to_market', 'fcf_to_market',
    'sales_to_market',
    # Reversal因子
    'small_cap_reversal_21d', 'price_dist',
    # Momentum因子
    'return_5d', 'return_21d', 'return_42d', 'return_63d', 'return_126d',
    'return_252d', 'ma_20d', 'price_position_ir_60d', 'rsrs', 'days_down_up',
    # Risk因子
    'return_std_21d', 'return_std_42d', 'return_std_63d', 'return_std_126d',
    'return_std_252d', 'sharpe_60d', 'sharpe_750d', 'adjusted_sharpe_750d',
    'high_low_21d', 'high_low_42d', 'high_low_63d', 'high_low_126d',
    'high_low_252d', 'days_beyond_upper_lower_21d', 'log_price',
    # Liquidity因子
    'avg_turnover_5d', 'avg_turnover_10d', 'avg_turnover_20d',
    'std_turnover_21d', 'std_turnover_42d', 'std_turnover_63d',
    'std_turnover_126d', 'std_turnover_252d',
    'avg_turnover_21d', 'avg_turnover_42d', 'avg_turnover_63d',
    'avg_turnover_126d', 'avg_turnover_252d',
    'bias_turn_21d_252d', 'bias_std_turn_21d_252d',
    'bias_turn_42d_252d', 'bias_turn_63d_252d', 'bias_turn_126d_252d',
    'bias_turn_21d_504d', 'bias_std_turn_21d_504d',
    'bias_turn_42d_504d', 'bias_std_turn_42d_504d',
    'bias_turn_63d_504d', 'bias_std_turn_63d_504d',
    'bias_turn_126d_504d', 'bias_std_turn_126d_504d',
    'turnover_ma_20d_120d',
    # Quality因子
    'roa_ttm', 'net_margin', 'current_ratio', 'quick_ratio',
    'cash_flow_to_debt', 'earnings_quality',
    # Growth因子
    'roe_growth_yoy', 'eps_growth_yoy', 'revenue_growth_qoq',
    'profit_growth_qoq', 'gross_margin_growth', 'net_margin_growth',
    'ocf_growth_yoy',
]


def build_design_matrix(df, mv_col='total_mv', ind_col='industry'):
    """
    预先构造中性化用的解释变量矩阵（对数市值 + 行业哑变量），每个截面只算一次。

    后续每个因子只是从这个矩阵里取各自的非空子集做最小二乘，
    避免对每个因子重复 get_dummies（否则 200+ 因子 × 2000+ 交易日会慢一个数量级）。
    """
    if mv_col not in df.columns or ind_col not in df.columns:
        return None, None
    log_mv = np.log(pd.to_numeric(df[mv_col], errors='coerce').replace([np.inf, -np.inf], np.nan))
    dummies = pd.get_dummies(df[ind_col], prefix='ind', drop_first=True).astype(float)
    design = pd.concat([log_mv.rename('log_mv'), dummies], axis=1)
    design = design.replace([np.inf, -np.inf], np.nan)
    # 解释变量完整可用的行（与因子值无关，可跨因子复用）
    valid = design.notna().all(axis=1)
    return design, valid


def neutralize_with_design(df, factor_name, design, valid):
    """
    用预构造的设计矩阵对单个因子做 OLS 中性化，返回残差。

    等价于 statsmodels 的 OLS(factor ~ 1 + log_mv + 行业哑变量).resid，
    改用 numpy lstsq 求解，数值一致但快一个数量级。
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if design is None or factor_name not in df.columns:
        return result
    y = pd.to_numeric(df[factor_name], errors='coerce')
    mask = y.notna() & valid.reindex(df.index, fill_value=False)
    if mask.sum() < 2:
        return result
    X = np.column_stack([np.ones(int(mask.sum())), design.loc[mask].to_numpy(dtype=float)])
    yv = y.loc[mask].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    result.loc[mask] = yv - X @ beta
    return result


def neutralize_batch(df, factor_names, design, valid):
    """
    批量中性化：把缺失值掩码相同的因子归为一组，一次 lstsq 同时求解多个因变量。

    单个因子的成本主要在 np.linalg.lstsq（约 3000×30 的设计矩阵）。
    200+ 个因子逐个求解时这一步会占掉整日耗时的绝大部分；
    而实际掩码种类通常只有个位数（多数因子要么全有值、要么同步缺失），
    分组后可以把上百次 lstsq 压到几次。
    """
    result = {}
    if design is None or valid is None:
        return {f'{f}_neutral': None for f in factor_names}

    valid_arr = np.asarray(valid.reindex(df.index, fill_value=False), dtype=bool)
    groups = {}
    for fac in factor_names:
        y = pd.to_numeric(df[fac], errors='coerce')
        mask = y.notna().to_numpy(dtype=bool) & valid_arr
        groups.setdefault(mask.tobytes(), [mask, []])[1].append(fac)

    for _, (mask, facs) in groups.items():
        n = int(mask.sum())
        if n < 2:
            for fac in facs:
                result[f'{fac}_neutral'] = pd.Series(np.nan, index=df.index, dtype=float)
            continue
        X = np.column_stack([np.ones(n), design.loc[mask].to_numpy(dtype=float)])
        Y = df.loc[mask, facs].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        for i, fac in enumerate(facs):
            s = pd.Series(np.nan, index=df.index, dtype=float)
            s.loc[mask] = resid[:, i]
            result[f'{fac}_neutral'] = s
    return result


def neutralize_one_day(df, factor_name, mv_col='total_mv', ind_col='industry'):
    """
    对某一天的截面数据做行业+市值中性化，返回中性化后的因子 Series。

    方法：以因子值为因变量，对数市值和行业哑变量做 OLS 回归，取残差。
    残差剔除了市值和行业的共同影响，保留股票特异性信息。
    缺失因子/市值/行业的股票不参与回归，结果位置保持 NaN。
    """
    design, valid = build_design_matrix(df, mv_col=mv_col, ind_col=ind_col)
    return neutralize_with_design(df, factor_name, design, valid)


def standardize(incremental: bool = False, only_dates=None, out_dir: str = None):
    """
    对截面层每个交易日做三步处理：
      1. 缩尾去极值（1%~99% 百分位截断）
      2. 行业+市值中性化（OLS 残差），中性化后的列命名为 <factor>_neutral
      3. Z-score 标准化，最终列命名为 <factor>_standard

    逐个交易日读写 data/market/trade_date=YYYYMMDD/，内存只保留一天的数据；
    处理完回写时会把分区内的多个 part 文件压实成单个文件。

    incremental=True 时，若分区中已包含 should_neutralize 所有列对应的
    _standard 列，则跳过该交易日。
    only_dates 用于只处理指定交易日（分段验证 / 补算）。
    out_dir  指定时结果写到另一个分区目录，写完后用 swap_market_dir() 整体换入。
    """

    stock_info_df = pd.read_csv('data/raw/stock_list/stock_list.csv')

    basic_column_list = [
        'ts_code', 'trade_date', 'name', 'reason',
        'pct_chg', 'pct_change', 'industry',
        'macd_air_refuel', 'macd_divergence', 'vol_breakout',
        'adj_factor', 'close_hfq', 'open_hfq', 'high_hfq', 'low_hfq',
    ]

    should_neutralize = list(SHOULD_NEUTRALIZE)

    def winsorize_series(s, lower_perc=0.01, upper_perc=0.99):
        return s.clip(s.quantile(lower_perc), s.quantile(upper_perc))

    store = DataStore()
    out_store = DataStore(market_dir=out_dir) if out_dir else store
    try:
        dates = store.get_all_dates()
        if only_dates is not None:
            wanted = {normalize_date(d) for d in only_dates}
            dates = [d for d in dates if d in wanted]

        total = len(dates)
        if total == 0:
            print('  无截面数据，跳过。')
            return

        print(f"使用分区存储处理截面数据，共 {total} 个交易日")

        for i, trade_date in enumerate(dates):
            df = store.read_section(trade_date)
            if df is None or df.empty:
                continue

            if incremental and _is_standardized(trade_date):
                # 已标准化过：直接跳过。判据见 _is_standardized 的说明。
                print(f'\r  跳过 {i + 1}/{total}', end='', flush=True)
                continue

            for col in ('industry', 'industry_x', 'industry_y'):
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
            df = df.merge(stock_info_df[['ts_code', 'industry']], on='ts_code', how='left')

            for fac in should_neutralize:
                if fac in df.columns:
                    df[fac] = winsorize_series(df[fac])

            # 解释变量矩阵每个截面只构造一次，200+ 因子复用同一份
            design, valid = build_design_matrix(df, mv_col='total_mv', ind_col='industry')

            neutralized_cols = {}
            todo_factors = []
            for fac in should_neutralize:
                if fac not in df.columns:
                    continue
                if fac + '_neutral' in df.columns:
                    continue
                if fac + '_standard' in df.columns:
                    continue
                if df[fac].isna().all():
                    neutralized_cols[fac + '_neutral'] = 0
                    continue
                todo_factors.append(fac)

            # 按缺失掩码分组批量求解，等价但快一个数量级
            if todo_factors and design is not None:
                neutralized_cols.update(neutralize_batch(df, todo_factors, design, valid))
            if neutralized_cols:
                df = pd.concat([df, pd.DataFrame(neutralized_cols, index=df.index)], axis=1)

            columns = df.columns.tolist()
            need_standardize_columns = [c for c in columns if c not in basic_column_list]
            new_columns = {}
            for col in need_standardize_columns:
                if '_standard' in col:
                    continue
                if col + '_standard' in df.columns:
                    continue
                std = df[col].std()
                if std == 0 or pd.isna(std):
                    continue
                if '_neutral' in col:
                    new_columns[col.replace('_neutral', '_standard')] = (df[col] - df[col].mean()) / std
                elif col + '_neutral' not in columns:
                    new_columns[col + '_standard'] = (df[col] - df[col].mean()) / std

            if not neutralized_cols and not new_columns:
                print(f'\r  跳过 {i + 1}/{total}', end='', flush=True)
                continue

            if new_columns:
                df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

            # 兜底：列名重复会让 parquet 写入直接失败
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]

            out_store.write_section(trade_date, df)
            print(f'\r  标准化 {i + 1}/{total}', end='', flush=True)
        print()
    finally:
        store.close()


def section_duplicates(only_dates=None):
    """清理截面分区中同一交易日内重复的 ts_code 行。

    only_dates 指定时只检查这些交易日（增量更新用，避免扫全量分区）。
    """
    store = DataStore()
    try:
        dates = store.get_all_dates()
        if only_dates is not None:
            wanted = {normalize_date(d) for d in only_dates}
            dates = [d for d in dates if d in wanted]
        for trade_date in dates:
            df = store.read_section(trade_date)
            if df is None or df.empty or 'ts_code' not in df.columns:
                continue
            deduped = df.drop_duplicates(subset="ts_code", keep='first', ignore_index=True)
            if len(deduped) != len(df):
                store.write_section(trade_date, deduped)
                print(f"  去重 {trade_date}: {len(df)} -> {len(deduped)}")
    finally:
        store.close()


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="时序层 → 截面层 → 标准化")
    ap.add_argument('--incremental', action='store_true',
                    help="只处理新交易日 / 未完成标准化的交易日")
    args = ap.parse_args()

    built = series_to_section(incremental=args.incremental)
    if args.incremental and not built:
        print("  无新增截面，跳过标准化。")
    else:
        standardize(incremental=args.incremental, only_dates=built or None)
    # section_duplicates(only_dates=built or None)
