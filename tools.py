import multiprocessing
import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def series_to_section(colume_list=None):
    """
    将 data/series/ 下按股票存储的时序 CSV 转换为按日期存储的截面 CSV，
    输出到 data/section/<trade_date>.csv。
    """
    series_path = "data/series/"
    final_result = []
    # 只匹配股票代码文件，排除 daily_basic_data.csv 等合并产物
    for filename in glob.glob(series_path + "[0-9]*.csv"):
        if colume_list is not None and "ts_code" in colume_list and "trade_date" in colume_list:
            df = pd.read_csv(filename, usecols=colume_list)
        else:
            df = pd.read_csv(filename)
        final_result.append(df)

    final_result = pd.concat(final_result, ignore_index=True)
    print(final_result.columns)

    section_path = "data/section/"
    for trade_date, group in final_result.groupby("trade_date"):
        group = group.sort_values("ts_code") if "ts_code" in group.columns else group
        # 同一交易日同一股票只保留第一条
        group = group.drop_duplicates(subset="ts_code", keep='first', ignore_index=True)
        output_file_name = section_path + str(trade_date) + ".csv"
        group.to_csv(output_file_name, index=False)
        print("已保存", str(trade_date))


def neutralize_one_day(df, factor_name, mv_col='total_mv', ind_col='industry'):
    """
    对某一天的截面数据做行业+市值中性化，返回中性化后的因子 Series。

    方法：以因子值为因变量，对数市值和行业哑变量做 OLS 回归，取残差。
    残差剔除了市值和行业的共同影响，保留股票特异性信息。
    缺失因子/市值/行业的股票不参与回归，结果位置保持 NaN。
    """
    work_df = df[[factor_name, mv_col, ind_col]].dropna().copy()

    work_df['log_mv'] = np.log(work_df[mv_col])
    ind_dummies = pd.get_dummies(work_df[ind_col], prefix='ind', drop_first=True)

    X = pd.concat([work_df[['log_mv']], ind_dummies], axis=1)
    X = sm.add_constant(X)

    y = work_df[factor_name]
    model = sm.OLS(y, X.astype(float)).fit()

    result = pd.Series(np.nan, index=df.index)
    result.loc[work_df.index] = model.resid
    return result


def _standardize_one_file(args: tuple) -> str:
    """处理单个截面文件的缩尾→中性化→标准化，返回文件路径。"""
    file, stock_info_df, basic_column_list, should_neutralize = args

    def winsorize_series(s, lower_perc=0.01, upper_perc=0.99):
        return s.clip(s.quantile(lower_perc), s.quantile(upper_perc))

    df = pd.read_csv(file)

    for col in ('industry', 'industry_x', 'industry_y'):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = df.merge(stock_info_df[['ts_code', 'industry']], on='ts_code', how='left')

    for fac in should_neutralize:
        if fac in df.columns:
            df[fac] = winsorize_series(df[fac])

    neutralized_cols = {}
    for fac in should_neutralize:
        if fac not in df.columns:
            continue
        if fac + '_neutral' in df.columns:
            continue
        if df[fac].isna().all():
            neutralized_cols[fac + '_neutral'] = 0
            continue
        neutralized_cols[fac + '_neutral'] = neutralize_one_day(
            df, factor_name=fac, mv_col='total_mv', ind_col='industry'
        )
    if neutralized_cols:
        df = pd.concat([df, pd.DataFrame(neutralized_cols, index=df.index)], axis=1)

    columns = df.columns.tolist()
    need_standardize_columns = [c for c in columns if c not in basic_column_list]
    new_columns = {}
    for col in need_standardize_columns:
        if '_standard' in col:
            continue
        std = df[col].std()
        if std == 0 or pd.isna(std):
            continue
        if '_neutral' in col:
            new_columns[col.replace('_neutral', '_standard')] = (df[col] - df[col].mean()) / std
        elif col + '_neutral' not in columns:
            new_columns[col + '_standard'] = (df[col] - df[col].mean()) / std

    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    df.to_csv(file, index=False)
    return file


def standardize():
    """
    对 data/section/ 下每张截面 CSV 做三步处理：
      1. 缩尾去极值（1%~99% 百分位截断）
      2. 行业+市值中性化（OLS 残差），中性化后的列命名为 <factor>_neutral
      3. Z-score 标准化，最终列命名为 <factor>_standard

    说明：
      - should_neutralize 中的因子先中性化再标准化
      - 不在 should_neutralize 也不在 basic_column_list 的因子直接 z-score 标准化
      - size_factor / smb_squared 是市值的直接函数，中性化会归零，故不放入中性化列表，
        走直接 z-score 标准化路径
    """

    stock_info_df = pd.read_csv('data/raw/stock_list/stock_list.csv')

    basic_column_list = [
        'ts_code', 'trade_date', 'name', 'reason',
        'pct_chg', 'pct_change', 'industry',
        'macd_air_refuel', 'macd_divergence', 'vol_breakout',
    ]

    should_neutralize = [
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
    ]

    section_path = 'data/section/'
    files = glob.glob(section_path + '*.csv')
    total = len(files)

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        task_args = [(f, stock_info_df, basic_column_list, should_neutralize) for f in files]
        futures = {executor.submit(_standardize_one_file, a): a[0] for a in task_args}
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                print(f'\r标准化 {completed}/{total}', end='', flush=True)
            except Exception as e:
                print(f'\n  ✗ {futures[future]}: {e}')
    print()


def section_duplicates():
    """清理截面文件中同一交易日内重复的 ts_code 行。"""
    path = "data/section/"
    for file in glob.glob(path + "*.csv"):
        df = pd.read_csv(file)
        df.drop_duplicates(subset="ts_code", keep='first', inplace=True, ignore_index=True)
        df.to_csv(file, index=False)


if __name__ == '__main__':
    series_to_section()
    standardize()
    # section_duplicates()
