import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
import os


def series_to_section(colume_list=None, incremental: bool = False, main_board_only: bool = True):
    """
    将 data/series/ 下按股票存储的时序 CSV 转换为按日期存储的截面 CSV，
    输出到 data/section/<trade_date>.csv。

    main_board_only=True（默认）时，只保留 data/raw/stock_list/stock_list.csv
    中的 ts_code（沪深两市主板），过滤掉创业板、科创板、北交所等。

    incremental=True 时的行为：
      - 读取所有 series 文件并 concat 成 final_result（同非增量模式）
      - 抽检任意一个已有截面文件，判断其列是否覆盖 final_result 的所有列
        （section 目录下所有文件列结构一致，一个文件代表全部）
        ① 列完整：已有日期的截面文件全部跳过，只写入新日期文件
        ② 列不完整：向已有日期的截面文件补充缺失列，同样写入新日期文件
    """
    series_path = "data/series/"
    final_result = []
    for filename in glob.glob(series_path + "[0-9]*.csv"):
        if colume_list is not None and "ts_code" in colume_list and "trade_date" in colume_list:
            df = pd.read_csv(filename, usecols=colume_list)
        else:
            df = pd.read_csv(filename)
        final_result.append(df)

    final_result = pd.concat(final_result, ignore_index=True)

    if main_board_only:
        stock_list = pd.read_csv("data/raw/stock_list/stock_list.csv", usecols=["ts_code"])
        main_board_codes = set(stock_list["ts_code"])
        before = len(final_result)
        final_result = final_result[final_result["ts_code"].isin(main_board_codes)]
        print(f"主板过滤：{before} 行 -> {len(final_result)} 行（保留 {final_result['ts_code'].nunique()} 只股票）")

    print(final_result.columns)

    section_path = "data/section/"

    # 增量模式：抽检一个已有截面文件，确定需要补充哪些列（空列表 = 列完整可全部跳过）
    missing_cols: list = []
    if incremental:
        sample_files = glob.glob(section_path + "*.csv")
        if sample_files:
            sample_cols = set(pd.read_csv(sample_files[0], nrows=0).columns)
            missing_cols = [c for c in final_result.columns if c not in sample_cols]
            if missing_cols:
                print(f"  截面文件缺少 {len(missing_cols)} 列，将向已有文件补列：{missing_cols}")
            else:
                print("  截面文件列完整，跳过已有日期。")

    for trade_date, group in final_result.groupby("trade_date"):
        output_file_name = section_path + str(trade_date) + ".csv"
        group = group.sort_values("ts_code") if "ts_code" in group.columns else group
        group = group.drop_duplicates(subset="ts_code", keep='first', ignore_index=True)

        if incremental and os.path.exists(output_file_name):
            if not missing_cols:
                continue  # 列完整，跳过
            # 列不完整：只补充缺失列（merge on ts_code）
            patch_cols = [c for c in missing_cols if c in group.columns]
            if patch_cols:
                existing_df = pd.read_csv(output_file_name)
                patch = group[['ts_code'] + patch_cols]
                existing_df = existing_df.merge(patch, on='ts_code', how='left')
                existing_df.to_csv(output_file_name, index=False)
                print(f"  已补列 {trade_date}")
            continue

        # 新日期（或非增量模式）：直接写入
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


def standardize(incremental: bool = False):
    """
    对 data/section/ 下每张截面 CSV 做三步处理：
      1. 缩尾去极值（1%~99% 百分位截断）
      2. 行业+市值中性化（OLS 残差），中性化后的列命名为 <factor>_neutral
      3. Z-score 标准化，最终列命名为 <factor>_standard

    incremental=True 时，若文件中已包含 should_neutralize 所有列对应的
    _standard 列，则跳过该文件。
    """

    stock_info_df = pd.read_csv('data/raw/stock_list/stock_list.csv')

    basic_column_list = [
        'ts_code', 'trade_date', 'name', 'reason',
        'pct_chg', 'pct_change', 'industry',
        'macd_air_refuel', 'macd_divergence', 'vol_breakout',
    ]

    should_neutralize = [
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
    ]

    def winsorize_series(s, lower_perc=0.01, upper_perc=0.99):
        return s.clip(s.quantile(lower_perc), s.quantile(upper_perc))

    section_path = 'data/section/'
    all_files = sorted(glob.glob(section_path + '*.csv'))
    if not all_files:
        print('  无截面文件，跳过。')
        return

    total = len(all_files)
    for i, file in enumerate(all_files):
        df = pd.read_csv(file)

        if incremental:
            existing = set(df.columns)
            needed = {f + '_standard' for f in should_neutralize}
            if needed.issubset(existing):
                print(f'\r  跳过 {i + 1}/{total}', end='', flush=True)
                continue

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
            if fac + '_standard' in df.columns:
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

        df.to_csv(file, index=False)
        print(f'\r  标准化 {i + 1}/{total}', end='', flush=True)
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
