import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
import os

def series_to_section(colume_list=None):
    """
    用于把计算好因子的时序数据转为截面数据
    :param file_path:
    :return:
    """
    series_path = "data/series/"
    final_result = []
    for filename in glob.glob(series_path + "*.csv"):
        # 增量更新指定列
        if colume_list is not None and "ts_code" in colume_list and "trade_date" in colume_list:
            df = pd.read_csv(filename, usecols=colume_list)
        else:
            df = pd.read_csv(filename)
        final_result.append(df)

    final_result = pd.concat(final_result)
    print(final_result.columns)

    section_path = "data/section/"
    for trade_date, group in final_result.groupby("trade_date"):
        group = group.sort_values("ts_code") if "ts_code" in group.columns else group
        output_file_name = section_path + str(trade_date) + ".csv"
        # if os.path.exists(output_file_name):
        #     basic_df = pd.read_csv(output_file_name)
        #     # 支持增量更新指定列
        #     if colume_list is not None:
        #         if "ts_code" in colume_list and "trade_date" in colume_list:
        #             basic_df = pd.merge(basic_df, group, how="left", on=["trade_date", "ts_code"])
        #             basic_df.to_csv(output_file_name, index=False)
        # else:
        output_file_name.drop_duplicates(subset="ts_code", keep='first', inplace=True, ignore_index=False)
        group.to_csv(output_file_name, index=False)
        print("已保存", str(trade_date))
    pass


def neutralize_one_day(df, factor_name, mv_col='total_mv', ind_col='industry'):
    """
    对某一天的截面数据做行业市值中性化
    返回：中性化后的因子 Series（索引与原数据对齐）
    """
    # 3.1 剔除缺失值（因子、市值、行业任一缺失则去掉）
    work_df = df[[factor_name, mv_col, ind_col]].dropna().copy()

    # 3.2 市值取对数
    work_df['log_mv'] = np.log(work_df[mv_col])

    # 3.3 生成行业哑变量（drop_first=True 避免共线性）
    ind_dummies = pd.get_dummies(work_df[ind_col], prefix='ind', drop_first=True)

    # 3.4 构建自变量矩阵：对数市值 + 行业哑变量 + 截距
    X = pd.concat([work_df[['log_mv']], ind_dummies], axis=1)
    X = sm.add_constant(X)

    # 3.5 OLS 回归，取残差
    y = work_df[factor_name]
    model = sm.OLS(y, X.astype(float)).fit()

    # 3.6 将残差放入原数据的相应位置（未参与回归的股票保持 NaN）
    result = pd.Series(np.nan, index=df.index)
    result.loc[work_df.index] = model.resid
    return result


def standardize():
    """
    用于标准化截面数据，使其可比
    :return:
    """

    # 获取下行业信息
    stock_info_df = pd.read_csv("data/raw/stock_list/stock_list.csv")

    basic_column_list = [
        'ts_code', 'trade_date', 'name', 'reason', 'pct_chg', 'pct_change', 'turnover_rate_x',
        'turnover_rate_y', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
        'dv_ratio', 'dv_ttm', 'net_rate', 'amount_rate', 'industry', 'macd_air_refuel', "macd_divergence",
        "vol_breakout", "lhb_strength_5d"
    ]

    # 需要中性化的因子：行业和市值不同带来巨大偏差的指标
    should_neutralize = [
        'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
        'total_share', 'float_share', 'free_share',
        'vol', 'amount_x', 'amount_y', 'turnover_rate_x', 'turnover_rate_f',
        'turnover_rate_y', 'volume_ratio',
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
        'volatility_20d', 'reversal_5d', 'high_low_spread', 'gross_margin', 'debt_ratio',
        'roe_ttm', 'revenue_growth_yoy', 'profit_growth_yoy', 'accruals'
    ]

    def winsorize_series(s, lower_perc=0.01, upper_perc=0.99):
        """对序列做百分位缩尾去极值"""
        low = s.quantile(lower_perc)
        up = s.quantile(upper_perc)
        return s.clip(low, up)

    section_path = "data/section/"
    for file in glob.glob(section_path + "*.csv"):
        df = pd.read_csv(file)
        columns = df.columns

        # 合并行业信息
        df = df.merge(stock_info_df[["ts_code", "industry"]], on='ts_code', how='left')

        # 先剔除下极端值
        for fac in should_neutralize:
            if fac in df.columns:
                df[fac] = winsorize_series(df[fac])
        # 中性化
        neutralized_cols = {}
        for fac in should_neutralize:
            if fac not in df.columns:
                continue
            neutralized_col = fac + '_neutral'
            neutralized_cols[neutralized_col] = neutralize_one_day(
                df,
                factor_name=fac,
                mv_col='total_mv',
                ind_col='industry'
            )

        if neutralized_cols:
            df = pd.concat([df, pd.DataFrame(neutralized_cols, index=df.index)], axis=1)

        columns = df.columns
        need_standardize_columns = [items for items in columns.tolist() if items not in basic_column_list]
        new_columns = {}  # 用来攒所有新增的标准化列

        for column in need_standardize_columns:
            if ("_standard" not in column
                    and column + "_neutral" not in need_standardize_columns
                    and "_neutral" not in column):
                new_col_name = column + "_standard"
                new_columns[new_col_name] = (df[column] - df[column].mean()) / df[column].std()
            elif "_neutral" in column:
                new_col_name = column.replace("_neutral", "_standard")
                new_columns[new_col_name] = (df[column] - df[column].mean()) / df[column].std()

        # 一次性将所有新列转为 DataFrame 并横向合并
        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        df.to_csv(file, index=False)
        print("标准化", str(file))

def section_duplicates():
    path = "data/section/"
    for file in glob.glob(path + "*.csv"):
        df = pd.read_csv(file)
        df.drop_duplicates(subset="ts_code", keep='first', inplace=True, ignore_index=False)
        df.to_csv(file, index=False)


if __name__ == '__main__':
    # series_to_section()
    # standardize()
    section_duplicates()