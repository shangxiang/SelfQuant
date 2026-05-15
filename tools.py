import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
import os


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

    stock_info_df = pd.read_csv("data/raw/stock_list/stock_list.csv")

    # 不参与标准化的原始列：仅保留标识列、收益率和二值信号
    # 凡是进入 should_neutralize 的字段均不在此处重复声明
    basic_column_list = [
        # 标识与原始行情
        'ts_code', 'trade_date', 'name', 'reason',
        'pct_chg', 'pct_change', 'industry',
        # 二值/离散信号：z-score 对 0/1 无意义
        'macd_air_refuel', 'macd_divergence', 'vol_breakout',
    ]

    # 需要先行业+市值中性化的因子
    # 注意：size_factor / smb_squared 本身就是市值变换，不在此列表
    should_neutralize = [
        # 估值类（行业间 PB/PE 差异极大）
        'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
        # 规模类（绝对值受市值影响）
        'total_share', 'float_share', 'free_share',
        'vol', 'amount_x', 'amount_y',
        'turnover_rate_x', 'turnover_rate_f', 'turnover_rate_y', 'volume_ratio',
        # 融资融券
        'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye',
        # 资金流（大中小单）
        'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
        'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
        'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
        'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
        'net_mf_vol', 'net_mf_amount',
        # 龙虎榜
        'l_sell', 'l_buy', 'l_amount', 'net_amount', 'net_rate', 'amount_rate',
        # 技术量价因子（含行业/市值偏差）
        'positive_flow', 'negative_flow', 'mfi',
        'raw_force_index', 'force_index_smoothed', 'force_index', 'vwap',
        'mtm_margin_balance_change', 'big_order_ratio', 'lhb_strength_5d',
        'volatility_20d', 'reversal_5d', 'high_low_spread',
        # 基本面因子
        'gross_margin', 'debt_ratio', 'roe_ttm',
        'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
        # 新增：FF 风格因子（行业/市值偏差显著）
        'asset_growth_yoy',   # 资产增速在重/轻资产行业差异大
        'value_factor',       # 1/PB：金融/科技行业天然差异
        'cma_factor',         # 资产增速取反，同 asset_growth_yoy
        'momentum_12_1',      # 行业轮动带来动量的行业偏差
        # 新增：高阶交叉因子（含动量/估值分量，需中性化）
        'smb_mom',            # 小盘动量：动量分量含行业偏差
        'smb_squared_mom',    # 规模²×动量：同上
        'hml_rmw',            # 估值×盈利：估值分量含行业偏差
        'smb_hml',            # 规模×估值：估值分量含行业偏差
        'vol_mom',            # 波动率×动量：两者均含行业/市值偏差
    ]

    def winsorize_series(s, lower_perc=0.01, upper_perc=0.99):
        """1%~99% 百分位缩尾，压制极端值对标准化的影响"""
        return s.clip(s.quantile(lower_perc), s.quantile(upper_perc))

    section_path = "data/section/"
    for file in glob.glob(section_path + "*.csv"):
        df = pd.read_csv(file)

        # 合并行业信息（用于中性化）
        df = df.merge(stock_info_df[["ts_code", "industry"]], on='ts_code', how='left')

        # 步骤 1：缩尾
        for fac in should_neutralize:
            if fac in df.columns:
                df[fac] = winsorize_series(df[fac])

        # 步骤 2：中性化
        neutralized_cols = {}
        for fac in should_neutralize:
            if fac not in df.columns:
                continue
            neutralized_cols[fac + '_neutral'] = neutralize_one_day(
                df, factor_name=fac, mv_col='total_mv', ind_col='industry'
            )
        if neutralized_cols:
            df = pd.concat([df, pd.DataFrame(neutralized_cols, index=df.index)], axis=1)

        # 步骤 3：Z-score 标准化
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
                # 中性化列 → 标准化后去掉 _neutral 后缀
                new_columns[col.replace('_neutral', '_standard')] = (df[col] - df[col].mean()) / std
            elif col + '_neutral' not in columns:
                # 无对应中性化版本 → 直接标准化原始列
                new_columns[col + '_standard'] = (df[col] - df[col].mean()) / std

        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        df.to_csv(file, index=False)
        print("标准化", str(file))


def section_duplicates():
    """清理截面文件中同一交易日内重复的 ts_code 行。"""
    path = "data/section/"
    for file in glob.glob(path + "*.csv"):
        df = pd.read_csv(file)
        df.drop_duplicates(subset="ts_code", keep='first', inplace=True, ignore_index=True)
        df.to_csv(file, index=False)


if __name__ == '__main__':
    # series_to_section()
    # standardize()
    section_duplicates()
