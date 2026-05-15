import pandas as pd
import glob
import numpy as np
from typing import Optional

raw_data_path = "data/raw/"

# ---- 利润表：需要做累计→单季度转换的列 ----
# 只保留因子计算所需的核心指标，剔除金融行业专项科目和分配明细等不用的列
income_quarterly_columns = [
    'total_revenue',       # 营业总收入（用于毛利率、净利率、营收同比）
    'oper_cost',           # 营业成本（用于毛利率 = (营收-成本)/营收）
    'n_income',           # 净利润（Tushare 字段名，用于净利率、同比增长）
    'n_income_attr_p',     # 归母净利润（用于归母 ROE、同比增长）
    'rd_exp',              # 研发费用
]

# ---- 现金流量表：需要做累计→单季度转换的列 ----
cashflow_quarterly_columns = [
    'n_cashflow_act',   # 经营活动现金流净额（反映主营业务造血能力）
    'free_cashflow',    # 自由现金流（= 经营现金流 - 资本开支）
]

# ---- 资产负债表保留列（时点数据，不做季度化）----
# ts_code / f_ann_date / end_date / ann_date / end_type 是合并和去重的必要键，不在此列出
BALANCESHEET_KEEP_COLS = [
    'total_assets',                  # 总资产（ROE 分母、负债率分母）
    'total_liab',                    # 总负债（负债率分子）
    'total_hldr_eqy_exc_min_int',    # 归母股东权益（ROE 分母）
    'money_cap',                     # 货币资金（应计项目计算用）
    'st_borr',                       # 短期借款
    'lt_borr',                       # 长期借款
]

# 合并/去重所需的键列（三张表通用）
_KEY_COLS = ['ts_code', 'f_ann_date', 'end_date', 'ann_date', 'end_type']


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理重复行：相同 (ts_code, f_ann_date, end_type) 只保留最佳行。
    最佳行定义：优先 ann_date 最新，次之非空数值列数最多。
    同一报告期的数据可能因更正重新披露，此步骤保留最新最完整的版本。
    """
    df = df.copy()
    # 统计每行非空数值列数，作为完整度的代理指标
    df['_non_null_count'] = df.select_dtypes(include=[np.number]).count(axis=1)

    def keep_best_group(g):
        g = g.sort_values(
            by=['ann_date', '_non_null_count'],
            ascending=[False, False],
            na_position='last'
        )
        return g.head(1)

    df_clean = (
        df.groupby(['ts_code', 'f_ann_date', 'end_type'], group_keys=False)
        .apply(keep_best_group)
        .drop(columns=['_non_null_count'])
        .reset_index(drop=True)
    )
    return df_clean


def convert_to_quarterly(df: pd.DataFrame, quarterly_cols: list) -> pd.DataFrame:
    """
    将利润表/现金流量表的累计季度值还原为单季度值。

    Tushare 财报数据以累计口径披露（如 Q3 = Q1+Q2+Q3），
    回测时需要还原为单季度数值以计算同比、环比等指标。
    规则：第一季度（3月末）保持不变；Q2/Q3/Q4 = 当期累计 - 上季度累计（同年内）。

    Parameters
    ----------
    df            : pd.DataFrame  财报数据（含 ts_code、end_date、目标列）
    quarterly_cols: list          需要做季度化的列名列表
    """
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y%m%d')
    df['year']  = df['end_date'].dt.year
    df['month'] = df['end_date'].dt.month
    df['end_date'] = df['end_date'].dt.strftime('%Y%m%d')

    # 季度顺序：3月=Q1、6月=Q2、9月=Q3、12月=Q4
    df['quarter_order'] = df['month'].map({3: 1, 6: 2, 9: 3, 12: 4})
    df = df.sort_values(['ts_code', 'year', 'quarter_order'])

    def compute_single_quarter(group):
        group = group.sort_values(['year', 'quarter_order'])
        for col in quarterly_cols:
            group[col] = pd.to_numeric(group[col], errors='coerce')
            prev_val = group[col].shift(1)
            # 同年且非第一季度：单季 = 当期累计 - 上季累计
            group.loc[:, col] = np.where(
                (group['quarter_order'] > 1) & (group['year'] == group['year'].shift(1)),
                group[col] - prev_val,
                group[col]
            )
        return group

    df = df.groupby('ts_code', group_keys=False).apply(compute_single_quarter)
    df.drop(columns=['year', 'month', 'quarter_order'], inplace=True)
    return df


def financial_data_preprocess():
    """
    合并三张财务报表（资产负债表、利润表、现金流量表），输出 data/financial.csv。

    只保留因子计算所需的核心列，大幅减少内存占用和磁盘空间。
    三表以 (ts_code, f_ann_date, end_date) 为键左连接，以资产负债表为主表。
    """

    # ---- 1. 资产负债表（时点数据，不做季度化）----
    keep_bs = _KEY_COLS + [c for c in BALANCESHEET_KEEP_COLS]
    all_bs = []
    for file in glob.glob(raw_data_path + "balancesheet/*.csv"):
        df = pd.read_csv(file)
        df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
        # 只取存在的列，避免因 Tushare 版本差异报 KeyError
        cols = [c for c in keep_bs if c in df.columns]
        df = clean_duplicates(df)[cols]
        all_bs.append(df)

    balancesheet_df = pd.concat(all_bs, ignore_index=True)
    # end_date 统一为 YYYYMMDD 字符串（merge 键需类型一致）
    balancesheet_df['end_date'] = pd.to_datetime(
        balancesheet_df['end_date'], format='%Y%m%d'
    ).dt.strftime('%Y%m%d')
    balancesheet_df.to_csv("data/balancesheet.csv", index=False)
    final_df = balancesheet_df

    # ---- 2. 利润表（累计→单季度）----
    keep_inc = _KEY_COLS + income_quarterly_columns
    all_inc = []
    for file in glob.glob(raw_data_path + "income/*.csv"):
        df = pd.read_csv(file)
        df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
        df = clean_duplicates(df)
        # 季度化只对实际存在的列操作
        q_cols = [c for c in income_quarterly_columns if c in df.columns]
        df = convert_to_quarterly(df, quarterly_cols=q_cols)
        cols = [c for c in keep_inc if c in df.columns]
        all_inc.append(df[cols])

    income_df = pd.concat(all_inc, ignore_index=True)
    income_df.to_csv("data/income.csv", index=False)
    final_df = pd.merge(final_df, income_df, how='left',
                        on=['ts_code', 'f_ann_date', 'end_date'])

    # ---- 3. 现金流量表（累计→单季度）----
    keep_cf = _KEY_COLS + cashflow_quarterly_columns
    all_cf = []
    for file in glob.glob(raw_data_path + "cashflow/*.csv"):
        df = pd.read_csv(file)
        df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
        df = clean_duplicates(df)
        q_cols = [c for c in cashflow_quarterly_columns if c in df.columns]
        df = convert_to_quarterly(df, quarterly_cols=q_cols)
        cols = [c for c in keep_cf if c in df.columns]
        all_cf.append(df[cols])

    cashflow_df = pd.concat(all_cf, ignore_index=True)
    cashflow_df.to_csv("data/cashflow.csv", index=False)
    final_df = pd.merge(final_df, cashflow_df, how='left',
                        on=['ts_code', 'f_ann_date', 'end_date'])

    print("financial.csv 列：", final_df.columns.tolist())
    final_df.to_csv("data/financial.csv", index=False)


def merge_basic_daily_data():
    """
    将每日截面数据（行情、基本面快照、融资融券、资金流、龙虎榜）按
    (ts_code, trade_date) 合并为宽表，按股票拆分后保存到 data/series/。
    """
    basic_daily_data_list = [
        "stock_data",
        "daily_basic_data",
        "margin_detail",
        "moneyflow",
        "top_list",
    ]
    final_result = None
    for section_name in basic_daily_data_list:
        all_daily = []
        for file in glob.glob(raw_data_path + section_name + "/*.csv"):
            df = pd.read_csv(file)
            df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
            all_daily.append(df)

        if not all_daily:
            print(f"  跳过 {section_name}（无文件）")
            continue

        daily_df = pd.concat(all_daily, ignore_index=True)
        daily_df = daily_df.set_index(['ts_code', 'trade_date']).sort_index()
        daily_df.to_csv(f"data/series/{section_name}.csv")
        print(f"合并 {section_name} → {len(daily_df)} 行")

        if final_result is None:
            final_result = daily_df
        else:
            final_result = pd.merge(final_result, daily_df, how='left',
                                    on=['ts_code', 'trade_date'])
        del daily_df

    if final_result is not None:
        final_result.to_csv("data/final_result.csv")


def split_to_series_section(input_file_name: str):
    """
    将宽表 final_result.csv 按股票代码拆分，每只股票保存为独立的时序 CSV。
    输出到 data/series/<ts_code>.csv，供 FactorManager 逐只计算因子。
    """
    series_path = "data/series/"
    df = pd.read_csv(input_file_name)
    print("列：", df.columns.tolist())
    for ts_code, group in df.groupby('ts_code'):
        if 'trade_date' in group.columns:
            group = group.sort_values('trade_date')
        group.to_csv(f"{series_path}{ts_code}.csv", index=False)
        print(f"  已保存 {ts_code}")


if __name__ == '__main__':
    financial_data_preprocess()
    merge_basic_daily_data()
    split_to_series_section("data/final_result.csv")
