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
    将每日截面数据（行情、复权因子、基本面快照、融资融券、资金流、龙虎榜）按
    (ts_code, trade_date) 合并为宽表，增量追加到 data/final_result.csv。

    增量逻辑：读取 final_result.csv 中已有的最大 trade_date，
    只加载各 raw 目录中日期更新的文件，合并后 append 并去重写回。
    """
    import os

    basic_daily_data_list = [
        "stock_data",
        "adj_factor",
        "daily_basic_data",
        "margin_detail",
        "moneyflow",
        "top_list",
    ]

    out_path = "data/final_result.csv"

    # 确定增量起点
    existing_df = None
    max_existing_date = None
    existing_codes: set = set()
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path, dtype={'trade_date': str})
        if 'trade_date' in existing_df.columns and not existing_df.empty:
            max_existing_date = existing_df['trade_date'].max()
            existing_codes = set(existing_df['ts_code'].unique()) if 'ts_code' in existing_df.columns else set()
            print(f"  final_result.csv 已有数据截止 {max_existing_date}，"
                  f"已有 {len(existing_codes)} 只股票，仅加载新增数据")

    final_result = None
    for section_name in basic_daily_data_list:
        all_daily = []
        for file in glob.glob(raw_data_path + section_name + "/*.csv"):
            # 按日期命名的文件（daily_basic_data/moneyflow/margin_detail/top_list）：
            # 文件内含当日全部股票，不能用文件名日期做整文件跳过——
            # 若 final_result 缺少某些股票，那些股票的历史数据就在这些旧文件里。
            # 只有 stock_data（按股票命名）才能做文件级过滤（basename 非纯数字，不触发此分支）。
            # 日期过滤统一在行级别通过 per-stock 逻辑处理（见下方）。
            df = pd.read_csv(file)
            df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
            all_daily.append(df)

        if not all_daily:
            print(f"  {section_name}：无新增文件，跳过")
            # 增量模式下该 section 没有新数据，用空 DataFrame 占位保持 merge 链不断
            if final_result is not None and max_existing_date is not None:
                continue
            # 全量模式下没有文件则真的跳过
            continue

        daily_df = pd.concat(all_daily, ignore_index=True)

        # stock_data 按股票存储，文件名不是日期，需要按 trade_date 列过滤
        if max_existing_date is not None and 'trade_date' in daily_df.columns:
            daily_df['trade_date'] = daily_df['trade_date'].astype(str)
            if existing_codes:
                # 已存在的股票：只取新日期；新股票：保留全部历史
                known_mask   = daily_df['ts_code'].isin(existing_codes)
                new_date_mask = daily_df['trade_date'] > max_existing_date
                daily_df = daily_df[~known_mask | new_date_mask]
            else:
                daily_df = daily_df[daily_df['trade_date'] > max_existing_date]

        if daily_df.empty:
            print(f"  {section_name}：无新增行，跳过")
            continue

        daily_df = daily_df.set_index(['ts_code', 'trade_date']).sort_index()
        print(f"  合并 {section_name} → 新增 {len(daily_df)} 行")

        if final_result is None:
            final_result = daily_df
        else:
            final_result = pd.merge(final_result, daily_df, how='outer',
                                    left_index=True, right_index=True)
        del daily_df

    if final_result is None:
        print("  无新增数据，final_result.csv 保持不变。")
        return

    # 合并新旧数据，按 (ts_code, trade_date) 去重
    new_df = final_result.reset_index()
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined['trade_date'] = combined['trade_date'].astype(str)
        combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        combined.sort_values(['ts_code', 'trade_date'], inplace=True)
        combined.to_csv(out_path, index=False)
        print(f"  final_result.csv 更新完成，共 {len(combined)} 行")
    else:
        new_df.to_csv(out_path, index=False)
        print(f"  final_result.csv 全量写入，共 {len(new_df)} 行")


def split_to_series_section(input_file_name: str):
    """
    将宽表 final_result.csv 按股票代码拆分，增量追加到 data/series/<ts_code>.csv。
    已存在的股票文件只追加新日期行；新股票直接创建文件。
    """
    import os
    series_path = "data/series/"

    df = pd.read_csv(input_file_name, dtype={'trade_date': str})
    print("列：", df.columns.tolist())

    # 确定每只股票已有的最大日期，只写入更新的行
    for ts_code, group in df.groupby('ts_code'):
        out_file = f"{series_path}{ts_code}.csv"
        if 'trade_date' in group.columns:
            group = group.sort_values('trade_date')

        if os.path.exists(out_file):
            existing = pd.read_csv(out_file, dtype={'trade_date': str})
            max_date = existing['trade_date'].max() if 'trade_date' in existing.columns else None
            if max_date is not None:
                new_rows = group[group['trade_date'] > max_date]
                if new_rows.empty:
                    continue
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined.sort_values('trade_date', inplace=True)
                combined.to_csv(out_file, index=False)
                print(f"  {ts_code} +{len(new_rows)} 行")
                continue

        group.to_csv(out_file, index=False)
        print(f"  {ts_code} 新建")


if __name__ == '__main__':
    financial_data_preprocess()
    merge_basic_daily_data()
    split_to_series_section("data/final_result.csv")
