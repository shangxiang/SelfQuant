import os
import glob
import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
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


def _read_csv_safe(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    读取单个 CSV。文件为空或解析失败时返回 None。

    Tushare 对"无记录"的股票会写出只含一个换行符的文件（namechange 下有 700+ 个），
    直接 pd.read_csv 会抛 EmptyDataError 打断整条流程。
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 10:
            return None
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"  跳过无法解析的文件 {os.path.basename(path)}: {type(e).__name__}: {e}")
        return None


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
    合并三张财务报表（资产负债表、利润表、现金流量表），输出 data/financial.parquet。

    只保留因子计算所需的核心列，大幅减少内存占用和磁盘空间。
    三表以 (ts_code, f_ann_date, end_date) 为键左连接，以资产负债表为主表。
    """

    # ---- 1. 资产负债表（时点数据，不做季度化）----
    keep_bs = _KEY_COLS + [c for c in BALANCESHEET_KEEP_COLS]
    all_bs = []
    for file in glob.glob(raw_data_path + "balancesheet/*.csv"):
        df = _read_csv_safe(file)
        if df is None:
            continue
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
    balancesheet_df.to_parquet("data/balancesheet.parquet", index=False)
    final_df = balancesheet_df

    # ---- 2. 利润表（累计→单季度）----
    keep_inc = _KEY_COLS + income_quarterly_columns
    all_inc = []
    for file in glob.glob(raw_data_path + "income/*.csv"):
        df = _read_csv_safe(file)
        if df is None:
            continue
        df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
        df = clean_duplicates(df)
        # 季度化只对实际存在的列操作
        q_cols = [c for c in income_quarterly_columns if c in df.columns]
        df = convert_to_quarterly(df, quarterly_cols=q_cols)
        cols = [c for c in keep_inc if c in df.columns]
        all_inc.append(df[cols])

    income_df = pd.concat(all_inc, ignore_index=True)
    income_df.to_parquet("data/income.parquet", index=False)
    final_df = pd.merge(final_df, income_df, how='left',
                        on=['ts_code', 'f_ann_date', 'end_date'])

    # ---- 3. 现金流量表（累计→单季度）----
    keep_cf = _KEY_COLS + cashflow_quarterly_columns
    all_cf = []
    for file in glob.glob(raw_data_path + "cashflow/*.csv"):
        df = _read_csv_safe(file)
        if df is None:
            continue
        df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
        df = clean_duplicates(df)
        q_cols = [c for c in cashflow_quarterly_columns if c in df.columns]
        df = convert_to_quarterly(df, quarterly_cols=q_cols)
        cols = [c for c in keep_cf if c in df.columns]
        all_cf.append(df[cols])

    cashflow_df = pd.concat(all_cf, ignore_index=True)
    cashflow_df.to_parquet("data/cashflow.parquet", index=False)
    final_df = pd.merge(final_df, cashflow_df, how='left',
                        on=['ts_code', 'f_ann_date', 'end_date'])

    print("financial.parquet 列：", final_df.columns.tolist())
    final_df.to_parquet("data/financial.parquet", index=False)


# ---- 增量更新相关路径 ----
NAME_STATE_PATH   = "data/name_state.csv"        # 每只股票最近一次记录的名称 + ST 状态
ST_FLAG_PATH      = "data/st_flag.parquet"       # (ts_code, trade_date, is_st)
FINAL_RESULT_PATH = "data/final_result.parquet"  # 日线宽表
NEW_CHUNK_PATH    = "data/_final_result_new.parquet"  # 增量合并产出的新增段（供拆分用）


def build_st_flag_from_namechange():
    """
    【历史口径，仅供首次全量建库】从 namechange 数据构建 ST 标识表。

    增量更新不再走这里（不调用 get_namechange 接口）：
    历史区间用一次全量建表打底，之后由 build_st_flag() 基于 stock_list 逐日追加。

    返回 DataFrame 包含列：ts_code, trade_date, is_st
    其中 is_st = 1 表示该股票在该日期是ST股，否则为 0。
    """
    # 读取所有 namechange 数据
    all_namechange = []
    for file in glob.glob(raw_data_path + "namechange/*.csv"):
        df = _read_csv_safe(file)
        if df is None:
            continue
        df.drop(columns='Unnamed: 0', errors='ignore', inplace=True)
        all_namechange.append(df)
    
    if not all_namechange:
        print("  namechange：无数据，跳过 ST 标识构建")
        return None
    
    namechange_df = pd.concat(all_namechange, ignore_index=True)
    
    # 筛选包含 ST 的记录（name 字段包含 "ST"）
    st_records = namechange_df[namechange_df['name'].str.contains('ST', case=False, na=False)].copy()

    if st_records.empty:
        print("  namechange：无 ST 记录，跳过 ST 标识构建")
        return None

    # 解析日期
    st_records['start_date'] = pd.to_datetime(st_records['start_date'], format='%Y%m%d', errors='coerce')
    st_records['end_date'] = pd.to_datetime(st_records['end_date'], format='%Y%m%d', errors='coerce')
    st_records = st_records.dropna(subset=['start_date'])

    # 读取交易日历，获取所有交易日
    trade_cal_path = raw_data_path + "trade_cal.csv"
    if not os.path.exists(trade_cal_path):
        print("  交易日历不存在，跳过 ST 标识构建")
        return None

    trade_cal = pd.read_csv(trade_cal_path, dtype={'cal_date': str})
    trade_days = sorted(trade_cal[trade_cal['is_open'] == 1]['cal_date'].dropna().tolist())

    # 向量化生成 ST 标识：对每个 (股票, 区间) 直接在整个交易日序列上做比较，
    # 避免「股票 × 交易日 × 区间」三层 Python 循环（原本约千万次迭代）。
    day_idx = pd.to_datetime(pd.Series(trade_days), format='%Y%m%d')
    day_arr = day_idx.values
    day_str = np.array(trade_days, dtype=object)

    frames = []
    for ts_code, group in st_records.groupby('ts_code'):
        mask = np.zeros(len(day_arr), dtype=bool)
        for start, end in zip(group['start_date'].values, group['end_date'].values):
            hit = day_arr >= np.datetime64(start)
            if not pd.isna(end):
                hit &= day_arr <= np.datetime64(end)
            mask |= hit
        if not mask.any():
            continue
        frames.append(pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': day_str,
            'is_st': mask.astype(int),
        }))

    if not frames:
        print("  namechange：无命中交易日的 ST 记录")
        return None

    st_flag_df = pd.concat(frames, ignore_index=True)
    st_flag_df.to_parquet(ST_FLAG_PATH, index=False)
    print(f"  st_flag.parquet 生成完成，共 {len(st_flag_df)} 行，其中 ST 记录 {st_flag_df['is_st'].sum()} 行")
    return st_flag_df


# ------------------------------------------------------------------ #
#  ST 标识：基于 stock_list 的增量维护                                  #
# ------------------------------------------------------------------ #

def is_main_board(ts_code: str) -> bool:
    """
    沪深主板判断（全项目唯一口径）：
      - 沪市：60xxxx / 601 / 603 / 605，排除 688（科创板）
      - 深市：000 / 001 / 002 / 003，排除 300 / 301（创业板）
        （002/003 原中小板已于 2021 年并入深市主板）
      - 北交所 BJ 及其他一律排除

    下载、宽表合并、ST 标识三处都用它，避免数据源口径变化后混入非主板股票。
    """
    code, exchange = str(ts_code)[:6], str(ts_code)[-2:]
    if exchange == 'SH':
        return code.startswith('6') and not code.startswith('688')
    if exchange == 'SZ':
        return code.startswith('0') and not code.startswith('3')
    return False


def _is_st_name(name) -> bool:
    """名称中带 ST / *ST 即视为 ST 股（与 namechange 版的判定规则保持一致）。"""
    return 'ST' in str(name).upper()


def _trade_days() -> list:
    """交易日历中的全部交易日（升序 YYYYMMDD 字符串）。"""
    cal_path = raw_data_path + "trade_cal.csv"
    if not os.path.exists(cal_path):
        return []
    cal = pd.read_csv(cal_path, dtype={'cal_date': str})
    return sorted(cal.loc[cal['is_open'] == 1, 'cal_date'].dropna().tolist())


def _st_flag_last_date(path: str = None):
    """已有 ST 标识表的最新交易日；表不存在时返回 None。"""
    path = path or ST_FLAG_PATH
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=['trade_date'])
        s = df['trade_date'].astype(str)
        return str(s.max()) if len(s) else None
    except Exception as e:
        print(f"  读取 {path} 失败：{e}")
        return None


def _load_name_state(path: str = None) -> pd.DataFrame:
    """
    读取名称状态表（ts_code, name, is_st, last_date）。
    文件不存在或损坏时返回空表（调用方据此做初始化）。
    """
    path = path or NAME_STATE_PATH
    cols = ['ts_code', 'name', 'is_st', 'last_date']
    if os.path.exists(path):
        try:
            st = pd.read_csv(path, dtype={'ts_code': str, 'name': str, 'last_date': str})
            for c in cols:
                if c not in st.columns:
                    st[c] = np.nan
            return st[cols]
        except Exception as e:
            print(f"  名称状态文件读取失败，将重新初始化：{e}")
    return pd.DataFrame(columns=cols)


def _bootstrap_name_state() -> pd.DataFrame:
    """
    首次运行时没有名称状态文件，用本地已下载的 namechange CSV 初始化
    「上一次记录的名称」（每只股票取 start_date 最大的一条）。

    只读本地文件，不调用 get_namechange 接口；历史区间之后的变更由
    build_st_flag() 基于 stock_list 逐日比对补齐。
    """
    frames = []
    for file in glob.glob(raw_data_path + "namechange/*.csv"):
        df = _read_csv_safe(file)
        if df is None or 'name' not in df.columns:
            continue
        df = df.dropna(subset=['name'])
        if df.empty:
            continue
        cols = [c for c in ('ts_code', 'name', 'start_date') if c in df.columns]
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=['ts_code', 'name', 'is_st', 'last_date'])

    all_nc = pd.concat(frames, ignore_index=True)
    if 'start_date' not in all_nc.columns:
        all_nc['start_date'] = ''
    all_nc['start_date'] = all_nc['start_date'].astype(str)
    all_nc = (all_nc.sort_values(['ts_code', 'start_date'])
                    .drop_duplicates('ts_code', keep='last'))
    out = all_nc[['ts_code', 'name']].copy()
    out['is_st'] = out['name'].map(lambda n: int(_is_st_name(n)))
    out['last_date'] = ''
    return out[['ts_code', 'name', 'is_st', 'last_date']]


def build_st_flag(dates=None, end_date: str = None, start_date: str = None):
    """
    用 stock_list 的当前名称增量维护 ST 标识表 data/st_flag.parquet。

    与旧口径（build_st_flag_from_namechange）的区别：
      - 不调用 get_namechange 接口，也不依赖 data/raw/namechange 的日更；
      - 判断依据是「本地记录的上一次名称」与 stock_list 当前名称是否一致，
        不一致即视为发生了变更（戴帽 / 摘帽），据此更新 is_st；
      - 只在表尾追加新的交易日，已落盘的历史行原样保留、绝不清理。

    每日收盘后跑一次即可保证名称信息连续：当天改名 → 当天即被记录。

    Parameters
    ----------
    dates      : list[str] | None  直接指定要写入的交易日（优先）
    end_date   : str | None        增量到该日为止（含）；None 时取今天
    start_date : str | None        起始日（含）；None 时取已有最新日期的下一个交易日

    Returns
    -------
    DataFrame | None  本次新增的行（ts_code, trade_date, is_st）；无新增返回 None
    """
    sl_path = raw_data_path + "stock_list/stock_list.csv"
    if not os.path.exists(sl_path):
        print("  stock_list.csv 不存在，跳过 ST 标识更新")
        return None

    cur = pd.read_csv(sl_path, usecols=['ts_code', 'name'], dtype=str)
    cur = cur.drop_duplicates('ts_code').reset_index(drop=True)
    n_all = len(cur)
    # 与宽表同一口径：只记沪深主板，增量更新也不会把非主板拉进来
    cur = cur[cur['ts_code'].map(is_main_board)].reset_index(drop=True)
    if n_all != len(cur):
        print(f"  股票范围：沪深主板 {len(cur)} 只（stock_list 共 {n_all} 只，已剔除 {n_all - len(cur)} 只非主板）")
    cur['is_st'] = cur['name'].map(lambda n: int(_is_st_name(n)))

    # ---- 目标交易日 ----
    last = _st_flag_last_date()
    if last is None:
        # 完全没有历史底表：本地若有 namechange 原始数据，先用它全量建一次
        hist = build_st_flag_from_namechange()
        if hist is not None and not hist.empty:
            last = _st_flag_last_date()

    if dates is not None:
        targets = sorted({str(d) for d in dates})
    else:
        trade_days = _trade_days()
        if start_date:
            lo = start_date
        elif last:
            later = [d for d in trade_days if d > last]
            lo = later[0] if later else None
        else:
            lo = trade_days[0] if trade_days else None
        hi = end_date or pd.Timestamp.today().strftime('%Y%m%d')
        targets = [d for d in trade_days if lo and lo <= d <= hi] if lo else []

    # ---- 与本地上一次记录的名称对比 ----
    state = _load_name_state()
    if state.empty:
        state = _bootstrap_name_state()
        if not state.empty:
            print(f"  名称状态初始化：从本地 namechange 载入 {len(state)} 只股票的上一次名称")

    prev = state.drop_duplicates('ts_code', keep='last').set_index('ts_code')
    cur['prev_name'] = cur['ts_code'].map(prev['name'])
    cur['prev_is_st'] = cur['ts_code'].map(prev['is_st']).fillna(0).astype(int)

    changed = cur[cur['prev_name'].notna() & (cur['prev_name'] != cur['name'])]
    if len(changed):
        print(f"  检测到 {len(changed)} 只股票名称变更（前 20 条）：")
        for _, r in changed.head(20).iterrows():
            print(f"    {r['ts_code']}  {r['prev_name']} → {r['name']}")

    on  = cur[(cur['prev_is_st'] == 0) & (cur['is_st'] == 1)]
    off = cur[(cur['prev_is_st'] == 1) & (cur['is_st'] == 0)]
    if len(on):
        print(f"  戴上 ST 帽 {len(on)} 只：{'、'.join(on['ts_code'].head(10))}"
              f"{' …' if len(on) > 10 else ''}")
    if len(off):
        print(f"  摘掉 ST 帽 {len(off)} 只：{'、'.join(off['ts_code'].head(10))}"
              f"{' …' if len(off) > 10 else ''}")

    # ---- 名称状态先落盘 ----
    # 即使本次没有新交易日也要刷新：状态里存的就是"上一次看到的名称"，
    # 下次更新时才能比对出改名（戴帽/摘帽）。漏记一次就会漏掉一次变更。
    new_state = cur[['ts_code', 'name', 'is_st']].copy()
    if targets:
        new_state['last_date'] = targets[-1]
    else:
        prev_last = prev['last_date'].to_dict() if len(prev) else {}
        new_state['last_date'] = new_state['ts_code'].map(prev_last)
    keep_old = state[~state['ts_code'].isin(new_state['ts_code'])]
    pd.concat([keep_old, new_state], ignore_index=True).to_csv(NAME_STATE_PATH, index=False)

    if not targets:
        print("  ST 标识已是最新，跳过（名称状态已刷新）。")
        return None

    # ---- 生成新增行：每只股票 × 每个新交易日 ----
    n_code, n_day = len(cur), len(targets)
    new_rows = pd.DataFrame({
        'ts_code':    np.repeat(cur['ts_code'].to_numpy(), n_day),
        'trade_date': np.tile(np.array(targets, dtype=object), n_code),
        'is_st':      np.repeat(cur['is_st'].to_numpy(dtype='int32'), n_day),
    })

    # ---- 追加写回：旧行按 (ts_code, trade_date) 去重，新值优先 ----
    if os.path.exists(ST_FLAG_PATH):
        old = pd.read_parquet(ST_FLAG_PATH)
        old['trade_date'] = old['trade_date'].astype(str)
        old = old[['ts_code', 'trade_date', 'is_st']]
        key_old = pd.MultiIndex.from_arrays([old['ts_code'], old['trade_date']])
        key_new = pd.MultiIndex.from_arrays([new_rows['ts_code'], new_rows['trade_date']])
        old = old[~key_old.isin(key_new)]
        all_df = pd.concat([old, new_rows], ignore_index=True)
    else:
        all_df = new_rows

    all_df = all_df.drop_duplicates(['ts_code', 'trade_date'], keep='last')
    all_df = all_df.sort_values(['ts_code', 'trade_date'])
    all_df.to_parquet(ST_FLAG_PATH, index=False)

    n_st = int(new_rows['is_st'].sum())
    print(f"  ST 标识追加 {len(targets)} 个交易日 × {n_code} 只 = {len(new_rows)} 行"
          f"（其中 ST {n_st} 行）；文件共 {len(all_df)} 行")
    return new_rows


def _view_columns(con, view: str) -> list:
    """取视图列名，剔除 CSV 里常见的 Unnamed 列。"""
    df = con.execute(f"DESCRIBE {view}").df()
    return [c for c in df['column_name'].tolist() if not str(c).lower().startswith('unnamed')]


def build_join_sql(con, views: list, key=("ts_code", "trade_date"), base: str = None) -> str:
    """
    把多个按 (ts_code, trade_date) 组织的数据源全外连接成一条 SQL。

    重名列沿用 pandas merge 的规则：左表加 _x、右表加 _y
    （下游 standardize.py 里就有 amount_x / turnover_rate_y 这类引用）。

    注意 USING 的两侧都必须显式选出键列，否则 DuckDB 报
    "Referenced column ... not found"。

    Parameters
    ----------
    views : list[str]  实际视图名（如 v_stock_data）
    base  : str        主表视图名，None 时优先用 v_stock_data
    """
    base = base or ('v_stock_data' if 'v_stock_data' in views else views[0])
    others = [v for v in views if v != base]

    out_cols = list(key) + [c for c in _view_columns(con, base) if c not in key]
    sql = "SELECT " + ', '.join(f'"{c}"' for c in out_cols) + f" FROM {base}"

    for v in others:
        new_cols = [c for c in _view_columns(con, v) if c not in key]
        # 左：键列原样输出；与右表重名的列改名为 <col>_x
        left_items = [(c, c) for c in key] + [
            (c, c + '_x' if c in new_cols else c) for c in out_cols if c not in key]
        # 右：键列原样输出；与左表重名的列改名为 <col>_y
        right_items = [(c, c) for c in key] + [
            (c, c + '_y' if c in out_cols else c) for c in new_cols]

        left_sql = "SELECT " + ', '.join(
            f'l."{c}" AS "{n}"' for c, n in left_items) + f" FROM ({sql}) l"
        right_sql = "SELECT " + ', '.join(
            f'r."{c}" AS "{n}"' for c, n in right_items) + f" FROM {v} r"

        left_names = [n for _, n in left_items if n not in key]
        right_names = [n for _, n in right_items if n not in key]
        sel = [f'"{c}"' for c in key] \
            + [f'l."{n}" AS "{n}"' for n in left_names] \
            + [f'r."{n}" AS "{n}"' for n in right_names]
        sql = (f"SELECT {', '.join(sel)} FROM ({left_sql}) l "
               f"FULL OUTER JOIN ({right_sql}) r USING ({', '.join(key)})")
        out_cols = list(key) + left_names + right_names

    return sql


def resolve_since(since: str = None, refresh_last_days: int = 0,
                  out_path: str = FINAL_RESULT_PATH):
    """
    把 since / refresh_last_days 解析成实际的增量起点（不含该日）。

    - since 为空 → 取已有宽表的最新交易日
    - refresh_last_days > 0 → 在此基础上再往前回退 N 个交易日重做
    - 宽表不存在 → 返回 None（调用方按全量构建处理）
    """
    if since is None:
        since = _final_result_last_date(out_path)
        if since is None:
            print("  未找到已有 final_result.parquet，转为全量构建。")
            return None
        print(f"  增量起点：已有宽表最新交易日 {since}（只处理其后的新日期）")
        if refresh_last_days > 0:
            earlier = [d for d in _trade_days() if d < since]
            if earlier:
                since = earlier[-min(refresh_last_days, len(earlier))]
                print(f"  回退重做最后 {refresh_last_days} 个交易日 → 起点 {since}")
    return since


def merge_basic_daily_data(incremental: bool = False, since: str = None,
                           refresh_last_days: int = 0):
    """
    将每日截面数据（行情、复权因子、基本面快照、融资融券、资金流、龙虎榜、ST 标识）按
    (ts_code, trade_date) 合并为宽表，输出 data/final_result.parquet。

    改用 DuckDB 执行而非 pandas：宽表约 677 万行 × 80 列，pandas 需要在内存里
    同时驻留多个中间 DataFrame（数 GB 量级），16GB 机器上很容易 OOM；
    DuckDB 内存不足时会溢写磁盘，并行解析两万多个 CSV 也快得多。

    重名列（如 amount、turnover_rate）沿用 pandas merge 的 _x / _y 后缀规则，
    以兼容下游 standardize.py 中已有的列名引用。

    Parameters
    ----------
    incremental : bool
        True 时只处理 trade_date > since 的新交易日：
        新增段先落到 data/_final_result_new.parquet，再整体并入已有宽表。
        已有行（trade_date <= since）原样保留，绝不清理。
    since : str | None
        增量起点（不含）。None 时取已有宽表的最新交易日。
        想重做某段历史时，把它设成那段历史的前一天即可（该日之后的行会被新值替换）。
    refresh_last_days : int
        在自动算出的起点上再往前回退 N 个交易日重做。用于"上次更新时当天数据
        还不完整"的场景（比如下午三点刚过就跑了更新），默认 0 表示不回退。

    Returns
    -------
    str | None  增量模式下返回新增段文件路径（供 split_to_series_section 直接拆分）；
                无新数据返回 None；全量模式返回 None。
    """
    basic_daily_data_list = [
        "stock_data",
        "adj_factor",
        "daily_basic_data",
        "margin_detail",
        "moneyflow",
        "top_list",
    ]
    # 按交易日分文件存放的数据源：增量时可以只挑新日期的文件，省掉大量 CSV 解析
    DATE_FILED_SECTIONS = {"daily_basic_data", "margin_detail", "moneyflow", "top_list"}
    out_path = FINAL_RESULT_PATH
    key = ("ts_code", "trade_date")
    # 每轮连接的数据源个数：分批物化中间结果，把峰值内存压下来
    join_group_size = 3

    # ---- 0. 增量起点 ----
    since = resolve_since(since, refresh_last_days, out_path)
    incremental = bool(since)

    con = duckdb.connect()
    new_chunk = None
    try:
        # 允许溢写磁盘 + 降低并行度，避免 16GB 机器上 OOM
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=4")
        con.execute(f"SET temp_directory='{os.path.abspath('data/_duckdb_tmp')}'")

        # 只保留股票列表内的代码：raw 里按日期存的文件含全市场 A 股（约 5400 只），
        # 而按股票存的数据只有主板（3196 只）。提前过滤可省掉约 35% 的行。
        keep_clause = ""
        sl_path = "data/raw/stock_list/stock_list.csv"
        if os.path.exists(sl_path):
            # 只保留沪深主板：raw 里按日期存的文件含全市场 A 股（约 5400 只），
            # 而按股票存的数据只有主板。提前过滤可省掉约 40% 的行。
            # 这里再用 is_main_board 兜一层，防止 stock_list 口径变化混入非主板。
            sl = pd.read_csv(sl_path, usecols=['ts_code'], dtype=str)
            codes = sorted({c for c in sl['ts_code'].dropna() if is_main_board(c)})
            con.register('_keep_codes_df', pd.DataFrame({'ts_code': codes}))
            con.execute("CREATE OR REPLACE VIEW _keep_codes AS "
                        "SELECT DISTINCT CAST(ts_code AS VARCHAR) AS ts_code FROM _keep_codes_df")
            keep_clause = " WHERE ts_code IN (SELECT ts_code FROM _keep_codes)"
            print(f"  股票范围：沪深主板 {len(codes)} 只"
                  + (f"（stock_list 共 {len(sl)} 只，已剔除 {len(sl) - len(codes)} 只非主板）"
                     if len(sl) != len(codes) else ""))

        # ---- 1. 逐个数据源建视图（顺带去重 + 剔除 Unnamed 列）----
        views = []
        for section_name in basic_daily_data_list:
            files = glob.glob(raw_data_path + section_name + "/*.csv")
            if not files:
                print(f"  {section_name}：无文件，跳过")
                continue

            # 增量时按日期裁掉旧文件（按交易日分文件的数据源才这么做）
            file_list = None
            if incremental and section_name in DATE_FILED_SECTIONS:
                picked = [f for f in files
                          if os.path.basename(f)[:-4].isdigit()
                          and os.path.basename(f)[:-4] > since]
                # 一个都没挑到就退回全量文件列表，保证列结构与全量时一致
                if picked:
                    file_list = picked

            if file_list is not None:
                src_arg = "[" + ", ".join(
                    "'" + os.path.abspath(p).replace('\\', '/') + "'" for p in file_list) + "]"
            else:
                src_arg = "'" + os.path.join(
                    raw_data_path, section_name, "*.csv").replace('\\', '/') + "'"

            # 每个数据源用独立的视图名：复用同一个名字会让先建的视图
            # 在后续 CREATE OR REPLACE 后指向新的源（视图是惰性求值的）
            src_view = f'_src_{section_name}'
            con.execute(
                f"CREATE OR REPLACE VIEW {src_view} AS SELECT * FROM read_csv_auto("
                f"{src_arg}, union_by_name=true, "
                f"types={{'ts_code':'VARCHAR','trade_date':'VARCHAR'}})")

            cols = _view_columns(con, src_view)
            if 'ts_code' not in cols or 'trade_date' not in cols:
                print(f"  {section_name}：缺少 ts_code/trade_date，跳过")
                continue

            # 增量时再在 SQL 层过滤一遍（按股票存的数据源只能靠这一层）
            date_clause = ""
            if incremental:
                date_clause = f" AND CAST(trade_date AS VARCHAR) > '{since}'"
            keep = keep_clause + (date_clause if keep_clause
                                  else date_clause.replace(" AND ", " WHERE ", 1))

            sel = ', '.join(f'"{c}"' for c in cols)
            # 同一 (ts_code, trade_date) 只保留一行，避免后续 JOIN 产生笛卡尔膨胀
            con.execute(
                f"CREATE OR REPLACE VIEW v_{section_name} AS "
                f"SELECT {sel} FROM ("
                f"  SELECT *, row_number() OVER (PARTITION BY ts_code, trade_date "
                f"             ORDER BY ts_code) AS _rn FROM {src_view}{keep})"
                f"  WHERE _rn = 1")

            n_rows = con.execute(f"SELECT count(*) FROM v_{section_name}").fetchone()[0]
            views.append(f"v_{section_name}")
            print(f"  载入 {section_name} → {n_rows} 行"
                  + (f"（新日期文件 {len(file_list)} 个）" if file_list else ""))

        if not views:
            print("  无数据源，final_result.parquet 未生成。")
            return None

        # ---- 2. 全外连接，冲突列按 _x / _y 规则改名；分批物化控制内存 ----
        remaining = list(views)
        stage_view = None
        sql = None
        group_no = 0
        while remaining:
            group = remaining[:join_group_size]
            remaining = remaining[join_group_size:]
            group_no += 1

            if stage_view is None:
                sql = build_join_sql(con, group, key)
            else:
                sql = build_join_sql(con, [stage_view] + group, key, base=stage_view)

            if remaining:
                stage_path = f"data/_join_stage{group_no}.parquet"
                if os.path.exists(stage_path):
                    os.remove(stage_path)
                con.execute(f"COPY (SELECT * FROM ({sql}) ORDER BY ts_code, trade_date) "
                            f"TO '{stage_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
                stage_view = f"v_stage{group_no}"
                con.execute(f"CREATE OR REPLACE VIEW {stage_view} AS "
                            f"SELECT * FROM read_parquet('{stage_path}')")
                n = con.execute(f"SELECT count(*) FROM {stage_view}").fetchone()[0]
                print(f"  中间结果 stage{group_no} 已落盘 → {n} 行")

        # ---- 3. 合并 ST 标识并落盘 ----
        if os.path.exists(ST_FLAG_PATH):
            con.execute("CREATE OR REPLACE VIEW v_st_flag AS "
                        f"SELECT ts_code, trade_date, is_st FROM read_parquet('{ST_FLAG_PATH}')")
            final_sql = (f"SELECT a.*, COALESCE(b.is_st, 0) AS is_st "
                         f"FROM ({sql}) a LEFT JOIN v_st_flag b USING (ts_code, trade_date)")
        else:
            print(f"  {ST_FLAG_PATH} 不存在，is_st 全部置 0")
            final_sql = f"SELECT a.*, 0 AS is_st FROM ({sql}) a"

        if incremental:
            # 新增段先单独落盘：供 split_to_series_section 只拆分新日期
            if os.path.exists(NEW_CHUNK_PATH):
                os.remove(NEW_CHUNK_PATH)
            con.execute(f"COPY (SELECT * FROM ({final_sql}) ORDER BY ts_code, trade_date) "
                        f"TO '{NEW_CHUNK_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
            n_new = con.execute(
                f"SELECT count(*) FROM read_parquet('{NEW_CHUNK_PATH}')").fetchone()[0]
            if n_new == 0:
                print("  无新增交易日数据，宽表保持原样。")
                os.remove(NEW_CHUNK_PATH)
                new_chunk = None
            else:
                # 旧宽表只保留 trade_date <= since 的部分，新增段整体并入
                tmp_all = out_path + ".tmp.parquet"
                if os.path.exists(tmp_all):
                    os.remove(tmp_all)
                con.execute(
                    f"COPY (SELECT * FROM ("
                    f"  SELECT * FROM read_parquet('{out_path}') "
                    f"    WHERE CAST(trade_date AS VARCHAR) <= '{since}'"
                    f"  UNION ALL BY NAME"
                    f"  SELECT * FROM read_parquet('{NEW_CHUNK_PATH}')"
                    f") ORDER BY ts_code, trade_date) "
                    f"TO '{tmp_all}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
                os.replace(tmp_all, out_path)
                new_chunk = NEW_CHUNK_PATH
                n = con.execute(
                    f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
                print(f"  宽表追加 {n_new} 行 → 共 {n} 行（新增段：{NEW_CHUNK_PATH}）")
        else:
            if os.path.exists(out_path):
                os.remove(out_path)
            con.execute(f"COPY (SELECT * FROM ({final_sql}) ORDER BY ts_code, trade_date) "
                        f"TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)")

            n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
            n_st = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}') WHERE is_st > 0").fetchone()[0]
            print(f"  final_result.parquet 生成完成，共 {n} 行，其中 ST 记录 {n_st} 行")

        for f in glob.glob("data/_join_stage*.parquet"):
            try:
                os.remove(f)
            except OSError:
                pass
        return new_chunk
    finally:
        con.close()


def _final_result_last_date(path: str = None):
    """已有宽表的最新交易日；文件不存在时返回 None。"""
    path = path or FINAL_RESULT_PATH
    if not os.path.exists(path):
        return None
    try:
        con = duckdb.connect()
        try:
            v = con.execute(
                f"SELECT max(CAST(trade_date AS VARCHAR)) FROM read_parquet('{path}')"
            ).fetchone()[0]
            return str(v) if v else None
        finally:
            con.close()
    except Exception as e:
        print(f"  读取 {path} 失败：{e}")
        return None


def split_to_series_section(input_file_name: str):
    """
    将宽表 final_result.parquet 按股票代码拆分到 data/series/<ts_code>.parquet。

    宽表已按 (ts_code, trade_date) 排序，因此用 pyarrow 按批流式读取，
    内存里只累积"单只股票"的数据，累计一只写一只，避免整表进内存。

    若目标文件已存在（含因子列），按交易日做增量 upsert：
    新覆盖的日期取新值，未覆盖的历史行连同其因子列一起保留。
    """
    series_path = "data/series/"
    os.makedirs(series_path, exist_ok=True)

    pf = pq.ParquetFile(input_file_name)
    print("列：", pf.schema_arrow.names)

    cur_code, chunks, written = None, [], 0

    def flush(code, parts):
        nonlocal written
        new_df = pa.concat_tables(parts).to_pandas()
        out_file = f"{series_path}{code}.parquet"
        if os.path.exists(out_file):
            old = pd.read_parquet(out_file)
            if 'trade_date' in old.columns and 'trade_date' in new_df.columns:
                old['trade_date'] = old['trade_date'].astype(str)
                hit = old['trade_date'].isin(set(new_df['trade_date']))
                keep = old[~hit]
                # 宽表新增的列也要带进时序文件（老行补 NaN），否则增量更新
                # 会静默丢掉新字段，与全量重建的结果不一致
                extra = [c for c in new_df.columns if c not in old.columns]
                # 时序文件里有、新宽表没有的列（大多是上一轮算好的因子）补 NaN；
                # 一次性 concat 而不是逐列赋值，避免 DataFrame 碎片化
                missing = [c for c in old.columns if c not in new_df.columns]
                if missing:
                    new_df = pd.concat(
                        [new_df, pd.DataFrame(np.nan, index=new_df.index, columns=missing)],
                        axis=1)
                new_df = new_df[list(old.columns) + extra]
                merged = pd.concat([keep, new_df], ignore_index=True)
            else:
                merged = new_df
        else:
            merged = new_df
        merged = merged.sort_values('trade_date') if 'trade_date' in merged.columns else merged
        merged.to_parquet(out_file, index=False, engine='pyarrow', compression='snappy')
        written += 1

    for batch in pf.iter_batches(batch_size=65536):
        tbl = pa.Table.from_batches([batch])
        if 'ts_code' not in tbl.column_names:
            raise KeyError('final_result 缺少 ts_code 列')
        codes = tbl.column('ts_code').to_pylist()
        n = tbl.num_rows
        i = 0
        while i < n:
            code = codes[i]
            j = i
            while j < n and codes[j] == code:
                j += 1
            if cur_code == code:
                chunks.append(tbl.slice(i, j - i))
            else:
                if cur_code is not None:
                    flush(cur_code, chunks)
                cur_code, chunks = code, [tbl.slice(i, j - i)]
            i = j

    if cur_code is not None:
        flush(cur_code, chunks)

    print(f"  拆分完成，共写入/更新 {written} 只股票的时序文件")


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="原始数据 → 宽表 → 时序层的合并流程")
    ap.add_argument('--incremental', action='store_true',
                    help="增量模式：只处理新交易日并追加，已有数据一律保留")
    ap.add_argument('--since', default=None,
                    help="增量/重做起点（不含该日）；默认取已有宽表最新交易日")
    ap.add_argument('--end', dest='end_date', default=None,
                    help="ST 标识增量到的日期（YYYYMMDD）；默认今天")
    args = ap.parse_args()

    financial_data_preprocess()
    build_st_flag(end_date=args.end_date)
    if args.incremental or args.since:
        chunk = merge_basic_daily_data(incremental=True, since=args.since)
        if chunk:
            split_to_series_section(chunk)
        else:
            print("  无新增数据，跳过拆分。")
    else:
        merge_basic_daily_data()
        split_to_series_section(FINAL_RESULT_PATH)
