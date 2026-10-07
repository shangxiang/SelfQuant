"""
run_signal.py — 增量数据更新 + 选股信号生成入口

从项目根目录执行：
    python run_signal.py                 # 每日收盘后跑：增量更新到最近交易日
    python run_signal.py --end 20260925  # 指定更新到哪一天
    python run_signal.py --since 20260901  # 重做 20260901 之后的历史
    python run_signal.py --full          # 全量重建（会覆盖 final_result.parquet）
    python run_signal.py --signal        # 更新完顺带出 T+1 的选股信号

流程：
  1. 确定最近交易日（T）和本地数据截止日
  2. 增量下载原始数据到 T 日（各数据源自己判断缺口，已下载的文件不动）
  3. merge_data：ST 标识增量 + 日线宽表增量追加 + 拆分 series（只 upsert 新日期）
  4. new_factor_manager：只算新行的因子（携带 260 日历史上下文）
  5. tools：只对新日期做截面化 + 标准化
  6. ElasticNetStrategy.fit(T) + generate_signals(T)：打分选股
  7. BlindWindowTiming.get_position_ratio(T+1)：择时仓位
  8. 输出下一交易日（T+1）的选股结果和仓位建议

增量更新的三条底线
------------------
- 已下载/已计算的数据只补充，不清理：宽表、时序层、截面层都是追加或按日 upsert；
- 历史行连因子列一起保留，被覆盖的日期才重算；
- ST 标识不再走 get_namechange，改用 stock_list 的当前名称与本地记录的
  上一次名称比对（见 merge_data.build_st_flag），每日更新即可保证名称连续。
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# ================================================================
# 配置区：按需修改
# ================================================================

# 选股策略
from strategy.selection.elastic_net_strategy import ElasticNetConfig, ElasticNetStrategy
from strategy.data_loader import DataLoader

# 择时策略
from backtest.timing import BlindWindowTiming, MATiming

# 止损策略（仅用于展示当前持仓的止损线，不影响选股）
from strategy.sell.hold_n_days import HoldNDaysSellStrategy

# 选股数量
TOP_N = 5

# 择时配置
TIMING_CFG = dict(
    small_file='data/raw/index_daily/932000.CSI.csv',
    large_file='data/raw/index_daily/000510.SH.csv',
    corr_pct=20.0,
    vol_pct=33.0,
    avoid_ratio=0.0,
    rolling_window=80
)

# 是否执行数据更新步骤（调试时可关闭跳过耗时步骤）
RUN_DOWNLOAD    = True
RUN_MERGE       = True
RUN_FACTORS     = True
RUN_STANDARDIZE = True

# ================================================================


def _get_latest_trade_date() -> str:
    """
    返回距今最近的交易日（is_open=1）。

    优先通过 Tushare API 获取最新日历，确保日历不过期：
    - 若日历最近一日是今天 → 返回今天（今天是交易日）
    - 若日历最近一日不是今天 → 返回日历里的最后一天（今天是非交易日）
    API 调用失败时回落到本地 trade_cal.csv。
    """
    today = datetime.today().strftime('%Y%m%d')

    def _latest_from(cal_df: pd.DataFrame) -> str:
        open_days = cal_df[cal_df['is_open'] == 1]
        past = open_days[open_days['cal_date'] <= today].sort_values('cal_date')
        if past.empty:
            raise RuntimeError('交易日历中找不到今日或之前的交易日，请先更新交易日历。')
        return past['cal_date'].iloc[-1]

    try:
        from data_api.tushareApi import TushareDataSource
        api = TushareDataSource()
        # is_open='' 拉取全部日期（含非交易日），用于正确判断今天是否是交易日
        df = api.get_trade_calender(startdate='20200101', enddate=today, is_open='')
        df['cal_date'] = df['cal_date'].astype(str)
        return _latest_from(df)
    except Exception:
        pass

    cal = pd.read_csv('data/raw/trade_cal.csv', dtype={'cal_date': str})
    return _latest_from(cal)


def _get_next_trade_date(t: str) -> Optional[str]:
    """返回 T 日之后的下一个交易日，不存在则返回 None。"""
    cal = pd.read_csv('data/raw/trade_cal.csv', dtype={'cal_date': str})
    cal = cal[cal['is_open'] == 1].sort_values('cal_date')
    future = cal[cal['cal_date'] > t]
    return future['cal_date'].iloc[0] if not future.empty else None


def _local_data_end() -> Optional[str]:
    """
    返回本地截面数据中最新的日期，作为判断是否需要增量更新的依据。

    优先用分区存储 data/market/trade_date=YYYYMMDD/（当前口径），
    再回退到旧版 data/section/ 下的 Parquet / CSV。
    """
    import re as _re
    dirs = glob.glob('data/market/trade_date=*')
    dates = [m for m in (_re.search(r'(\d{8})$', d) for d in dirs) if m]
    if dates:
        return max(m.group(1) for m in dates)

    parquet_files = glob.glob('data/section/[0-9]*.parquet')
    if parquet_files:
        return max(os.path.basename(f).replace('.parquet', '') for f in parquet_files)

    csv_files = glob.glob('data/section/[0-9]*.csv')
    if not csv_files:
        return None
    return max(os.path.basename(f).replace('.csv', '') for f in csv_files)


def step_download(end_date: str) -> None:
    """
    增量下载原始数据到 end_date。

    各数据源自己判断本地已有最新日期，只拉缺失的部分；已下载的文件一律不动。
    注意：不再下载 namechange —— ST 标识改用 stock_list 的当前名称比对（见 step_merge）。
    """
    print(f'\n[1/6] 增量下载原始数据 → {end_date}')
    from data_local_dump import DownloadData
    d = DownloadData()
    d.trade_cal()
    d.stock_list()
    d.daily_basic_data(end=end_date)
    d.moneyflow(end=end_date)
    d.margin_detail(end=end_date)
    d.top_list(end=end_date)
    d.stock_data(end=end_date)
    d.adj_factor(end=end_date)         # 复权因子：缺了它后复权价与 label 全是 NaN
    d.income(end=end_date)
    d.balancesheet(end=end_date)
    d.cashflow(end=end_date)
    d.index_daily_selected(end=end_date)   # 精选指数（原 index_daily 会下全部 SSE 指数）
    print('  下载完成。')


def step_merge(end_date: str = None, incremental: bool = True,
               since: str = None, refresh_last_days: int = 0) -> Optional[list]:
    """
    合并财务数据、日线宽表，拆分 series。

    incremental=True（默认）时全流程只追加不覆盖：
      - ST 标识：用 stock_list 的当前名称与本地上一次记录比对，只追加新交易日；
      - 宽表：只合并 trade_date > since 的新日期，追加进已有的 final_result.parquet；
      - 拆分：只拿新增段去 upsert 各股票的时序文件，历史行连同因子列原样保留。

    想重做某段历史：since 设为那段历史的前一天（该日之后的行会被新值替换）。

    Returns
    -------
    list[str] | None  本次新增的交易日；无新增返回 None。
    """
    print('\n[2/6] 合并数据（merge_data）'
          + ('  [增量]' if incremental else '  [全量重建]'))
    from merge_data import (
        financial_data_preprocess, build_st_flag, resolve_since,
        merge_basic_daily_data, split_to_series_section,
    )
    financial_data_preprocess()
    # ST 标识：必须在合并前生成，merge 时作为 is_st 列并入宽表
    build_st_flag(end_date=end_date)

    if incremental or since:
        # 先算出实际起点，重做历史时才能同步回退因子进度
        since = resolve_since(since, refresh_last_days)
        chunk = merge_basic_daily_data(incremental=True, since=since)
        if not chunk:
            print('  无新增日线数据，后续步骤跳过。')
            return None
        split_to_series_section(chunk)
        # 重做历史（since 早于现有最新日 / 回退重做）时，被覆盖行的因子列已被重置为 NaN，
        # 必须把因子进度回退到 since，否则 FactorManager 会认为这些行算过了。
        if since:
            _rewind_date_range(since)
        import pandas as _pd
        return sorted(_pd.read_parquet(chunk, columns=['trade_date'])['trade_date']
                      .astype(str).unique().tolist())

    merge_basic_daily_data()
    split_to_series_section('data/final_result.parquet')
    return None


def _rewind_date_range(since: str, path: str = 'data/series/_date_range.csv') -> None:
    """
    把因子计算进度回退到 since：data/series/_date_range.csv 里
    end_date > since 的记录一律改成 since，迫使 FactorManager 重算这些行。

    只在「重做某段历史」时需要；日常追加新日期时 end_date 本来就落后于新行。
    """
    import pandas as _pd
    if not os.path.exists(path):
        return
    cfg = _pd.read_csv(path, dtype=str)
    end = _pd.to_numeric(cfg['end_date'], errors='coerce')
    hit = end > int(since)
    if not hit.any():
        return
    cfg.loc[hit, 'end_date'] = since
    cfg.to_csv(path, index=False)
    print(f'  因子进度回退到 {since}：{int(hit.sum())} 只股票将在下一步重算因子')


def step_factors() -> None:
    """计算时序因子（new_factor_manager），多进程并行。

    FactorManager 自带增量判断：列维度缺什么补什么，行维度只算 _date_range.csv
    记录之后的新行（携带 260 日历史上下文），历史因子不会被清空。
    """
    print('\n[3/6] 计算时序因子（new_factor_manager）')
    import glob as _glob
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from factor.new_factor_manager import (
        build_financial_features, _process_one_stock, update_series_config,
    )
    # 优先使用Parquet格式
    if os.path.exists('data/financial.parquet'):
        fin_df = pd.read_parquet('data/financial.parquet')
    else:
        fin_df = pd.read_csv('data/financial.csv')
    fin_features = build_financial_features(fin_df)
    # 优先查找Parquet文件
    parquet_files = _glob.glob('data/series/[0-9]*.parquet')
    if parquet_files:
        files = parquet_files
    else:
        files = _glob.glob('data/series/[0-9]*.csv')
    total = len(files)
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    completed = 0
    date_range_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_stock, (f, fin_features)): f for f in files}
        for future in as_completed(futures):
            completed += 1
            print(f'\r  {completed}/{total}', end='', flush=True)
            try:
                _, ts_code, start_date, end_date = future.result()
                date_range_results.append((ts_code, start_date, end_date))
            except Exception as e:
                print(f'\n  ✗ {futures[future]}: {e}')
    update_series_config(date_range_results, 'data/series/_date_range.csv')
    print('\n  因子计算完成，date range config 已更新。')


def step_standardize(only_dates: list = None, force: bool = False) -> None:
    """
    截面化 + 标准化。

    默认只处理本次新增（或上次没跑完）的交易日：
      - series_to_section 返回实际构建的日期；
      - standardize 只对这些日期做缩尾 / 中性化 / z-score；
      - 已完成标准化的历史分区不会被重算，更不会被清空。

    Parameters
    ----------
    only_dates : list[str] | None  强制重建这些交易日（重做历史时用）
    force      : bool              全量重建整个截面层（--full 时用）
    """
    print('\n[4/6] 截面化 + 标准化（tools）')
    from standardize import (
        series_to_section, standardize, section_duplicates, pending_section_dates,
    )
    if force:
        built = series_to_section(incremental=False)
    else:
        # 新增交易日 + 上次没跑完的半成品
        targets = sorted(set(only_dates or []) | set(pending_section_dates()))
        if not targets:
            print('  无新增截面，跳过。')
            return
        built = series_to_section(only_dates=targets)

    if not built:
        print('  无截面产出，跳过标准化。')
        return

    print(f'  待标准化 {len(built)} 个交易日：{built[0]} ~ {built[-1]}')
    standardize(incremental=True, only_dates=built)
    section_duplicates(only_dates=built)
    print('  标准化完成。')


def step_generate_signals(signal_date: str, next_date: str) -> None:
    """
    用 signal_date（T 日）的截面数据训练模型并打分，
    输出 next_date（T+1 日）的选股结果和择时仓位。
    """
    print(f'\n[5/6] 训练模型并生成信号（信号日={signal_date}）')

    # s_cfg = ElasticNetConfig()
    # loader = DataLoader(s_cfg)
    # strategy = ElasticNetStrategy(s_cfg, loader)

    # LightGBM（非线性，使用原始因子值，截面 rank 标签）
    from strategy.selection.lgbm_strategy import LGBMConfig, LGBMStrategy
    s_cfg    = LGBMConfig()
    loader   = DataLoader(s_cfg)
    strategy = LGBMStrategy(s_cfg, loader)

    ok = strategy.fit(signal_date)
    if not ok:
        print(f'  ✗ 模型训练失败（数据不足或 {signal_date} 非交易日）')
        return

    signals = strategy.generate_signals(signal_date)
    if signals is None or signals.empty:
        print('  ✗ 信号生成失败（截面数据缺失）')
        return

    top = signals.head(TOP_N).copy()
    print(f'  ✓ 打分完成，共 {len(signals)} 只股票参与评分')

    # 附上股票名称
    stock_df = pd.read_csv(s_cfg.stock_list_file, usecols=['ts_code', 'name'])
    top = top.merge(stock_df, on='ts_code', how='left')
    top.insert(1, 'name', top.pop('name'))

    print(f'\n[6/6] 择时判断（信号日={signal_date}）')
    # timing = BlindWindowTiming(**TIMING_CFG)
    timing = MATiming(
        index_file='data/raw/index_daily/932000.CSI.csv',
        ma_period=60,
    )
    # prepare 需要足够长的历史，从数据起点到 signal_date
    timing.prepare('20200101', signal_date)
    ratio = timing.get_position_ratio(next_date)

    # ── 构建输出内容 ──────────────────────────────────────────────
    top1_score = top['score'].iloc[0] if not top.empty else 0.0
    MIN_SCORE = 0.0001  # 与 BacktestConfig.min_score_threshold 保持一致

    if ratio == 0.0:
        timing_label = '空仓'
    elif ratio < 1.0:
        timing_label = f'半仓（{ratio:.0%}）'
    else:
        timing_label = '满仓'

    model_label = f'模型有效（top1={top1_score:.6f}）' if top1_score >= MIN_SCORE else f'模型强度不足（top1={top1_score:.6f} < {MIN_SCORE}），建议空仓'

    lines = [
        '=' * 60,
        f'  信号日（T）  : {signal_date}',
        f'  买入日（T+1）: {next_date}',
        f'  择时仓位     : {ratio:.0%}  ← {timing_label}',
        f'  模型强度     : {model_label}',
        '=' * 60,
    ]

    if ratio == 0.0:
        lines.append('\n择时信号为空仓，无需建仓。')
    else:
        lines.append(f'\n下一交易日（{next_date}）建议买入 Top-{TOP_N}（按打分降序）：\n')
        lines.append(top[['ts_code', 'name', 'score']].to_string(index=False, float_format='{:+.4f}'.format))

    output = '\n'.join(lines)
    print('\n' + output)

    # ── 保存结果 ──────────────────────────────────────────────────
    out_dir = 'signals'
    os.makedirs(out_dir, exist_ok=True)

    # 文字摘要
    summary_path = os.path.join(out_dir, f'{next_date}_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(output + '\n')
    print(f'\n已保存摘要 → {summary_path}')

    # 完整打分表（含所有评分股票，不只 Top-N）
    full_path = os.path.join(out_dir, f'{next_date}_full_scores.csv')
    stock_df_full = pd.read_csv(s_cfg.stock_list_file, usecols=['ts_code', 'name'])
    signals_full = signals.merge(stock_df_full, on='ts_code', how='left')
    signals_full.insert(1, 'name', signals_full.pop('name'))
    signals_full['signal_date'] = signal_date
    signals_full['buy_date'] = next_date
    signals_full['timing_ratio'] = ratio
    signals_full.to_csv(full_path, index=False, float_format='%.6f')
    print(f'已保存完整打分 → {full_path}')

    # Top-N 精简表
    top_path = os.path.join(out_dir, f'{next_date}_top{TOP_N}.csv')
    top['signal_date'] = signal_date
    top['buy_date'] = next_date
    top['timing_ratio'] = ratio
    top.to_csv(top_path, index=False, float_format='%.6f')
    print(f'已保存 Top-{TOP_N} → {top_path}')


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description='run_signal.py — 增量数据更新 + 选股信号')
    ap.add_argument('--end', dest='end_date', default=None,
                    help='更新到的交易日（YYYYMMDD），默认自动取最近交易日')
    ap.add_argument('--since', default=None,
                    help='重做起点（不含该日）：该日之后的宽表行会被新值替换')
    ap.add_argument('--refresh', dest='refresh_last_days', type=int, default=0,
                    help='回退重做最后 N 个交易日（上次更新时数据不完整时用，如 --refresh 1）')
    ap.add_argument('--full', action='store_true',
                    help='全量重建宽表（默认增量；全量会覆盖 final_result.parquet）')
    ap.add_argument('--signal', action='store_true',
                    help='更新完数据后顺带生成 T+1 的选股信号')
    ap.add_argument('--skip-download', action='store_true')
    ap.add_argument('--skip-merge', action='store_true')
    ap.add_argument('--skip-factors', action='store_true')
    ap.add_argument('--skip-standardize', action='store_true')
    args = ap.parse_args()

    print('=' * 60)
    print('  run_signal.py — 增量数据更新 + 选股信号')
    print('  （默认增量：已下载/已计算的数据只补充，不清理）')
    print('=' * 60)

    # 确定关键日期
    t_date   = args.end_date or _get_latest_trade_date()
    t1_date  = _get_next_trade_date(t_date)
    local_end = _local_data_end()

    print(f'\n最近交易日（T）  : {t_date}')
    print(f'下一交易日（T+1）: {t1_date or "未知（日历不足）"}')
    print(f'本地截面数据截止 : {local_end or "无"}')

    if t1_date is None:
        # 日历未更新到 T+1，用自然日顺延估算（跳过周末）
        from datetime import timedelta
        dt = datetime.strptime(t_date, '%Y%m%d')
        dt += timedelta(days=1)
        while dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            dt += timedelta(days=1)
        t1_date = dt.strftime('%Y%m%d')
        print(f'  日历不足，T+1 估算为 {t1_date}（跳过周末，未排除节假日）')

    needs_update = args.full or args.since or (local_end is None) or (local_end < t_date)
    if not needs_update:
        print('\n本地数据已是最新，跳过数据更新步骤。')

    new_dates = None
    if RUN_DOWNLOAD and needs_update and not args.skip_download:
        step_download(t_date)

    if RUN_MERGE and needs_update and not args.skip_merge:
        new_dates = step_merge(end_date=t_date,
                               incremental=not args.full,
                               since=args.since,
                               refresh_last_days=args.refresh_last_days)

    if RUN_FACTORS and needs_update and not args.skip_factors:
        if new_dates is None or new_dates:
            step_factors()
        else:
            print('\n[3/6] 无新增数据，跳过因子计算。')

    if RUN_STANDARDIZE and needs_update and not args.skip_standardize:
        if new_dates is None or new_dates:
            step_standardize(only_dates=new_dates)
        else:
            print('\n[4/6] 无新增数据，跳过标准化。')

    if args.signal:
        step_generate_signals(t_date, t1_date)


if __name__ == '__main__':
    main()
