"""
run_signal.py — 一站式选股信号生成入口

从项目根目录执行：
    python run_signal.py

流程：
  1. 确定最近交易日（T）和本地数据截止日
  2. 增量下载原始数据到 T 日
  3. merge_data：合并财务数据 + 日线宽表 + 拆分 series
  4. new_factor_manager：计算时序因子
  5. tools：截面化 + 标准化
  6. ElasticNetStrategy.fit(T) + generate_signals(T)：打分选股
  7. BlindWindowTiming.get_position_ratio(T+1)：择时仓位
  8. 输出下一交易日（T+1）的选股结果和仓位建议
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
TOP_N = 10

# 择时配置
TIMING_CFG = dict(
    small_file='data/raw/index_daily/932000.CSI.csv',
    large_file='data/raw/index_daily/000510.CSI.csv',
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
    返回本地截面数据（data/section/）中最新的日期，
    作为判断是否需要增量更新的依据。
    """
    files = glob.glob('data/section/[0-9]*.csv')
    if not files:
        return None
    return max(os.path.basename(f).replace('.csv', '') for f in files)


def step_download(end_date: str) -> None:
    """增量下载原始数据到 end_date。"""
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
    d.income(end=end_date)
    d.balancesheet(end=end_date)
    d.cashflow(end=end_date)
    d.index_daily(end=end_date)
    print('  下载完成。')


def step_merge() -> None:
    """合并财务数据、日线宽表，拆分 series。"""
    print('\n[2/6] 合并数据（merge_data）')
    from merge_data import financial_data_preprocess, merge_basic_daily_data, split_to_series_section
    financial_data_preprocess()
    merge_basic_daily_data()
    split_to_series_section('data/final_result.csv')
    print('  合并完成。')


def step_factors() -> None:
    """计算时序因子（new_factor_manager），多进程并行。"""
    print('\n[3/6] 计算时序因子（new_factor_manager）')
    import glob as _glob
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from factor.new_factor_manager import (
        build_financial_features, _process_one_stock, update_series_config,
    )
    fin_df = pd.read_csv('data/financial.csv')
    fin_features = build_financial_features(fin_df)
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


def step_standardize() -> None:
    """截面化 + 标准化（tools）。"""
    print('\n[4/6] 截面化 + 标准化（tools）')
    from tools import series_to_section, standardize, section_duplicates
    series_to_section(incremental=True)
    standardize(incremental=True)
    section_duplicates()
    print('  标准化完成。')


def step_generate_signals(signal_date: str, next_date: str) -> None:
    """
    用 signal_date（T 日）的截面数据训练模型并打分，
    输出 next_date（T+1 日）的选股结果和择时仓位。
    """
    print(f'\n[5/6] 训练模型并生成信号（信号日={signal_date}）')

    s_cfg = ElasticNetConfig()
    loader = DataLoader(s_cfg)
    strategy = ElasticNetStrategy(s_cfg, loader)

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
    print('=' * 60)
    print('  run_signal.py — 一站式选股信号生成')
    print('=' * 60)

    # 确定关键日期
    t_date   = _get_latest_trade_date()
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

    needs_update = (local_end is None) or (local_end < t_date)
    if not needs_update:
        print('\n本地数据已是最新，跳过数据更新步骤。')

    if RUN_DOWNLOAD and needs_update:
        step_download(t_date)

    if RUN_MERGE and needs_update:
        step_merge()

    if RUN_FACTORS and needs_update:
        step_factors()

    if RUN_STANDARDIZE and needs_update:
        step_standardize()

    # step_generate_signals(t_date, t1_date)


if __name__ == '__main__':
    main()
