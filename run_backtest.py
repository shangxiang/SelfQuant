"""
回测主入口。从项目根目录执行：
    python run_backtest.py

也可以传入日期查询当日选股和择时结果：
    python run_backtest.py 20260522
"""
import sys
import os
import pandas as pd
from strategy.data_loader import DataLoader
from strategy.selection.elastic_net_strategy import ElasticNetConfig, ElasticNetStrategy
from strategy.selection.lgbm_strategy import LGBMConfig, LGBMStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine, resolve_date_range
from backtest.timing import MATiming, StyleConvergenceTiming, LastBatchTiming, BlindWindowTiming, BlindWindowVolTiming
from strategy.selection.lgbm_ranker_strategy import LGBMRankerConfig, LGBMRankerStrategy
from strategy.selection.ic_weighted_strategy import ICWeightedConfig, ICWeightedStrategy
from strategy.sell.hold_n_days_batch_stop import HoldNDaysBatchStopStrategy
from backtest.timing import LGBMDriftTiming
from strategy.selection.lgbm_dynamic_strategy import LGBMDynamicConfig, LGBMDynamicStrategy
from strategy.selection.lgbm_binary_strategy import LGBMBinaryConfig, LGBMBinaryStrategy
from strategy.selection.tabnet_strategy import TabNetConfig, TabNetStrategy

TOP_N = 10

TIMING_CFG = dict(
    small_file='data/raw/index_daily/932000.CSI.csv',
    large_file='data/raw/index_daily/000510.SH.csv',
    corr_pct=20.0,
    vol_pct=33.0,
    avoid_ratio=0.0,
    rolling_window=80,
    trend_ma=20,    # 趋势滤波：仅vol触发且在MA20上方时不空仓（0=禁用，退化为原始OR逻辑）
)


def query_signal(signal_date: str) -> None:
    """
    传入信号日（T），输出截面选股打分和择时仓位建议。
    信号日必须是交易日且本地截面数据已覆盖该日期。
    """
    cal = pd.read_csv('data/raw/trade_cal.csv', dtype={'cal_date': str})
    cal = cal[cal['is_open'] == 1].sort_values('cal_date')
    future = cal[cal['cal_date'] > signal_date]
    next_date = future['cal_date'].iloc[0] if not future.empty else signal_date

    # s_cfg = ElasticNetConfig()
    # loader = DataLoader(s_cfg)
    # strategy = ElasticNetStrategy(s_cfg, loader)

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
    stock_df = pd.read_csv(s_cfg.stock_list_file, usecols=['ts_code', 'name'])
    top = top.merge(stock_df, on='ts_code', how='left')
    top.insert(1, 'name', top.pop('name'))
    print(f'  ✓ 打分完成，共 {len(signals)} 只股票参与评分')

    timing = BlindWindowTiming(**TIMING_CFG)
    timing.prepare('20200101', signal_date)
    ratio = timing.get_position_ratio(next_date)

    if ratio == 0.0:
        timing_label = '空仓'
    elif ratio < 1.0:
        timing_label = f'半仓（{ratio:.0%}）'
    else:
        timing_label = '满仓'

    MIN_SCORE = 0.0001
    top1_score = top['score'].iloc[0] if not top.empty else 0.0
    model_label = f'模型有效（top1={top1_score:.6f}）' if top1_score >= MIN_SCORE else f'模型强度不足（top1={top1_score:.6f} < {MIN_SCORE}），建议空仓'

    print('\n' + '=' * 60)
    print(f'  信号日（T）  : {signal_date}')
    print(f'  买入日（T+1）: {next_date}')
    print(f'  择时仓位     : {ratio:.0%}  ← {timing_label}')
    print(f'  模型强度     : {model_label}')
    print('=' * 60)

    if ratio == 0.0:
        print('\n择时信号为空仓，无需建仓。')
    elif top1_score < MIN_SCORE:
        print('\n模型强度不足，建议空仓。')
    else:
        print(f'\n下一交易日（{next_date}）建议买入 Top-{TOP_N}（按打分降序）：\n')
        print(top[['ts_code', 'name', 'score']].to_string(index=False, float_format='{:+.4f}'.format))


def _parse_args():
    """命令行参数：位置参数=信号日查询；--start/--end 指定回测区间。"""
    import argparse
    p = argparse.ArgumentParser(description='SelfQuant 回测入口')
    p.add_argument('signal_date', nargs='?',
                   help='信号日 YYYYMMDD；传了该参数则只查询当日选股与择时')
    p.add_argument('--start', help='回测开始日 YYYYMMDD')
    p.add_argument('--end', help='回测结束日 YYYYMMDD')
    p.add_argument('--ic', action='store_true',
                   help='回测前额外跑一次纯因子评估（RankIC / 分层收益）')
    p.add_argument('--top-n', type=int, help='每次建仓选股数量')
    p.add_argument('--hold', type=int, help='持有期（交易日），同时作为卖出策略参数')
    p.add_argument('--commission', type=float, help='单边佣金费率，如 0.0001')
    p.add_argument('--stamp', type=float, help='卖出印花税率，如 0.0005')
    p.add_argument('--slippage', type=float, help='双边滑点（按成交价比例），如 0.001')
    p.add_argument('--no-limit', action='store_true', help='关闭涨跌停约束（不推荐）')
    p.add_argument('--factor-set', choices=('curated', 'all'),
                   help="因子集：curated=手工挑选，all=手工 ∪ 数据中全部因子")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.signal_date:
        query_signal(args.signal_date)
        sys.exit(0)

    # ── 选择策略 ──────────────────────────────────────────
    # ElasticNet（线性，特征需标准化）
    # s_cfg    = ElasticNetConfig()
    # loader   = DataLoader(s_cfg)
    # strategy = ElasticNetStrategy(s_cfg, loader)

    # LightGBM（非线性，使用原始因子值，截面 rank 标签）
    s_cfg    = LGBMConfig()
    loader   = DataLoader(s_cfg)
    strategy = LGBMStrategy(s_cfg, loader)

    # LightGBM 动态因子（非线性，使用原始因子值，截面 rank 标签）
    # s_cfg    = LGBMDynamicConfig()
    # loader   = DataLoader(s_cfg)
    # strategy = LGBMDynamicStrategy(s_cfg, loader)

    # LightGBM 二分类（专注 Top-K 识别，不做全截面排序）
    # s_cfg    = LGBMBinaryConfig()
    # loader   = DataLoader(s_cfg)
    # strategy = LGBMBinaryStrategy(s_cfg, loader)

    # LGBMRanker（使用 LambdaRank，支持时间衰减）
    # cfg = LGBMRankerConfig()
    # loader = DataLoader(cfg)
    # strategy = LGBMRankerStrategy(cfg, loader)

    # TabNet（深度表格网络，需额外安装 torch + pytorch-tabnet）
    # refit_every 控制重训频率：设为 1 表示每日重训，2120 天回测会非常慢
    # cfg      = TabNetConfig()
    # loader   = DataLoader(cfg)
    # strategy = TabNetStrategy(cfg, loader)

    # IC 加权选股
    # cfg      = ICWeightedConfig()
    # loader   = DataLoader(cfg)
    # strategy = ICWeightedStrategy(cfg, loader)

    # ──────────────────────────────────────────────────────

    # 命令行覆盖因子集（默认由各 Config 自己决定）
    if args.factor_set:
        s_cfg.factor_set = args.factor_set

    if args.ic:
        # 纯因子评估（Rank IC + 分层收益）
        strategy.simple_backtest('20240101', '20250605')

    # ── 回测配置 ──────────────────────────────────────────
    # 交易成本与约束的默认值见 backtest/config.py，命令行参数可覆盖
    b_cfg = BacktestConfig()
    b_cfg.top_n = args.top_n or TOP_N
    b_cfg.holding_period = args.hold or 5
    if args.commission is not None:
        b_cfg.commission = args.commission
    if args.stamp is not None:
        b_cfg.stamp_duty = args.stamp
    if args.slippage is not None:
        b_cfg.slippage = args.slippage
    if args.no_limit:
        b_cfg.enable_limit_check = False

    # 未指定区间时，默认回测截面数据中最近约一年的交易日
    available = loader.get_trading_dates()
    if not available:
        print('  截面数据为空，请先运行 run_signal.py 生成 data/market/ 分区数据')
        sys.exit(1)
    req_start = args.start or available[max(0, len(available) - 250)]
    req_end = args.end or available[-1]
    # 起止日可能不是交易日（如 20260925 中秋休市），对齐到区间内最近的交易日
    start, end = resolve_date_range(available, req_start, req_end)
    n_days = available.index(end) - available.index(start) + 1

    # 择时：指数日线缺失时自动禁用，避免 FileNotFoundError
    if os.path.exists(TIMING_CFG['small_file']):
        timing = MATiming(index_file=TIMING_CFG['small_file'], ma_period=60)
    else:
        print(f"  指数文件缺失（{TIMING_CFG['small_file']}），本次回测不启用择时，始终满仓")
        timing = None
    sell = HoldNDaysSellStrategy(n=b_cfg.holding_period)

    print(f'\n回测区间: {start} ~ {end}（{n_days} 个交易日；本地共 {len(available)} 天，'
          f'最新 {available[-1]}）')
    if (start, end) != (req_start, req_end):
        print(f'  ⚠ 已对齐到最近交易日：{req_start} → {start}，{req_end} → {end}')
    print(f'成本设置: 佣金 {b_cfg.commission:.4%} | 印花税 {b_cfg.stamp_duty:.4%} | '
          f'滑点 {b_cfg.slippage:.4%} | 涨跌停约束 {"开" if b_cfg.enable_limit_check else "关"}')
    print(f'建仓设置: top_n={b_cfg.top_n} | 持有 {b_cfg.holding_period} 个交易日')
    engine = BacktestEngine(strategy, loader, b_cfg, timing=timing, sell_strategy=sell)
    nav_df, trades = engine.run(start, end)
    engine.report(nav_df)
