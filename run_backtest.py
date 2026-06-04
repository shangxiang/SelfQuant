"""
回测主入口。从项目根目录执行：
    python run_backtest.py

也可以传入日期查询当日选股和择时结果：
    python run_backtest.py 20260522
"""
import sys
import pandas as pd
from strategy.data_loader import DataLoader
from strategy.selection.elastic_net_strategy import ElasticNetConfig, ElasticNetStrategy
from strategy.selection.lgbm_strategy import LGBMConfig, LGBMStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.timing import MATiming, StyleConvergenceTiming, LastBatchTiming, BlindWindowTiming, BlindWindowVolTiming
from strategy.selection.lgbm_ranker_strategy import LGBMRankerConfig, LGBMRankerStrategy
from strategy.selection.ic_weighted_strategy import ICWeightedConfig, ICWeightedStrategy
from strategy.sell.hold_n_days_batch_stop import HoldNDaysBatchStopStrategy


TOP_N = 10

TIMING_CFG = dict(
    small_file='data/raw/index_daily/932000.CSI.csv',
    large_file='data/raw/index_daily/000510.CSI.csv',
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


if __name__ == '__main__':
    if len(sys.argv) == 2:
        query_signal(sys.argv[1])
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

    # LGBMRanker（使用 LambdaRank，支持时间衰减）
    # cfg = LGBMRankerConfig()
    # loader = DataLoader(cfg)
    # strategy = LGBMRankerStrategy(cfg, loader)

    # IC 加权选股
    # cfg      = ICWeightedConfig()
    # loader   = DataLoader(cfg)
    # strategy = ICWeightedStrategy(cfg, loader)

    # ──────────────────────────────────────────────────────

    b_cfg        = BacktestConfig()
    b_cfg.top_n  = 10

    # 纯因子评估（Rank IC + 分层收益）
    # strategy.simple_backtest('20240101', '20250520')

    # 均线择时（原方案）
    timing = MATiming(
        index_file='data/raw/index_daily/932000.CSI.csv',
        ma_period=60,
    )
    # timing = BlindWindowVolTiming(
    #     vol_low_pct  = 20,   # 只有最低 20% 波动率才空仓（而非 33%）
    #     vol_high_pct = 20,   # 与 low 相同 → 退化为两档：vol < 20th → 0，否则 → 1
    #     avoid_ratio  = 0.0,
    #     mid_ratio    = 1.0,  # 中间区间也满仓
    # )
    # timing = BlindWindowTiming(
    #     small_file='data/raw/index_daily/932000.CSI.csv',
    #     large_file='data/raw/index_daily/000510.CSI.csv',
    #     corr_pct=20.0,   # corr_5d 低于历史 20% 分位 → 空仓
    #     vol_pct=33.0,    # l_vol   低于历史 33% 分位 → 空仓
    #     avoid_ratio=0.0, # 触发时空仓（改成 0.5 则半仓）
    #     rolling_window=80,
    #     trend_ma=0,
    # )

    # # 大小盘风格趋同择时：高相关 + 双双下行时空仓
    # # timing = StyleConvergenceTiming(
    # #     roll_window=5,        # 滚动窗口（交易日）
    # #     corr_threshold=0.75,  # 相关系数触发阈值
    # #     avoid_ratio=0.0,      # 触发时建仓比例（0.0=空仓，0.5=半仓）
    # # )
    sell = HoldNDaysSellStrategy(n=5)

    engine = BacktestEngine(strategy, loader, b_cfg, timing=timing, sell_strategy=sell)
    nav_df, trades = engine.run('20240103', '20260522')
    engine.report(nav_df)
