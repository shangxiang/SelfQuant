"""
回测主入口。从项目根目录执行：
    python run_backtest.py
"""
from strategy.data_loader import DataLoader
from strategy.selection.elastic_net_strategy import ElasticNetConfig, ElasticNetStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.timing import MATiming, StyleConvergenceTiming


if __name__ == '__main__':
    s_cfg = ElasticNetConfig()
    b_cfg = BacktestConfig()
    b_cfg.top_n = 10

    loader = DataLoader(s_cfg)
    strategy = ElasticNetStrategy(s_cfg, loader)

    # # 纯因子评估（Rank IC + 分层收益）
    # strategy.simple_backtest('20250101', '20260507')

    # 均线择时（原方案）
    # timing = MATiming(
    #     index_file='data/raw/index_daily/399005.SZ.csv',
    #     ma_period=60,
    # )

    # 大小盘风格趋同择时：高相关 + 双双下行时空仓
    timing = StyleConvergenceTiming(
        roll_window=5,        # 滚动窗口（交易日）
        corr_threshold=0.75,  # 相关系数触发阈值
        avoid_ratio=0.0,      # 触发时建仓比例（0.0=空仓，0.5=半仓）
    )
    sell = HoldNDaysSellStrategy(n=5)

    engine = BacktestEngine(strategy, loader, b_cfg, timing=timing, sell_strategy=sell)
    nav_df, trades = engine.run('20230103', '20260331')
    engine.report(nav_df)
