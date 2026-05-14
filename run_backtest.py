"""
回测主入口。从项目根目录执行：
    python run_backtest.py
"""
from strategy.data_loader import DataLoader
from strategy.selection.elastic_net_strategy import ElasticNetConfig, ElasticNetStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.timing import MATiming


if __name__ == '__main__':
    s_cfg = ElasticNetConfig()
    b_cfg = BacktestConfig()
    b_cfg.top_n = 100

    loader = DataLoader(s_cfg)
    strategy = ElasticNetStrategy(s_cfg, loader)

    # 纯因子评估（Rank IC + 分层收益）
    strategy.simple_backtest('20230101', '20260331')

    # 模拟交易回测
    timing = MATiming(
        index_file='data/raw/index_daily/000905.SH.csv',
        ma_period=60,
    )
    sell = HoldNDaysSellStrategy(n=5)

    engine = BacktestEngine(strategy, loader, b_cfg, timing=timing, sell_strategy=sell)
    nav_df, trades = engine.run('20230103', '20260331')
    engine.report(nav_df)
