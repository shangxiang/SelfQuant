class BacktestConfig:
    def __init__(self):
        # 选股参数
        self.top_n = 100

        # 交易成本
        self.commission = 0.0001         # 万分之一（单边）
        self.initial_capital = 1_000_000

        # 波动率控制（use_vol_control=False 则不限制仓位上限）
        self.use_vol_control = False
        self.target_vol = 0.15
        self.vol_window = 20

        # 风控（预留接口，None 表示不启用）
        self.stop_loss = None            # 如 -0.08 表示 8% 止损
        self.take_profit = None          # 如 0.15 表示 15% 止盈
