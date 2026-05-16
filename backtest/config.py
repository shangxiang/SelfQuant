class BacktestConfig:
    """
    回测引擎参数配置。

    只包含与交易模拟相关的参数，策略超参数由各策略自身的 Config 类管理。
    """

    def __init__(self):
        # ---- 选股参数 ----
        # 每次建仓选取打分最高的前 N 只股票，等权分配资金
        self.top_n: int = 100

        # ---- 交易成本 ----
        # 单边佣金费率（买入和卖出各收一次），万分之一对应互联网券商折扣价
        self.commission: float = 0.0001     # 万分之一，单边
        # 初始资金（元）
        self.initial_capital: float = 1_000_000

        # ---- 风控（预留接口，None 表示当前不启用）----
        # 个股止损阈值：当持仓浮亏达到此比例时触发清仓，如 -0.08 表示 8% 止损
        # 需在自定义 SellStrategy 中读取并使用，引擎本身不处理
        self.stop_loss: float | None = None

        # 个股止盈阈值：当持仓浮盈达到此比例时触发清仓，如 0.15 表示 15% 止盈
        # 需在自定义 SellStrategy 中读取并使用，引擎本身不处理
        self.take_profit: float | None = None

        # ---- 结果输出 ----
        # 回测结果保存目录；None = 自动生成 backtest/results/<start>_<end>_<timestamp>/
        # 设为空字符串 "" 可禁用文件输出
        self.result_dir: str | None = None
