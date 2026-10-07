from typing import Optional

class BacktestConfig:
    """
    回测引擎参数配置。

    只包含与交易模拟相关的参数，策略超参数由各策略自身的 Config 类管理。
    """

    def __init__(self):
        # ---- 选股参数 ----
        # 每次建仓选取打分最高的前 N 只股票，等权分配资金
        self.top_n: int = 100

        # 固定持有期（交易日数），同时作为默认卖出策略 HoldNDaysSellStrategy 的参数
        self.holding_period: int = 5

        # ---- 交易成本 ----
        # 单边佣金费率（买入和卖出各收一次），万分之一对应互联网券商折扣价
        self.commission: float = 0.0001     # 万分之一，单边

        # 证券交易印花税，A 股自 2023-08-28 起为 0.05%，仅卖出时单边收取
        # 5 日换手一年约 50 个来回 → 仅印花税就有 50 × 0.05% = 2.5%/年
        self.stamp_duty: float = 0.0005     # 万分之五，仅卖出

        # 滑点：按成交价的比例双向扣减（买入价上浮、卖出价下浮）
        # 覆盖买卖价差 + 小额冲击成本。小市值股票建议 0.001~0.002，
        # 流动性好的大盘股可降到 0.0005。设为 0 则不计滑点。
        self.slippage: float = 0.001        # 千分之一，双边

        # 初始资金（元）
        self.initial_capital: float = 1_000_000

        # ---- 涨跌停约束 ----
        # A 股主板 ±10%（ST 为 ±5%，但 ST 已被 DataLoader 过滤掉）
        # T+1 以收盘价成交，若当日涨停则实际买不进，跌停则卖不出。
        # 用 9.8 / -9.8 而不是 10.0，是为了容纳价格四舍五入带来的偏差。
        self.enable_limit_check: bool = True
        self.limit_up_pct: float = 9.8      # 涨幅 ≥ 此值视为涨停，不可买入
        self.limit_down_pct: float = -9.8   # 跌幅 ≤ 此值视为跌停，不可卖出

        # 模型强度过滤：top-1 score 低于此阈值时跳过建仓（视为模型失效）
        # None 或 0 表示不过滤
        self.min_score_threshold: float = 0.001

        # ---- 风控（预留接口，None 表示当前不启用）----
        # 个股止损阈值：当持仓浮亏达到此比例时触发清仓，如 -0.08 表示 8% 止损
        # 需在自定义 SellStrategy 中读取并使用，引擎本身不处理
        self.stop_loss: Optional[float] = None

        # 个股止盈阈值：当持仓浮盈达到此比例时触发清仓，如 0.15 表示 15% 止盈
        # 需在自定义 SellStrategy 中读取并使用，引擎本身不处理
        self.take_profit: Optional[float] = None

        # ---- 结果输出 ----
        # 回测结果保存目录；None = 自动生成 backtest/results/<start>_<end>_<timestamp>/
        # 设为空字符串 "" 可禁用文件输出
        self.result_dir: Optional[str] = None
