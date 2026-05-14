from abc import ABC, abstractmethod
import pandas as pd


class BaseTimingStrategy(ABC):
    """
    择时策略抽象基类。
    BacktestEngine 在回测开始前调用 prepare()，之后每日买入时调用 get_position_ratio()。
    """

    def prepare(self, start_date: str, end_date: str) -> None:
        """回测开始前的预计算（如读取指数、计算均线）。子类可选覆盖。"""
        pass

    @abstractmethod
    def get_position_ratio(self, date_str: str) -> float:
        """
        返回当日建仓比例。
        0.0 = 不建仓，1.0 = 用全部可用资金建仓，0.6 = 用60%资金建仓。
        """
        pass


class MATiming(BaseTimingStrategy):
    """
    均线择时：指数昨日收盘 > N 日均线 → 满仓(1.0)，否则空仓(0.0)。
    如需渐进式仓位，可继承此类覆盖 get_position_ratio。
    """

    def __init__(self, index_file: str, ma_period: int = 60):
        self.index_file = index_file
        self.ma_period = ma_period
        self._map: dict[str, float] = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        index_df = pd.read_csv(self.index_file, parse_dates=['trade_date'])
        index_df = index_df.sort_values('trade_date')
        close = index_df['close']
        ma = close.rolling(self.ma_period).mean()
        is_bull = (close.shift(1) > ma.shift(1)).fillna(False)
        self._map = dict(zip(
            index_df['trade_date'].dt.strftime('%Y%m%d'),
            is_bull.map({True: 1.0, False: 0.0})
        ))

    def get_position_ratio(self, date_str: str) -> float:
        return self._map.get(date_str, 1.0)
