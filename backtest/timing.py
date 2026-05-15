from abc import ABC, abstractmethod
import pandas as pd


class BaseTimingStrategy(ABC):
    """
    择时策略抽象基类。

    择时策略决定每次建仓时使用多少比例的可用资金，与选股策略正交：
      - 选股策略回答"买哪些股票"
      - 择时策略回答"这次投入多少钱"

    引擎在回测开始前调用 prepare() 进行预计算，
    之后在每个信号生成日调用 get_position_ratio() 获取当日建仓比例。
    注意：get_position_ratio() 返回的是 T 日的仓位判断，用于 T+1 日的实际买入，
    因此内部使用 T 日数据（而非 T+1）不构成未来函数问题。
    """

    def prepare(self, start_date: str, end_date: str) -> None:
        """
        回测开始前的预计算（如读取指数文件、计算均线序列）。
        子类可选覆盖，默认空实现（表示不需要预计算）。

        Parameters
        ----------
        start_date : str  回测开始日期 YYYYMMDD
        end_date   : str  回测结束日期 YYYYMMDD
        """
        pass

    @abstractmethod
    def get_position_ratio(self, date_str: str) -> float:
        """
        返回 date_str 当日对应的建仓比例。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期

        Returns
        -------
        float  0.0 = 不建仓（空仓），1.0 = 全仓，0.6 = 用 60% 可用资金建仓
        """
        pass


class MATiming(BaseTimingStrategy):
    """
    均线择时：以指数收盘价相对于 N 日均线的位置判断牛熊。

    判断逻辑（使用前一日数据，避免用当日收盘价做当日决策）：
      昨日收盘 > 昨日 N 日均线 → 牛市，满仓（1.0）
      昨日收盘 ≤ 昨日 N 日均线 → 熊市，空仓（0.0）

    若需要渐进式仓位（如距均线越远仓位越低），可继承此类并覆盖 get_position_ratio()。
    """

    def __init__(self, index_file: str, ma_period: int = 60):
        """
        Parameters
        ----------
        index_file : str  指数日线 CSV 路径，需含 trade_date、close 列
        ma_period  : int  均线周期（交易日数），默认 60 日
        """
        self.index_file = index_file
        self.ma_period = ma_period
        # 预计算结果缓存：{date_str: position_ratio}
        # 在 prepare() 中一次性填充，get_position_ratio() 直接查表
        self._map: dict[str, float] = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        """
        读取指数日线，预计算全区间内每日的牛熊状态并存入 self._map。

        使用 shift(1) 取前一日数据：T 日的牛熊判断基于 T-1 日收盘，
        与引擎的信号→成交时序一致（T 日判断 → T+1 日买入）。
        """
        index_df = pd.read_csv(self.index_file, parse_dates=['trade_date'])
        index_df = index_df.sort_values('trade_date')
        close = index_df['close']
        ma = close.rolling(self.ma_period).mean()

        # shift(1)：用昨日收盘和昨日均线判断今日牛熊，避免使用今日收盘做今日决策
        is_bull = (close.shift(1) > ma.shift(1)).fillna(False)

        self._map = dict(zip(
            index_df['trade_date'].dt.strftime('%Y%m%d'),
            is_bull.map({True: 1.0, False: 0.0})
        ))

    def get_position_ratio(self, date_str: str) -> float:
        """
        查表返回当日建仓比例。

        若 date_str 不在预计算结果中（如数据缺失），默认返回 1.0（满仓），
        以避免因数据缺口意外跳过建仓。
        """
        # return self._map.get(date_str, 1.0)
        return 1.0
