from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class BaseStrategy(ABC):
    """
    截面选股策略抽象基类。

    子类只需实现 fit() 和 generate_signals()，引擎负责调用时序和资金分配。
    设计约束：
      - fit() 和 generate_signals() 均只能使用 date_str 当日及之前的数据，
        严禁引用未来信息（point-in-time 原则）。
      - generate_signals() 不依赖 label 列，使训练（fit）和推理（generate_signals）
        在实盘环境中可以分离运行。
    """

    @abstractmethod
    def fit(self, date_str: str) -> bool:
        """
        使用 date_str 当日及之前的历史数据训练或更新模型。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期

        Returns
        -------
        bool  True 表示训练成功，False 表示数据不足或该日非交易日
        """
        pass

    @abstractmethod
    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        用当前已训练的模型对 date_str 当日全量股票打分。

        返回的 DataFrame 至少包含两列：
          ts_code : str    股票代码
          score   : float  得分，越高表示越值得买入

        按 score 降序排列，引擎取 top_n 行建仓。
        返回 None 表示当日无法生成信号（模型未训练、数据缺失等）。
        注意：此方法不得使用 label 列，以保证实盘可用性。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期
        """
        pass

    def reset(self) -> None:
        """
        重置策略内部状态（如已训练的权重、缓存等）。
        引擎在每次独立回测开始前调用，防止多次 run() 之间状态污染。
        子类若有持久状态（模型权重等），需覆盖此方法清空。
        """
        pass

    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        纯信号质量评估（不涉及资金和交易），计算 Rank IC、分层多空收益等指标。
        子类可选覆盖以提供具体实现，默认抛出 NotImplementedError。

        Parameters
        ----------
        start_date : str  回测开始日期 YYYYMMDD
        end_date   : str  回测结束日期 YYYYMMDD

        Returns
        -------
        dict  评估结果，具体结构由子类定义
        """
        raise NotImplementedError
