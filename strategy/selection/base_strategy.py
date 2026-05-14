from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class BaseStrategy(ABC):

    @abstractmethod
    def fit(self, date_str: str) -> bool:
        """用 date_str 之前的数据训练/更新模型，成功返回 True"""
        pass

    @abstractmethod
    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        返回当日所有股票的打分 DataFrame。
        至少包含列: [ts_code, score]，按 score 降序排列。
        返回 None 表示当日无法生成信号（数据不足、模型未训练等）。
        """
        pass

    def reset(self) -> None:
        """重置内部状态（如已训练的权重），在每次新的独立回测前调用"""
        pass

    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        纯信号质量评估：Rank IC、分层多空收益。
        不涉及交易模拟，子类可选覆盖。
        """
        raise NotImplementedError
