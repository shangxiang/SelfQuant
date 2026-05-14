from abc import ABC, abstractmethod


class BaseSellStrategy(ABC):
    """
    卖出策略抽象基类。
    引擎每日调用 evaluate()，根据返回的保留比例决定减仓或清仓。
    """

    @abstractmethod
    def evaluate(self, positions: list[dict], today: str, all_dates: list[str]) -> dict[str, float]:
        """
        评估当前持仓，返回每只股票的保留比例。

        positions: 持仓列表，每项包含：
            ts_code       - 股票代码
            buy_date      - 买入日期（YYYYMMDD）
            buy_price     - 买入均价
            shares        - 持仓数量
            current_price - 今日收盘价（由引擎在调用前填充）

        today:     当日日期字符串（YYYYMMDD）
        all_dates: 完整交易日历（用于计算持仓天数等）

        返回: {ts_code: keep_ratio}
            1.0 = 全部保留
            0.0 = 全部卖出
            0.5 = 保留一半（减仓50%）
        """
        pass
