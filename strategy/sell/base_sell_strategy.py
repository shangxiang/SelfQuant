from abc import ABC, abstractmethod


class BaseSellStrategy(ABC):
    """
    卖出策略抽象基类。

    引擎在每个交易日收盘后（已有持仓时）调用 evaluate()，
    根据返回的保留比例决定是否减仓或清仓。
    卖出与买入分离：本类只负责"已有仓位是否卖"，不感知选股逻辑。
    """

    @abstractmethod
    def evaluate(self, positions: list[dict], today: str, all_dates: list[str]) -> dict[str, float]:
        """
        评估当前持仓，返回每只股票的保留比例（keep_ratio）。

        Parameters
        ----------
        positions : list[dict]
            当前持仓列表，每项包含：
              ts_code       - 股票代码（str）
              buy_date      - 买入日期（YYYYMMDD str）
              buy_price     - 买入均价（float）
              shares        - 持仓数量（int）
              current_price - 今日收盘价，由引擎在调用前写入（float）

        today     : str        当日日期，YYYYMMDD 格式
        all_dates : list[str]  完整交易日历，用于计算持仓交易日数等

        Returns
        -------
        dict[str, float]  {ts_code: keep_ratio}
            1.0 → 全部保留（不卖）
            0.0 → 全部卖出（清仓）
            0~1 → 保留对应比例，其余卖出（如 0.5 = 减仓 50%）
        """
        pass
