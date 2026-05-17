from strategy.sell.base_sell_strategy import BaseSellStrategy


class HoldNDaysSellStrategy(BaseSellStrategy):
    """
    固定持有期策略：持有满 N 个交易日后全部清仓，否则继续持有。

    持仓天数以交易日计算（不含买入当日），即买入后第 N 个交易日触发卖出。
    这是最简单的卖出策略，适合作为基准或初始测试。
    """

    def __init__(self, n: int = 5):
        """
        Parameters
        ----------
        n : int  持有期阈值（交易日数），达到或超过此值时清仓
        """
        self.n = n

    def evaluate(self, positions: list[dict], today: str, all_dates: list[str]) -> dict[str, float]:
        """
        计算每只持仓股已持有的交易日数，到期则返回 0.0（清仓），否则返回 1.0（保留）。

        Parameters
        ----------
        positions : list[dict]  持仓列表（含 ts_code、buy_date 等字段）
        today     : str         当日日期 YYYYMMDD
        all_dates : list[str]   完整交易日历

        Returns
        -------
        dict[str, float]  {ts_code: 0.0 或 1.0}
        """
        today_idx = all_dates.index(today)
        result = {}
        for pos in positions:
            buy_date = pos['buy_date']
            buy_price = pos['buy_price']
            current_price = pos['current_price']
            if buy_date not in all_dates:
                # buy_date 不在日历中属于异常情况（数据错误），直接清仓
                result[pos['ts_code']] = 0.0
                continue
            # held = 从买入日到今日经过的交易日数（今日 - 买入日，以索引差计）
            held = today_idx - all_dates.index(buy_date)
            result[pos['ts_code']] = 0.0 if held >= self.n else 1.0
            if (current_price - buy_price) / buy_price < -0.05:
                result[pos['ts_code']] = 0.0

        return result
