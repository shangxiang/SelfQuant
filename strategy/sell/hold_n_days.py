from strategy.sell.base_sell_strategy import BaseSellStrategy


class HoldNDaysSellStrategy(BaseSellStrategy):
    """持有满 N 个交易日后全部清仓，否则继续持有。"""

    def __init__(self, n: int = 5):
        self.n = n

    def evaluate(self, positions: list[dict], today: str, all_dates: list[str]) -> dict[str, float]:
        today_idx = all_dates.index(today)
        result = {}
        for pos in positions:
            buy_date = pos['buy_date']
            if buy_date not in all_dates:
                result[pos['ts_code']] = 0.0
                continue
            held = today_idx - all_dates.index(buy_date)
            result[pos['ts_code']] = 0.0 if held >= self.n else 1.0
        return result
