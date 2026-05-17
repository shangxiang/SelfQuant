from strategy.sell.base_sell_strategy import BaseSellStrategy


class HoldNDaysBatchStopStrategy(BaseSellStrategy):
    """
    固定持有期 + 个股止损 + 批次止损策略。

    卖出触发条件（优先级从高到低）：
      1. 批次止损：持仓中 >= batch_stop_ratio 比例的股票浮亏，则**全部清仓**
      2. 个股止损：单只股票浮亏超过 stop_loss_pct，仅卖出该股
      3. 到期清仓：持仓满 n 个交易日，仅卖出该股

    批次止损优先级最高：只要触发，无论其他股票是否盈利、是否到期，
    当日全部卖出。
    """

    def __init__(
        self,
        n: int = 5,
        stop_loss_pct: float = 0.05,
        batch_stop_ratio: float = 0.60,
    ):
        """
        Parameters
        ----------
        n               : int    最长持有期（交易日数）
        stop_loss_pct   : float  个股止损阈值，如 0.05 表示下跌 5% 即触发
        batch_stop_ratio: float  批次止损触发比例，如 0.60 表示 ≥60% 个股亏损时全仓清出
        """
        self.n = n
        self.stop_loss_pct = stop_loss_pct
        self.batch_stop_ratio = batch_stop_ratio

    def evaluate(
        self, positions: list[dict], today: str, all_dates: list[str]
    ) -> dict[str, float]:
        if not positions:
            return {}

        # ── 批次止损检查（优先）──────────────────────────────────────
        n_loss = sum(
            1 for pos in positions
            if pos['current_price'] < pos['buy_price']
        )
        if n_loss / len(positions) >= self.batch_stop_ratio:
            return {pos['ts_code']: 0.0 for pos in positions}

        # ── 逐只检查：到期 + 个股止损 ────────────────────────────────
        today_idx = all_dates.index(today)
        result = {}
        for pos in positions:
            buy_date = pos['buy_date']
            if buy_date not in all_dates:
                result[pos['ts_code']] = 0.0
                continue
            held = today_idx - all_dates.index(buy_date)
            ret  = (pos['current_price'] - pos['buy_price']) / pos['buy_price']
            if held >= self.n or ret < -self.stop_loss_pct:
                result[pos['ts_code']] = 0.0
            else:
                result[pos['ts_code']] = 1.0

        return result
