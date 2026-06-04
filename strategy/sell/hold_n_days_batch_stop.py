import numpy as np
from typing import Optional
from strategy.sell.base_sell_strategy import BaseSellStrategy


class HoldNDaysBatchStopStrategy(BaseSellStrategy):
    """
    固定持有期 + 虚拟组合止损策略。

    默认行为：持有 n 个交易日后全部清仓。

    止损逻辑（虚拟组合检验）：
      以真实信号日 T 为基准，构造 5 个虚拟持仓 V1~V5：
        Vi: T-(6-i) 日信号 → T-(5-i) 日买入收盘 → T+i 日卖出收盘
      Vi 的持有区间与真实持仓（T+1 买 → T+6 卖）在时间上完全并行，
      是模型在「盲窗口」内的「模拟考试」——模型训练不包含这段数据。

      只检验 V1..V_{check_count}（默认 V1~V3，信号最早、挽回损失最多）：
        - V1 的结果最早在真实持仓建立后的第 1 个交易日（T+2）可得
        - 每日检查所有已完成的 Vi，若其中 ≥ min_trigger 个
          虚拟组合收益 < loss_threshold → 当日收盘全部清仓

    止损触发条件（AND）：
      已完成且在 check_count 范围内的 Vi 中，至少 min_trigger 个收益 < loss_threshold

    时序示意（n=5, label_lookahead=6）：
      T-5 T-4 T-3 T-2 T-1  T  T+1 T+2 T+3 T+4 T+5 T+6
      ├─V1买─────────V1卖┤
           ├─V2买─────────V2卖┤
                ├─V3买─────────V3卖┤
                                ├────真实持仓────────┤
                  (T为信号日，T+1收盘买，T+6收盘卖)
    """

    PRICE_COL = 'close_x'
    STOCK_COL  = 'ts_code'

    def __init__(
        self,
        selection_strategy,
        data_loader,
        top_n: int = 10,
        n: int = 5,
        label_lookahead: int = 6,
        loss_threshold: float = -0.05,
        min_trigger: int = 2,
        check_count: int = 3,
    ):
        """
        Parameters
        ----------
        selection_strategy : 选股策略对象，需实现 generate_signals(date_str)
        data_loader        : DataLoader，用于读取截面收盘价
        top_n              : 虚拟组合股票数（与真实持仓相同）
        n                  : 正常持有期（交易日数），默认 5
        label_lookahead    : 模型盲窗口长度，决定信号日与买入日的偏移，默认 6
        loss_threshold     : 虚拟组合触发阈值，如 -0.05 表示 5 日收益 < -5%
        min_trigger        : 需要多少个虚拟组合同时触发才止损，默认 2
        check_count        : 检查前几个虚拟组合（V1~V{check_count}），默认 3
        """
        self.strategy       = selection_strategy
        self.loader         = data_loader
        self.top_n          = top_n
        self.n              = n
        self.lookahead      = label_lookahead
        self.loss_threshold = loss_threshold
        self.min_trigger    = min_trigger
        self.check_count    = check_count
        # {(signal_date, buy_date, sell_date): Optional[float]}
        self._cache: dict = {}

    # ------------------------------------------------------------------ #
    #  虚拟组合收益计算                                                     #
    # ------------------------------------------------------------------ #

    def _virt_return(self, signal_date: str, buy_date: str, sell_date: str) -> Optional[float]:
        """
        用当前模型对 signal_date 截面打分，取 Top-N，
        计算 buy_date 收盘买入 → sell_date 收盘卖出的等权平均收益。

        使用当前模型（训练至 T-6）对过去截面重新打分，
        评估的是「当前模型的判断逻辑在盲窗口内是否有效」。
        结果缓存，同一虚拟组合只计算一次。
        """
        key = (signal_date, buy_date, sell_date)
        if key in self._cache:
            return self._cache[key]

        signals = self.strategy.generate_signals(signal_date)
        if signals is None or signals.empty:
            self._cache[key] = None
            return None

        top_codes = signals.head(self.top_n)[self.STOCK_COL].tolist()

        buy_df  = self.loader.get_data(buy_date)
        sell_df = self.loader.get_data(sell_date)
        if buy_df is None or sell_df is None:
            self._cache[key] = None
            return None

        buy_prices  = buy_df.set_index(self.STOCK_COL)[self.PRICE_COL]
        sell_prices = sell_df.set_index(self.STOCK_COL)[self.PRICE_COL]

        rets = []
        for code in top_codes:
            if code in buy_prices.index and code in sell_prices.index:
                bp = buy_prices[code]
                sp = sell_prices[code]
                if bp > 0:
                    rets.append(sp / bp - 1)

        result = float(np.mean(rets)) if rets else None
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------ #
    #  主接口                                                               #
    # ------------------------------------------------------------------ #

    def evaluate(
        self, positions: list[dict], today: str, all_dates: list[str]
    ) -> dict[str, float]:
        if not positions:
            return {}

        today_idx = all_dates.index(today)

        # 取最近一次买入日（同批次持仓应相同）
        buy_date = max(
            (pos['buy_date'] for pos in positions if pos['buy_date'] in all_dates),
            key=lambda d: all_dates.index(d),
            default=None,
        )
        if buy_date is None:
            return {pos['ts_code']: 0.0 for pos in positions}

        buy_date_idx = all_dates.index(buy_date)
        days_held    = today_idx - buy_date_idx

        # ── 正常到期清仓 ─────────────────────────────────────────────────
        if days_held >= self.n:
            return {pos['ts_code']: 0.0 for pos in positions}

        # ── 虚拟组合止损检验 ──────────────────────────────────────────────
        # 真实信号日 T 在 buy_date 前一个交易日
        signal_date_idx = buy_date_idx - 1
        if signal_date_idx < 0:
            return {pos['ts_code']: 1.0 for pos in positions}

        # Vi: signal = T-(6-i), buy = T-(5-i), sell = T+i
        # Vi 在 today_idx >= signal_date_idx + i 时已完成
        # 只检查 V1..V{check_count}
        trigger_count = 0
        for vi in range(1, self.check_count + 1):
            vsell_idx = signal_date_idx + vi
            if vsell_idx > today_idx:
                # Vi 尚未完成
                break

            vsig_idx  = signal_date_idx - (self.lookahead - vi)
            vbuy_idx  = signal_date_idx - (self.lookahead - vi - 1)

            if vsig_idx < 0 or vbuy_idx < 0 or vsell_idx >= len(all_dates):
                continue

            sig_d   = all_dates[vsig_idx]
            vbuy_d  = all_dates[vbuy_idx]
            vsell_d = all_dates[vsell_idx]

            ret = self._virt_return(sig_d, vbuy_d, vsell_d)
            if ret is not None and ret < self.loss_threshold:
                trigger_count += 1

        if trigger_count >= self.min_trigger:
            print(
                f"[VirtualPicksStop] {today} 止损触发："
                f"{trigger_count} 个虚拟组合收益 < {self.loss_threshold:.1%}"
            )
            return {pos['ts_code']: 0.0 for pos in positions}

        # ── 继续持仓 ─────────────────────────────────────────────────────
        return {pos['ts_code']: 1.0 for pos in positions}

    def reset(self) -> None:
        """清空虚拟组合收益缓存，回测开始前调用。"""
        self._cache.clear()
