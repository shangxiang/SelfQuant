import numpy as np
import pandas as pd

from strategy.selection.base_strategy import BaseStrategy
from strategy.sell.base_sell_strategy import BaseSellStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.timing import BaseTimingStrategy
from typing import Optional


class BacktestEngine:
    """
    模拟交易回测引擎。

    职责：管理资金、模拟买卖、统计绩效。
    不内置任何策略逻辑——全部通过构造参数注入。

    每日流程：
        T 日：更新持仓市值 → 记录 NAV → 执行卖出 → 执行 T-1 日信号的买入 → 生成 T 日信号（供 T+1 消费）
    信号与成交错开一天，避免使用当日收盘价生成信号后立即以当日收盘价成交的未来函数问题。
"""

    PRICE_COL = 'close_x'
    STOCK_COL = 'ts_code'

    def __init__(
        self,
        strategy: BaseStrategy,
        data_loader,
        config,
        timing: Optional[BaseTimingStrategy] = None,
        sell_strategy: Optional[BaseSellStrategy] = None,
    ):
        self.strategy = strategy
        self.loader = data_loader
        self.cfg = config
        self.timing = timing
        # 未指定卖出策略时，默认持有5天清仓
        self.sell_strategy = sell_strategy if sell_strategy is not None else HoldNDaysSellStrategy(5)

    # ------------------------------------------------------------------ #
    #  内部工具                                                             #
    # ------------------------------------------------------------------ #

    def _mark_prices(self, positions: dict, price_index: pd.DataFrame) -> None:
        """将今日收盘价写入各持仓，无价格则沿用成本价"""
        for stock, pos in positions.items():
            pos['current_price'] = (
                price_index.loc[stock, self.PRICE_COL]
                if stock in price_index.index
                else pos['buy_price']
            )

    def _holdings_value(self, positions: dict) -> float:
        return sum(pos['shares'] * pos['current_price'] for pos in positions.values())

    def _vol_ratio(self, daily_nav: list) -> float:
        """波动率目标控制：返回仓位缩放系数"""
        if not self.cfg.use_vol_control or len(daily_nav) < self.cfg.vol_window:
            return 1.0
        nav_s = pd.Series([v[1] for v in daily_nav])
        rv = nav_s.pct_change().dropna().iloc[-self.cfg.vol_window:].std() * np.sqrt(252)
        return min(1.0, self.cfg.target_vol / rv) if rv > 0 else 1.0

    # ------------------------------------------------------------------ #
    #  卖出执行                                                             #
    # ------------------------------------------------------------------ #

    def _execute_sells(
        self, positions: dict, today: str, all_dates: list, trade_log: list
    ) -> float:
        """
        调用卖出策略，执行减仓/清仓，返回卖出所得现金。
        keep_ratio:
            1.0 → 不动
            0.0 → 全卖
            0~1 → 按比例减仓
        """
        pos_list = [{'ts_code': s, **pos} for s, pos in positions.items()]
        keep_ratios = self.sell_strategy.evaluate(pos_list, today, all_dates)

        cash_from_sells = 0.0
        for stock, ratio in keep_ratios.items():
            if stock not in positions or ratio >= 1.0:
                continue
            pos = positions[stock]
            if pos.get('current_price') is None:
                continue

            sell_shares = pos['shares'] if ratio == 0.0 else int(pos['shares'] * (1 - ratio))
            # print("stock:", stock)
            # print("sell_shares", sell_shares)
            if sell_shares <= 0:
                continue

            sell_price = pos['current_price']
            proceeds = sell_shares * sell_price * (1 - self.cfg.commission)
            cash_from_sells += proceeds
            trade_log.append((today, 'SELL', stock, sell_shares, sell_price, proceeds))

            pos['shares'] -= sell_shares
            if pos['shares'] <= 0:
                del positions[stock]

        return cash_from_sells

    # ------------------------------------------------------------------ #
    #  买入执行                                                             #
    # ------------------------------------------------------------------ #

    def _execute_buys(
        self, today: str, invest_cash: float, signals: pd.DataFrame,
        positions: dict, price_index: pd.DataFrame, trade_log: list
    ) -> float:
        """按前一日信号、以今日收盘价等权买入 top_n 只股票，返回实际花费现金。"""
        top = signals.head(min(self.cfg.top_n, len(signals)))
        n = len(top)
        if n == 0:
            return 0.0

        per_stock = invest_cash / n
        cash_spent = 0.0
        for _, row in top.iterrows():
            stock = row[self.STOCK_COL]
            if stock not in price_index.index:
                continue
            buy_price = price_index.loc[stock, self.PRICE_COL]
            shares = int(per_stock / buy_price / 100) * 100
            if shares <= 0:
                continue
            cost = shares * buy_price * (1 + self.cfg.commission)
            cash_spent += cost
            positions[stock] = {
                'buy_date': today,
                'buy_price': buy_price,
                'shares': shares,
                'current_price': buy_price,
            }
            trade_log.append((today, 'BUY', stock, shares, buy_price, cost))

        return cash_spent

    # ------------------------------------------------------------------ #
    #  主循环                                                               #
    # ------------------------------------------------------------------ #

    def run(self, start_date: str, end_date: str, capital: float = None) -> tuple:
        """
        执行回测，返回 (nav_df, trade_log)。
          nav_df    : DataFrame[date, nav]
          trade_log : list of (date, 'BUY'|'SELL', ts_code, shares, price, amount)
        """
        if capital is None:
            capital = self.cfg.initial_capital

        all_dates = self.loader.get_trading_dates()
        trade_dates = all_dates[all_dates.index(start_date): all_dates.index(end_date) + 1]

        if self.timing is not None:
            self.timing.prepare(start_date, end_date)
        self.strategy.reset()

        cash = capital
        positions: dict = {}
        daily_nav: list = []
        trade_log: list = []
        pending_buy: Optional[tuple] = None  # (signals_df, timing_ratio) from previous day

        for today in trade_dates:
            df_today = self.loader.get_data(today)
            if df_today is None:
                daily_nav.append((today, daily_nav[-1][1] if daily_nav else cash))
                continue

            price_index = df_today.set_index(self.STOCK_COL)

            # ① 更新持仓市值
            self._mark_prices(positions, price_index)

            # ② 记录当日 NAV（交易前估值）
            daily_nav.append((today, cash + self._holdings_value(positions)))

            # ③ 执行卖出（用今日收盘价）
            if positions:
                cash += self._execute_sells(positions, today, all_dates, trade_log)

            # ④ 执行前一日信号的买入（用今日收盘价成交，信号来自昨日收盘后，只有清仓了才执行，假设有剩余仓位，即便能买也不买，省的麻烦）
            if not positions:
                if pending_buy is not None:
                    signals, timing_ratio = pending_buy
                    vol_scale = self._vol_ratio(daily_nav)
                    invest = cash * timing_ratio * vol_scale
                    print("today is ", today)
                    cash -= self._execute_buys(today, invest, signals, positions, price_index, trade_log)
                pending_buy = None

            # ⑤ 用今日数据训练并生成信号，供明日买入消费
            if self.strategy.fit(today):
                signals = self.strategy.generate_signals(today)
                if signals is not None and not signals.empty:
                    t_ratio = self.timing.get_position_ratio(today) if self.timing else 1.0
                    if t_ratio > 0:
                        pending_buy = (signals, t_ratio)

            # ⑥ 交易后更新当日 NAV
            daily_nav[-1] = (today, cash + self._holdings_value(positions))
            print("cash is ", cash)
            print("positions is ", positions)
            prof = 0
            for stock in positions:
                prof += positions[stock]['shares'] * positions[stock]['current_price']
            print("prof is ", prof)
            print("今天总共", cash + prof)

        # ---- 回测结束，强制平仓剩余持仓 ----
        if positions:
            last_date = trade_dates[-1]
            df_last = self.loader.get_data(last_date)
            if df_last is not None:
                price_index_last = df_last.set_index(self.STOCK_COL)
                for stock, pos in list(positions.items()):
                    price = (
                        price_index_last.loc[stock, self.PRICE_COL]
                        if stock in price_index_last.index
                        else pos['buy_price']
                    )
                    cash += pos['shares'] * price * (1 - self.cfg.commission)
                positions.clear()
                if daily_nav:
                    daily_nav[-1] = (last_date, cash)

        nav_df = pd.DataFrame(daily_nav, columns=['date', 'nav'])
        return nav_df, trade_log

    # ------------------------------------------------------------------ #
    #  绩效报告                                                             #
    # ------------------------------------------------------------------ #

    def report(self, nav_df: pd.DataFrame, capital: float = None) -> None:
        """打印绩效报告：总收益、年化收益、夏普、最大回撤、日胜率"""
        if capital is None:
            capital = self.cfg.initial_capital
        nav = nav_df.copy()
        nav['ret'] = nav['nav'].pct_change()
        std_ret = nav['ret'].std()
        sharpe = (nav['ret'].mean() / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0
        total_return = nav['nav'].iloc[-1] / capital - 1
        nav['drawdown'] = (nav['nav'] - nav['nav'].cummax()) / nav['nav'].cummax()
        ann_return = (nav['nav'].iloc[-1] / capital) ** (252 / len(nav)) - 1

        print("========== 回测报告 ==========")
        print(f"初始资金:     {capital:>15,.0f}")
        print(f"最终资金:     {nav['nav'].iloc[-1]:>15,.0f}")
        print(f"总收益率:     {total_return:>14.2%}")
        print(f"年化收益率:   {ann_return:>14.2%}")
        print(f"年化夏普:     {sharpe:>14.4f}")
        print(f"最大回撤:     {nav['drawdown'].min():>14.2%}")
        print(f"日胜率:       {(nav['ret'] > 0).mean():>14.2%}")
