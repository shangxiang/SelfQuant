import os
import numpy as np
import pandas as pd
from datetime import datetime

from strategy.selection.base_strategy import BaseStrategy
from strategy.sell.base_sell_strategy import BaseSellStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.timing import BaseTimingStrategy
from typing import Optional


class BacktestEngine:
    """
    模拟交易回测引擎。

    职责：管理资金、模拟买卖、统计绩效。
    不内置任何策略逻辑——选股、择时、卖出全部通过构造参数注入，可自由替换。

    信号与成交时序（避免未来函数）：
        T 日收盘后：fit(T) + generate_signals(T) → 存入 pending_buy
        T+1 日收盘：用 pending_buy 里的信号以 T+1 收盘价成交

    每日完整流程：
        ① 用今日收盘价更新持仓市值
        ② 记录今日 NAV（交易前估值，反映昨日持仓的当日价值变化）
        ③ 执行卖出策略（若有持仓）
        ④ 执行前一日信号的买入（仅在全部清仓后触发，避免新旧持仓混仓）
        ⑤ 用今日数据生成明日信号
        ⑥ 更新今日 NAV（交易后）
    """

    # 截面 CSV 中的收盘价列名（与 stock_list 合并后 close 变为 close_x）
    PRICE_COL = 'close_x'
    # 股票代码列名，用于 DataFrame 索引和 position dict 的 key
    STOCK_COL = 'ts_code'

    def __init__(
        self,
        strategy: BaseStrategy,
        data_loader,
        config,
        timing: Optional[BaseTimingStrategy] = None,
        sell_strategy: Optional[BaseSellStrategy] = None,
    ):
        """
        Parameters
        ----------
        strategy      : BaseStrategy          选股策略，负责 fit 和 generate_signals
        data_loader   : DataLoader            截面数据加载器
        config        : BacktestConfig        回测参数（top_n、commission 等）
        timing        : BaseTimingStrategy    择时策略，None 表示始终满仓
        sell_strategy : BaseSellStrategy      卖出策略，None 时默认持有5天清仓
        """
        self.strategy = strategy
        self.loader = data_loader
        self.cfg = config
        self.timing = timing
        # 未指定卖出策略时，默认按 config.holding_period 持有后全部清仓
        self.sell_strategy = sell_strategy if sell_strategy is not None else HoldNDaysSellStrategy(config.holding_period)

    # ------------------------------------------------------------------ #
    #  内部工具                                                             #
    # ------------------------------------------------------------------ #

    def _mark_prices(self, positions: dict, price_index: pd.DataFrame) -> None:
        """
        将今日收盘价写入各持仓的 current_price 字段。
        若当日该股票无行情数据（停牌等），保留买入价作为估值，避免 NAV 出现 None。

        Parameters
        ----------
        positions   : dict           {ts_code: pos_dict}，直接修改 current_price 字段
        price_index : pd.DataFrame   以 ts_code 为索引的当日截面数据
        """
        for stock, pos in positions.items():
            pos['current_price'] = (
                price_index.loc[stock, self.PRICE_COL]
                if stock in price_index.index
                else pos['buy_price']
            )

    def _holdings_value(self, positions: dict) -> float:
        """计算当前持仓总市值（按 current_price 估值）。"""
        return sum(pos['shares'] * pos['current_price'] for pos in positions.values())

    # ------------------------------------------------------------------ #
    #  卖出执行                                                             #
    # ------------------------------------------------------------------ #

    def _execute_sells(
        self, positions: dict, today: str, all_dates: list, trade_log: list
    ) -> float:
        """
        调用卖出策略获取各持仓的 keep_ratio，按比例卖出对应股份，返回卖出所得现金。

        keep_ratio 含义：
            1.0 → 不动
            0.0 → 全部卖出
            0~1 → 保留对应比例，其余卖出

        部分卖出时，sell_shares 向下取整到整数股（未考虑手数约束，精度误差极小）。
        卖出后 pos['shares'] 降为 0 的持仓从 positions 中删除。

        Parameters
        ----------
        positions : dict        当前持仓字典，原地修改
        today     : str         当日日期 YYYYMMDD
        all_dates : list[str]   完整交易日历
        trade_log : list        交易记录，追加 SELL 记录

        Returns
        -------
        float  本次卖出所得现金总额（扣除佣金后）
        """
        # 将 dict 转为 list[dict] 传给卖出策略（策略接口约定）
        pos_list = [{'ts_code': s, **pos} for s, pos in positions.items()]
        keep_ratios = self.sell_strategy.evaluate(pos_list, today, all_dates)

        cash_from_sells = 0.0
        for stock, ratio in keep_ratios.items():
            if stock not in positions or ratio >= 1.0:
                continue
            pos = positions[stock]
            if pos.get('current_price') is None:
                continue

            # ratio=0.0 时全卖，否则按 (1-ratio) 比例卖出
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
        positions: dict, price_index: pd.DataFrame, trade_log: list,
        picks_log: list
    ) -> float:
        """
        根据前一日生成的信号，以今日收盘价等权买入 top_n 只股票。

        等权分配：invest_cash 按信号 top_n 均分，不足整手（100股）则跳过。
        买入后将新持仓写入 positions，记录买入日期供卖出策略计算持仓天数。

        Parameters
        ----------
        today        : str           今日日期 YYYYMMDD（作为新持仓的 buy_date）
        invest_cash  : float         本次可用于建仓的资金
        signals      : pd.DataFrame  前一日生成的打分表（含 ts_code、score 列，已降序）
        positions    : dict          当前持仓，原地写入新持仓
        price_index  : pd.DataFrame  以 ts_code 为索引的今日截面
        trade_log    : list          交易记录，追加 BUY 记录
        picks_log    : list          选股记录，追加 (date, ts_code, score, buy_price, shares)

        Returns
        -------
        float  实际花费的现金总额（含佣金）
        """
        # 取分数最高的 top_n 只股票
        top = signals.head(min(self.cfg.top_n, len(signals)))
        n = len(top)
        if n == 0:
            return 0.0

        # 买入当天 pct_chg 列名（Tushare 标准字段）
        pct_col = next((c for c in ('pct_chg', 'pct_change') if c in price_index.columns), None)

        # 每只股票等权分配的资金
        per_stock = invest_cash / n
        cash_spent = 0.0
        for _, row in top.iterrows():
            stock = row[self.STOCK_COL]
            score   = row.get('score', float('nan'))
            pick_mv = row.get('_pick_mv', float('nan'))
            if stock not in price_index.index:
                # 今日该股票无行情（停牌等），跳过
                continue
            buy_price = price_index.loc[stock, self.PRICE_COL]
            # 买入当天涨幅（T+1 日，即实际成交日）
            pct_chg = price_index.loc[stock, pct_col] if pct_col else float('nan')
            # 按 100 股/手取整，A 股最小交易单位为 1 手（100 股）
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
            picks_log.append((today, stock, score, buy_price, shares, pick_mv, pct_chg))

        return cash_spent

    # ------------------------------------------------------------------ #
    #  主循环                                                               #
    # ------------------------------------------------------------------ #

    def run(self, start_date: str, end_date: str, capital: float = None) -> tuple:
        """
        执行全量回测，返回每日净值序列和完整交易记录。

        Parameters
        ----------
        start_date : str    回测开始日期 YYYYMMDD
        end_date   : str    回测结束日期 YYYYMMDD
        capital    : float  初始资金，None 时使用 config.initial_capital

        Returns
        -------
        nav_df    : DataFrame[date, nav]
            每日净值，date 为 YYYYMMDD 字符串，nav 为当日结束时的总资产
        trade_log : list of tuple
            每条记录格式：(date, 'BUY'|'SELL', ts_code, shares, price, amount)
        """
        if capital is None:
            capital = self.cfg.initial_capital

        all_dates = self.loader.get_trading_dates()
        # 截取回测区间内的交易日列表
        trade_dates = all_dates[all_dates.index(start_date): all_dates.index(end_date) + 1]

        if self.timing is not None:
            # 预计算全区间内的择时状态（如均线序列），避免在每日循环内重复读文件
            self.timing.prepare(start_date, end_date)
        self.strategy.reset()

        cash = capital                          # 当前账户现金
        positions: dict = {}                    # {ts_code: {buy_date, buy_price, shares, current_price}}
        daily_nav: list = []                    # [(date, nav), ...]，最终转为 DataFrame
        trade_log: list = []                    # 完整成交记录
        picks_log: list = []                    # 选股打分记录：(date, ts_code, score, buy_price, shares)
        pending_buy: Optional[tuple] = None     # T 日生成的信号，在 T+1 日成交

        # 下一次需要生成信号的 trade_dates 下标
        # 空仓期间每天都要生成（随时准备建仓）；持仓期间只在卖出前一天生成一次
        next_signal_idx: int = 0

        for i, today in enumerate(trade_dates):
            df_today = self.loader.get_data(today)
            if df_today is None:
                # 当日无截面数据（节假日、数据缺失），NAV 沿用前一日
                daily_nav.append((today, daily_nav[-1][1] if daily_nav else cash))
                continue

            # 以 ts_code 为索引，便于后续按股票代码快速查价格
            price_index = df_today.set_index(self.STOCK_COL)

            # ① 用今日收盘价更新所有持仓的 current_price
            self._mark_prices(positions, price_index)

            # ② 记录今日 NAV（交易前，反映持仓资产的价格变动）
            daily_nav.append((today, cash + self._holdings_value(positions)))

            # ③ 执行卖出策略（用今日收盘价结算）
            if positions:
                cash += self._execute_sells(positions, today, all_dates, trade_log)

            # ④ 执行前一日信号的买入（用今日收盘价成交）
            # 仅在全部清仓后触发：避免持仓期间反复用旧信号建仓，导致持仓周期混乱
            if not positions:
                if pending_buy is not None:
                    signals, timing_ratio = pending_buy
                    invest = cash * timing_ratio
                    cash -= self._execute_buys(today, invest, signals, positions, price_index, trade_log, picks_log)
                    # 建仓成功后：下一次信号在 holding_period-1 天后生成（卖出前一天）
                    if positions:
                        next_signal_idx = i + self.cfg.holding_period - 1
                # 无论本日是否买入，都清空待执行信号，防止下一轮持仓结束后用过期信号建仓
                pending_buy = None

            # ⑤ 生成明日信号（仅在必要时执行，减少无效计算）
            # 触发条件：空仓中（随时准备入场）或已到下次预定信号日
            if not positions or i >= next_signal_idx:
                if self.strategy.fit(today):
                    signals = self.strategy.generate_signals(today)
                    if signals is not None and not signals.empty:
                        # 将 pick 日市值附到 signals 上，供 _execute_buys 写入 picks_log
                        if 'total_mv' in df_today.columns:
                            mv_map = price_index['total_mv']
                            signals = signals.copy()
                            signals['_pick_mv'] = signals[self.STOCK_COL].map(mv_map)
                        # 同时记录今日择时比例，确保信号和仓位判断来自同一时间点
                        t_ratio = self.timing.get_position_ratio(today) if self.timing else 1.0
                        if t_ratio > 0:
                            pending_buy = (signals, t_ratio)

            # ⑥ 更新今日 NAV（交易后，含新建仓位的成本）
            daily_nav[-1] = (today, cash + self._holdings_value(positions))

        # ---- 回测结束，强制平仓所有剩余持仓（用最后一日收盘价结算）----
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
        self._dump_results(start_date, end_date, nav_df, trade_log, picks_log)
        return nav_df, trade_log

    def _dump_results(
        self, start_date: str, end_date: str,
        nav_df: pd.DataFrame, trade_log: list, picks_log: list
    ) -> None:
        """
        将回测结果持久化到本地目录。

        输出目录优先使用 config.result_dir；若为 None 则自动生成带时间戳的子目录；
        若为空字符串 "" 则跳过所有文件输出。

        生成文件：
          nav.csv          — 每日净值 (date, nav)
          trade_log.csv    — 完整成交记录 (date, side, ts_code, shares, price, amount)
          daily_picks.csv  — 买入时的每股打分 (date, ts_code, score, buy_price, shares)
        """
        result_dir = self.cfg.result_dir
        if result_dir == '':
            return

        if result_dir is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_dir = os.path.join('backtest', 'results',
                                      f'{start_date}_{end_date}_{ts}')

        os.makedirs(result_dir, exist_ok=True)

        nav_df.to_csv(os.path.join(result_dir, 'nav.csv'), index=False)

        pd.DataFrame(
            trade_log,
            columns=['date', 'side', 'ts_code', 'shares', 'price', 'amount']
        ).to_csv(os.path.join(result_dir, 'trade_log.csv'), index=False)

        pd.DataFrame(
            picks_log,
            columns=['date', 'ts_code', 'score', 'buy_price', 'shares', 'total_mv', 'pct_change']
        ).to_csv(os.path.join(result_dir, 'daily_picks.csv'), index=False)

        print(f'回测结果已保存 → {result_dir}')

    # ------------------------------------------------------------------ #
    #  绩效报告                                                             #
    # ------------------------------------------------------------------ #

    def report(self, nav_df: pd.DataFrame, capital: float = None) -> None:
        """
        根据每日净值序列打印绩效摘要。

        指标说明：
            总收益率   = (期末NAV / 初始资金) - 1
            年化收益率 = 以 252 个交易日为基准的复利年化
            年化夏普   = (日均收益率 / 日收益率标准差) * sqrt(252)，无风险利率近似为 0
            最大回撤   = NAV 从历史峰值的最大跌幅
            日胜率     = 日收益率 > 0 的交易日占比

        Parameters
        ----------
        nav_df  : DataFrame[date, nav]  由 run() 返回的净值序列
        capital : float                 初始资金，None 时使用 config.initial_capital
        """
        if capital is None:
            capital = self.cfg.initial_capital
        nav = nav_df.copy()
        nav['ret'] = nav['nav'].pct_change()   # 日收益率序列
        std_ret = nav['ret'].std()
        # 年化夏普：假设无风险利率为 0，252 为年交易日数
        sharpe = (nav['ret'].mean() / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0
        total_return = nav['nav'].iloc[-1] / capital - 1
        # 逐日计算相对历史峰值的回撤幅度
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
