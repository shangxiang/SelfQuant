import os
import numpy as np
import pandas as pd
from datetime import datetime

from strategy.selection.base_strategy import BaseStrategy
from strategy.sell.base_sell_strategy import BaseSellStrategy
from strategy.sell.hold_n_days import HoldNDaysSellStrategy
from backtest.timing import BaseTimingStrategy
from typing import Optional


def resolve_date_range(all_dates: list, start_date: str = None, end_date: str = None) -> tuple:
    """
    把用户传入的起止日期对齐到实际存在的交易日上。

    对齐规则（只内缩、不外扩，避免用到区间外的数据）：
        起始日 → >= start_date 的第一个交易日
        结束日 → <= end_date   的最后一个交易日

    这样传入节假日/周末（如 20260925 中秋休市）不会再抛 ValueError。

    Parameters
    ----------
    all_dates  : list[str]  升序的交易日列表
    start_date : str        YYYYMMDD，None 表示取第一个交易日
    end_date   : str        YYYYMMDD，None 表示取最后一个交易日

    Returns
    -------
    (start_eff, end_eff) : tuple[str, str]  对齐后的起止交易日
    """
    if not all_dates:
        raise ValueError('交易日历为空：请先运行 run_signal.py 生成 data/market/ 分区数据')

    start_date = start_date or all_dates[0]
    end_date = end_date or all_dates[-1]

    after = [d for d in all_dates if d >= start_date]
    if not after:
        raise ValueError(f'开始日 {start_date} 晚于本地最后一个交易日 {all_dates[-1]}')
    start_eff = after[0]

    before = [d for d in all_dates if d <= end_date]
    if not before:
        raise ValueError(f'结束日 {end_date} 早于本地第一个交易日 {all_dates[0]}')
    end_eff = before[-1]

    if start_eff > end_eff:
        raise ValueError(
            f'区间 {start_date} ~ {end_date} 内没有可用交易日'
            f'（邻近交易日：{start_eff} / {end_eff}）'
        )
    return start_eff, end_eff


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
    # 注意：close_x 是不复权的真实交易价格，用于回测中的买卖执行
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
        self.log_file = open('tmp.csv', 'w')
        # 交易成本台账：佣金 / 印花税 / 滑点 / 成交额，report() 里汇总展示
        self.costs = {'commission': 0.0, 'stamp_duty': 0.0, 'slippage': 0.0, 'turnover': 0.0}

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
            if stock not in price_index.index:
                # 今日该股票无行情（停牌等），沿用买入价估值
                pos['current_price'] = pos['buy_price']
                continue
            px = price_index.loc[stock, self.PRICE_COL]
            # 停牌日的价格可能是 NaN，直接赋进去会把 NAV 变成 NaN
            pos['current_price'] = pos['buy_price'] if pd.isna(px) else px

    def _holdings_value(self, positions: dict) -> float:
        """计算当前持仓总市值（按 current_price 估值）。"""
        return sum(pos['shares'] * pos['current_price'] for pos in positions.values())

    # ------------------------------------------------------------------ #
    #  卖出执行                                                             #
    # ------------------------------------------------------------------ #

    def _execute_sells(
        self, positions: dict, today: str, all_dates: list, trade_log: list,
        price_index: pd.DataFrame = None
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

        price_index : pd.DataFrame   当日截面（用于判断跌停），可为 None

        Returns
        -------
        float  本次卖出所得现金总额（已扣佣金、印花税与滑点）
        """
        # 将 dict 转为 list[dict] 传给卖出策略（策略接口约定）
        pos_list = [{'ts_code': s, **pos} for s, pos in positions.items()]
        keep_ratios = self.sell_strategy.evaluate(pos_list, today, all_dates)

        pct_col = (next((c for c in ('pct_chg', 'pct_change') if c in price_index.columns), None)
                   if price_index is not None else None)

        cash_from_sells = 0.0
        for stock, ratio in keep_ratios.items():
            if stock not in positions or ratio >= 1.0:
                continue
            pos = positions[stock]
            if pos.get('current_price') is None or pd.isna(pos['current_price']):
                continue

            # 跌停无法卖出（一字跌停完全没有对手盘）
            if self.cfg.enable_limit_check and pct_col is not None and stock in price_index.index:
                pct = price_index.loc[stock, pct_col]
                if pd.notna(pct) and pct <= self.cfg.limit_down_pct:
                    continue

            # ratio=0.0 时全卖，否则按 (1-ratio) 比例卖出
            sell_shares = pos['shares'] if ratio == 0.0 else int(pos['shares'] * (1 - ratio))
            # print("stock:", stock)
            # print("sell_shares", sell_shares)
            if sell_shares <= 0:
                continue

            ref_price = pos['current_price']                       # 收盘价（估值用）
            sell_price = ref_price * (1 - self.cfg.slippage)       # 实际成交价（含滑点）
            proceeds = sell_shares * sell_price * (1 - self.cfg.commission - self.cfg.stamp_duty)
            cash_from_sells += proceeds
            trade_log.append((today, 'SELL', stock, sell_shares, sell_price, proceeds))

            self.costs['commission'] += sell_shares * sell_price * self.cfg.commission
            self.costs['stamp_duty'] += sell_shares * sell_price * self.cfg.stamp_duty
            self.costs['slippage']   += sell_shares * (ref_price - sell_price)
            self.costs['turnover']   += sell_shares * sell_price

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
    ) -> tuple:
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
        (cash_spent, alias)  实际花费现金总额（含佣金）及打分偏离度
        """
        # 取分数最高的 top_n 只股票
        top = signals.head(min(self.cfg.top_n, len(signals)))
        n = len(top)
        if n == 0:
            return 0.0, float('nan')

        score_std = signals['score'].std()
        score_mean = signals['score'].mean()
        nth = min(self.cfg.top_n, len(signals)) - 1
        score_num_n = signals['score'].iloc[nth]
        alias = (score_num_n - score_mean) / score_std if score_std != 0 else float('nan')
        print("alias:", alias)
        self.log_file.write(f"{alias},")

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
            # 买入当天涨幅（T+1 日，即实际成交日）
            pct_chg = price_index.loc[stock, pct_col] if pct_col else float('nan')
            # 涨停无法买入
            if self.cfg.enable_limit_check and pd.notna(pct_chg) and pct_chg >= self.cfg.limit_up_pct:
                continue

            ref_price = price_index.loc[stock, self.PRICE_COL]     # 收盘价
            if pd.isna(ref_price):
                # 停牌或数据缺失，无法成交
                continue
            buy_price = ref_price * (1 + self.cfg.slippage)        # 实际成交价（含滑点）
            # 按 100 股/手取整，A 股最小交易单位为 1 手（100 股）
            shares = int(per_stock / buy_price / 100) * 100
            if shares <= 0:
                continue
            cost = shares * buy_price * (1 + self.cfg.commission)
            self.costs['commission'] += shares * buy_price * self.cfg.commission
            self.costs['slippage']   += shares * (buy_price - ref_price)
            self.costs['turnover']   += shares * buy_price
            cash_spent += cost
            positions[stock] = {
                'buy_date': today,
                'buy_price': buy_price,
                'shares': shares,
                'current_price': buy_price,
            }
            trade_log.append((today, 'BUY', stock, shares, buy_price, cost))
            picks_log.append((today, stock, score, buy_price, shares, pick_mv, pct_chg))

        return cash_spent, alias

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

        # 起止日可能落在非交易日（周末/节假日，如 20260925 中秋休市），
        # 先对齐到区间内实际存在的交易日，再切片
        req_start, req_end = start_date, end_date
        start_date, end_date = resolve_date_range(all_dates, start_date, end_date)
        if (start_date, end_date) != (req_start, req_end):
            print(f'  ⚠ 回测起止日已对齐到最近交易日：{req_start} → {start_date}，'
                  f'{req_end} → {end_date}（非交易日不参与回测）')

        trade_dates = all_dates[all_dates.index(start_date): all_dates.index(end_date) + 1]

        if self.timing is not None:
            # 预计算全区间内的择时状态（如均线序列），避免在每日循环内重复读文件
            self.timing.prepare(start_date, end_date)
        self.strategy.reset()
        self.costs = {'commission': 0.0, 'stamp_duty': 0.0, 'slippage': 0.0, 'turnover': 0.0}

        cash = capital                          # 当前账户现金
        positions: dict = {}                    # {ts_code: {buy_date, buy_price, shares, current_price}}
        daily_nav: list = []                    # [(date, nav), ...]，最终转为 DataFrame
        trade_log: list = []                    # 完整成交记录
        picks_log: list = []                    # 选股打分记录：(date, ts_code, score, buy_price, shares)
        batch_log: list = []                    # 每批买入记录：(buy_date, alias)
        last_batch_cost: float = 0.0            # 上批买入总成本，用于计算已实现收益率
        batch_sell_proceeds: float = 0.0        # 本批累计卖出金额（跨多日，止损+到期合计）
        pending_buy: Optional[tuple] = None     # T 日生成的信号，在 T+1 日成交

        # 下一次需要生成信号的 trade_dates 下标
        # 初始为 0（从第一天起尝试建仓）；买入成功后推进到下一个周期节点
        # 提前清仓后不立即重入，等到预定的 next_signal_idx 才重新生成信号
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
            cash_from_sells = 0
            if positions:
                cash_from_sells = self._execute_sells(positions, today, all_dates,
                                                      trade_log, price_index)
                cash += cash_from_sells
                if cash_from_sells > 0:
                    batch_sell_proceeds += cash_from_sells   # 累积本批所有卖出（含止损中途卖出）
                    print('今日收盘价卖出，today = ', today)
                    if not positions and i < next_signal_idx:
                        print('提前清仓了')
                    # 全部清仓后通知择时策略本批已实现收益率（用累计卖出，而非仅当日卖出）
                    if not positions and self.timing is not None and last_batch_cost > 0:
                        batch_profit = (batch_sell_proceeds - last_batch_cost) / last_batch_cost
                        self.timing.on_batch_sold(batch_profit)

            # ④ 执行前一日信号的买入（用今日收盘价成交）
            # 仅在全部清仓后触发：避免持仓期间反复用旧信号建仓，导致持仓周期混乱
            if not positions:
                if pending_buy is not None:
                    print("今天是", today)
                    # 仓位比例在买入时查询（而非信号生成时），确保能用最新的择时判断
                    t_ratio = self.timing.get_position_ratio(today) if self.timing else 1.0
                    if t_ratio > 0:
                        # 模型强度过滤：top-1 score 低于阈值时跳过建仓
                        top1_score = pending_buy['score'].iloc[0] if not pending_buy.empty else 0.0
                        min_thr = self.cfg.min_score_threshold or 0.0
                        if top1_score < min_thr:
                            print(f"  模型强度不足（top1 score={top1_score:.6f} < {min_thr}），跳过建仓")
                            next_signal_idx = i
                        else:
                            invest = cash * t_ratio
                            cash_spent, alias = self._execute_buys(today, invest, pending_buy, positions, price_index, trade_log, picks_log)
                            cash -= cash_spent
                            last_batch_cost = cash_spent
                            batch_sell_proceeds = 0.0            # 新批次开始，重置累计卖出金额
                            batch_log.append((today, alias))
                            if positions:
                                next_signal_idx = i + self.cfg.holding_period - 1
                    else:
                        # 择时空仓：当天步骤⑤仍需生成信号（next_signal_idx = i），明天再判断择时
                        next_signal_idx = i
                # 无论本日是否买入，都清空待执行信号，防止下一轮持仓结束后用过期信号建仓
                pending_buy = None

            # ⑤ 生成明日信号（固定周期触发）
            # 只在预定的信号日运行，提前清仓后也等到下一个周期节点再入场
            if i >= next_signal_idx:
                if self.strategy.fit(today):
                    signals = self.strategy.generate_signals(today)
                    if signals is not None and not signals.empty:
                        # 将 pick 日市值附到 signals 上，供 _execute_buys 写入 picks_log
                        if 'total_mv' in df_today.columns:
                            mv_map = price_index['total_mv']
                            signals = signals.copy()
                            signals['_pick_mv'] = signals[self.STOCK_COL].map(mv_map)
                        # 仓位比例延迟到买入时查询，此处只存信号
                        pending_buy = signals

            # ⑥ 更新今日 NAV（交易后，含新建仓位的成本）
            daily_nav[-1] = (today, cash + self._holdings_value(positions))
            print(daily_nav[-1])


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
                    proceeds = pos['shares'] * price * (1 - self.cfg.commission)
                    cash += proceeds
                    trade_log.append((last_date, 'SELL', stock, pos['shares'], price, proceeds))
                positions.clear()
                if daily_nav:
                    daily_nav[-1] = (last_date, cash)

        nav_df = pd.DataFrame(daily_nav, columns=['date', 'nav'])
        self._dump_results(start_date, end_date, nav_df, trade_log, picks_log, batch_log)
        return nav_df, trade_log

    def _dump_results(
        self, start_date: str, end_date: str,
        nav_df: pd.DataFrame, trade_log: list, picks_log: list, batch_log: list
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

        if batch_log:
            batch_df = self._compute_batch_profits(trade_log, batch_log)
            batch_df.to_csv(os.path.join(result_dir, 'batch_alias_profit.csv'), index=False)

        print(f'回测结果已保存 → {result_dir}')

    # ------------------------------------------------------------------ #
    #  批次收益计算                                                          #
    # ------------------------------------------------------------------ #

    def _compute_batch_profits(self, trade_log: list, batch_log: list) -> pd.DataFrame:
        """
        计算每批买入对应的实现收益率，与 alias 对齐输出。

        匹配逻辑：每批 BUY 发生在 buy_date，对应 SELLs 在下一批 BUY 之前（含强制平仓）。
        trade_profit = (sell_proceeds_sum - buy_cost_sum) / buy_cost_sum
        """
        tl = pd.DataFrame(trade_log, columns=['date', 'side', 'ts_code', 'shares', 'price', 'amount'])
        buy_dates = [d for d, _ in batch_log]
        rows = []
        for i, (buy_date, alias) in enumerate(batch_log):
            total_cost = tl.loc[(tl['date'] == buy_date) & (tl['side'] == 'BUY'), 'amount'].sum()
            next_buy = buy_dates[i + 1] if i + 1 < len(buy_dates) else None
            sell_mask = (tl['side'] == 'SELL') & (tl['date'] > buy_date)
            if next_buy:
                sell_mask &= (tl['date'] <= next_buy)
            total_proceeds = tl.loc[sell_mask, 'amount'].sum()
            trade_profit = (total_proceeds - total_cost) / total_cost if total_cost > 0 else float('nan')
            rows.append((buy_date, alias, trade_profit))
        return pd.DataFrame(rows, columns=['trade_date', 'alias', 'trade_profit'])

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
        nav = nav_df.copy().reset_index(drop=True)

        # 跳过前置空仓预热期（净值持续不变的阶段）
        # 预热期每日 pct_change = 0，会稀释胜率、夏普、年化收益率
        daily_ret_all = nav['nav'].pct_change().fillna(0)
        first_active  = int((daily_ret_all.abs() > 1e-9).idxmax())
        nav_active    = nav.iloc[first_active:].copy().reset_index(drop=True)

        nav_active['ret'] = nav_active['nav'].pct_change()
        ret = nav_active['ret'].dropna()

        std_ret = ret.std()
        sharpe  = (ret.mean() / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0
        total_return = nav['nav'].iloc[-1] / capital - 1
        ann_return   = (nav['nav'].iloc[-1] / capital) ** (252 / len(nav_active)) - 1
        nav_active['drawdown'] = (
            (nav_active['nav'] - nav_active['nav'].cummax()) / nav_active['nav'].cummax()
        )
        win_rate = (ret > 0).mean()

        print("========== 回测报告 ==========")
        print(f"初始资金:     {capital:>15,.0f}")
        print(f"最终资金:     {nav['nav'].iloc[-1]:>15,.0f}")
        print(f"总收益率:     {total_return:>14.2%}")
        print(f"年化收益率:   {ann_return:>14.2%}")
        print(f"年化夏普:     {sharpe:>14.4f}")
        print(f"最大回撤:     {nav_active['drawdown'].min():>14.2%}")
        print(f"日胜率:       {win_rate:>14.2%}")
        print(f"（统计区间: {nav_active['date'].iloc[0]} ~ {nav_active['date'].iloc[-1]}，"
              f"共 {len(nav_active)} 个交易日，预热期已排除）")

        # ---- 交易成本明细 ----
        total_cost = (self.costs['commission'] + self.costs['stamp_duty']
                      + self.costs['slippage'])
        print("-" * 30)
        print(f"佣金累计:     {self.costs['commission']:>15,.0f}")
        print(f"印花税累计:   {self.costs['stamp_duty']:>15,.0f}")
        print(f"滑点成本:     {self.costs['slippage']:>15,.0f}")
        print(f"成本合计:     {total_cost:>15,.0f}  （占初始资金 {total_cost / capital:.2%}，"
              f"约占交易额 {total_cost / max(self.costs['turnover'], 1e-9):.3%}）")
