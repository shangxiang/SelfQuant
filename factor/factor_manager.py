import numpy as np
import pandas as pd
import glob

class FactorManager:
    """
    给个时序df，直接计算出来对应的factor
    """
    def __init__(self, path, fin_df):
        self.path = path
        self.df = pd.read_csv(path)
        self.fin = fin_df.copy()

        # 预处理日期类型
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        self.fin['end_date'] = pd.to_datetime(self.fin['end_date'])
        self.fin['f_ann_date'] = pd.to_datetime(self.fin['f_ann_date'])

        # 预先按股票和发布日期排序，方便后续高效查找
        self.fin = self.fin.sort_values(['ts_code', 'f_ann_date', 'end_date'])

        # 预计算每个交易日对应的最新财报快照
        self._build_snapshot_map()

    def _build_snapshot_map(self):
        """
        对每个交易日，构建 'ts_code -> 最新财报行' 的 DataFrame，
        利用 merge_asof 按 f_ann_date 匹配，确保无未来信息。
        """
        # 得到所有唯一交易日
        dates = self.df['trade_date'].drop_duplicates().sort_values()
        snapshots = {}
        for td in dates:
            # 所有发布日期 <= td 的财报
            available = self.fin[self.fin['f_ann_date'] <= td].copy()
            if available.empty:
                snapshots[td] = pd.DataFrame()
                continue
            # 对每只股票，保留 f_ann_date 最大的一行
            # 使用 sort + drop_duplicates 保留最后一条
            latest = available.sort_values(['ts_code', 'f_ann_date', 'end_date']) \
                .drop_duplicates('ts_code', keep='last')
            snapshots[td] = latest.set_index('ts_code')
        self.snapshots = snapshots

    def _get_latest_row(self, td, code):
        """获取 (td, code) 的最新财报行（Series），无数据则返回 None"""
        snap = self.snapshots.get(td)
        if snap is None or snap.empty:
            return None
        try:
            return snap.loc[code]
        except KeyError:
            return None

    def gross_margin(self, revenue_col='total_revenue', cost_col='oper_cost') -> pd.DataFrame:
        res = []
        for _, row in self.df.iterrows():
            td = row['trade_date']
            code = row['ts_code']
            fin_row = self._get_latest_row(td, code)
            if fin_row is None:
                res.append(0.0)
                continue
            rev, cost = fin_row.get(revenue_col), fin_row.get(cost_col)
            if pd.isna(rev) or pd.isna(cost) or rev == 0:
                res.append(0.0)
            else:
                res.append((rev - cost) / rev)
        return pd.DataFrame({'gross_margin': res}, index=self.df.index)

    def debt_ratio(self, total_assets_col='total_assets', total_liabilities_col='total_liabilities') -> pd.DataFrame:
        res = []
        for _, row in self.df.iterrows():
            td, code = row['trade_date'], row['ts_code']
            fin_row = self._get_latest_row(td, code)
            if fin_row is None:
                res.append(0.0)
                continue
            assets = fin_row.get(total_assets_col)
            liabilities = fin_row.get(total_liabilities_col)
            if pd.isna(assets) or pd.isna(liabilities) or assets == 0:
                res.append(0.0)
            else:
                res.append(liabilities / assets)
        return pd.DataFrame({'debt_ratio': res}, index=self.df.index)

    def roe_ttm(self, net_profit_col='n_income_attr_p', equity_col='total_hldr_eqy_exc_min_int') -> pd.DataFrame:
        # 需要最近4个季度，可用 fin_df 在原日期范围内筛选
        res = []
        for _, row in self.df.iterrows():
            td, code = row['trade_date'], row['ts_code']
            # 取出该股票所有已发布财报
            code_fin = self.fin[(self.fin['ts_code'] == code) & (self.fin['f_ann_date'] <= td)]
            if code_fin.empty:
                res.append(0.0)
                continue
            code_fin = code_fin.sort_values('end_date', ascending=False)
            last4 = code_fin.head(4)
            if len(last4) < 4:
                res.append(0.0)
                continue
            np_sum = last4[net_profit_col].sum()
            eq_avg = last4[equity_col].mean()
            if pd.isna(np_sum) or pd.isna(eq_avg) or eq_avg == 0:
                res.append(0.0)
            else:
                res.append(np_sum / eq_avg)
        return pd.DataFrame({'roe_ttm': res}, index=self.df.index)

    def revenue_growth_yoy(self, revenue_col='total_revenue') -> pd.DataFrame:
        res = []
        for _, row in self.df.iterrows():
            td, code = row['trade_date'], row['ts_code']
            code_fin = self.fin[(self.fin['ts_code'] == code) & (self.fin['f_ann_date'] <= td)]
            if code_fin.empty:
                res.append(0.0)
                continue
            code_fin = code_fin.sort_values('end_date', ascending=False)
            latest = code_fin.iloc[0]
            target_end = latest['end_date'] - pd.DateOffset(years=1)
            same_quarter = code_fin[code_fin['end_date'] == target_end]
            if same_quarter.empty:
                res.append(0.0)
                continue
            rev_current = latest[revenue_col]
            rev_prev = same_quarter[revenue_col].iloc[0]
            if pd.isna(rev_current) or pd.isna(rev_prev) or rev_prev == 0:
                res.append(0.0)
            else:
                res.append((rev_current - rev_prev) / abs(rev_prev))
        return pd.DataFrame({'revenue_growth_yoy': res}, index=self.df.index)

    def profit_growth_yoy(self, profit_col='net_profit') -> pd.DataFrame:
        res = []
        for _, row in self.df.iterrows():
            td, code = row['trade_date'], row['ts_code']
            code_fin = self.fin[(self.fin['ts_code'] == code) & (self.fin['f_ann_date'] <= td)]
            if code_fin.empty:
                res.append(0.0)
                continue
            code_fin = code_fin.sort_values('end_date', ascending=False)
            latest = code_fin.iloc[0]
            target_end = latest['end_date'] - pd.DateOffset(years=1)
            same_quarter = code_fin[code_fin['end_date'] == target_end]
            if same_quarter.empty:
                res.append(0.0)
                continue
            p_current = latest[profit_col]
            p_prev = same_quarter[profit_col].iloc[0]
            if pd.isna(p_current) or pd.isna(p_prev) or p_prev == 0:
                res.append(0.0)
            else:
                res.append((p_current - p_prev) / abs(p_prev))
        return pd.DataFrame({'profit_growth_yoy': res}, index=self.df.index)

    def accruals(self,
                 total_assets_col='total_assets',
                 total_liabilities_col='total_liab',
                 cash_col='money_cap',
                 short_debt_col='st_borr') -> pd.DataFrame:
        def calc_noa(row):
            return (row[total_assets_col] - row[cash_col]) - (
                    row[total_liabilities_col] - row[short_debt_col])

        res = []
        for _, row in self.df.iterrows():
            td, code = row['trade_date'], row['ts_code']
            code_fin = self.fin[(self.fin['ts_code'] == code) & (self.fin['f_ann_date'] <= td)]
            if code_fin.empty:
                res.append(0.0)
                continue
            code_fin = code_fin.sort_values('end_date', ascending=False)
            if len(code_fin) < 2:
                res.append(0.0)
                continue
            cur, prev = code_fin.iloc[0], code_fin.iloc[1]
            noa_cur = calc_noa(cur)
            noa_prev = calc_noa(prev)
            avg_assets = (cur[total_assets_col] + prev[total_assets_col]) / 2
            if pd.isna(noa_cur) or pd.isna(noa_prev) or avg_assets == 0:
                res.append(0.0)
            else:
                res.append((noa_cur - noa_prev) / avg_assets)
        return pd.DataFrame({'accruals': res}, index=self.df.index)

    def label(self, period=5):
        """
        未来函数，标准答案作为label
        :return:
        """
        df = self.df
        df["label"] = (df["close_x"].shift(-period) - df["close_x"]) / df["close_x"]
        return df[["label"]]

    def macd(self, price_col='close_x', fast=12, slow=26, signal=9):
        df = self.df
        df['ema_fast'] = df[price_col].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df[price_col].ewm(span=slow, adjust=False).mean()
        df['dif'] = df['ema_fast'] - df['ema_slow']
        df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])
        df.drop(columns=['ema_fast', 'ema_slow'], inplace=True)
        return df[['dif', 'dea', 'macd']]

    def kdj(self, high_col='high', low_col='low', close_col='close_x',
                      period=9, k_period=3, d_period=3):
        df = self.df

        # 滚动窗口计算
        df['lowest_low'] = df[low_col].rolling(window=period, min_periods=1).min()
        df['highest_high'] = df[high_col].rolling(window=period, min_periods=1).max()

        # 计算 RSV，处理分母为零
        denom = df['highest_high'] - df['lowest_low']
        rsv = pd.Series(index=df.index, dtype=float)
        # 避免除零
        mask = denom == 0
        rsv[~mask] = 100 * (df.loc[~mask, close_col] - df.loc[~mask, 'lowest_low']) / denom[~mask]
        rsv[mask] = 50.0  # 最高=最低 时 RSV 取 50

        # 初始化 K, D
        ks = [50.0] * len(df)
        ds = [50.0] * len(df)

        for i in range(len(df)):
            if i == 0:
                ks[i] = rsv.iloc[i]
                ds[i] = rsv.iloc[i]
            else:
                ks[i] = (ks[i - 1] * (k_period - 1) + rsv.iloc[i]) / k_period
                ds[i] = (ds[i - 1] * (d_period - 1) + ks[i]) / d_period

        df['K'] = ks
        df['D'] = ds
        df['J'] = 3 * df['K'] - 2 * df['D']

        df.drop(['lowest_low', 'highest_high'], axis=1, inplace=True)
        return df[['K', 'D', 'J']]

    def rsi(self, price_col: str = 'close_x', period: int = 14) -> pd.DataFrame:
        """
        计算相对强弱指数 (Relative Strength Index)
        """
        df = self.df
        # 1. 计算价格变动
        delta = df[price_col].diff()
        # 2. 分离上涨和下跌，并计算平均涨幅和跌幅
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # 3. 使用移动平均进行计算 (与 Wilder's 方法不同，此处使用简单移动/指数平均均可)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # 4. 计算RS值并处理分母为0的情况
        rs = avg_gain / avg_loss
        rs = rs.fillna(0)  # 防止除零，若 avg_loss 为 0，则 RS 为正无穷，RSI 为 100
        # 5. 计算RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        # 对于所有价格没有波动的时期，RSI 设为 50
        df['rsi'] = df['rsi'].fillna(50)
        return df[['rsi']]

    def cci(self, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close_x',
                      period: int = 14) -> pd.DataFrame:
        """
        计算商品通道指数 (Commodity Channel Index)
        """
        df = self.df
        # 1. 计算典型价格 (Typical Price)
        tp = (df[high_col] + df[low_col] + df[close_col]) / 3
        # 2. 计算 TP 的简单移动平均
        tp_sma = tp.rolling(window=period, min_periods=period).mean()
        # 3. 计算平均绝对偏差 (Mean Absolute Deviation)
        mad = tp.rolling(window=period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean())
        # 4. 计算 CCI，处理分母为0
        cci = (tp - tp_sma) / (0.015 * mad).replace(0, np.nan)
        df['cci'] = cci.fillna(0)
        return df[['cci']]

    def force_index(self, volume_col: str = 'vol', price_col: str = 'close_x',
                              window: int = 1) -> pd.DataFrame:
        """
        计算强力指数 (Force Index)，并可进行平滑处理
        """
        df = self.df
        # 1. 计算原始强力指数
        df['raw_force_index'] = df[volume_col] * df[price_col].diff()
        # 2. 可选：使用 EMA 进行平滑
        df['force_index_smoothed'] = df['raw_force_index'].ewm(span=window, adjust=False).mean()
        # 3. 返回光滑后的值，不足行数用0填充
        df['force_index'] = df['force_index_smoothed'].fillna(0)
        return df[['force_index']]

    def vwap(self) -> pd.DataFrame:
        """
        计算当日VWAP指标，并计算偏离度
        """
        df = self.df
        df['vwap'] = (df['amount_x'] / df['vol']) * 10
        df['close_to_vwap_ratio'] = (df['close_x'] - df['vwap']) / df['vwap']
        return df[['vwap', 'close_to_vwap_ratio']]

    def mfi(self, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close_x',
                      volume_col: str = 'vol', period: int = 14) -> pd.DataFrame:
        """
        计算资金流量指标 (Money Flow Index)
        """
        df = self.df
        # 1. 计算典型价格
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        # 2. 计算原始资金流量
        raw_money_flow = typical_price * df[volume_col]
        # 3. 确定资金流向的正负
        price_diff = df[close_col].diff()
        df['positive_flow'] = np.where(price_diff >= 0, raw_money_flow, 0)
        df['negative_flow'] = np.where(price_diff < 0, raw_money_flow, 0)
        # 4. 计算周期内正负资金的累积和
        sum_positive_flow = df['positive_flow'].rolling(window=period, min_periods=period).sum()
        sum_negative_flow = df['negative_flow'].rolling(window=period, min_periods=period).sum()
        # 5. 计算资金比率和MFI
        money_flow_ratio = sum_positive_flow / sum_negative_flow.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        df['mfi'] = mfi.fillna(50)  # 无量不足或平衡时期设为50
        return df[['mfi']]

    def mtm_margin(self, margin_balance_col: str = 'rzye', period: int = 5) -> pd.DataFrame:
        """
        计算融资融券余额的 N 日动量
        """
        df = self.df
        # 计算 N 日融资余额变化率
        df['mtm_margin_balance_change'] = df[margin_balance_col].pct_change(periods=period)
        df['mtm_margin_balance_change'] = df['mtm_margin_balance_change'].fillna(0)
        return df[['mtm_margin_balance_change']]

    def macd_air_refuel(self,
                        dif_col: str = 'dif',
                        dea_col: str = 'dea',
                        macd_col: str = 'macd') -> pd.DataFrame:
        """
        MACD空中加油（放宽版）：
        条件：
        1. 处于金叉状态（DIF > DEA）或当日刚金叉
        2. MACD柱今日 > 昨日（柱线向上），说明回调结束重新发力
        3. 不强制要求DIF在零轴上方，允许从零轴下启动
        """
        df = self.df

        # 条件1：DIF在DEA上方（金叉状态延续）
        golden_state = df[dif_col] > df[dea_col]

        # 条件2：MACD柱向上（今日 > 昨日），代表绿柱缩短或红柱变长
        macd_rising = df[macd_col] > df[macd_col].shift(1)

        # 条件3（可选，增强信号纯度）：昨日的MACD柱 ≤ 0 或 昨日柱线在缩短（即回调发生）
        # 这样能过滤掉连续红柱延长的情况，只捕捉回调后的二次启动
        yesterday_macd = df[macd_col].shift(1)
        yesterday_macd_slope = df[macd_col].diff().shift(1)
        was_pullback = (yesterday_macd <= 0) | (yesterday_macd_slope < 0)  # 昨日处于回调

        # 最终信号：金叉状态 + 柱线向上 + 此前有回调
        air_refuel = golden_state & macd_rising & was_pullback

        df['macd_air_refuel'] = air_refuel.astype(int)
        df['macd_air_refuel'] = df['macd_air_refuel'].fillna(0)
        return df[['macd_air_refuel']]

    def macd_divergence(self, window: int = 5) -> pd.DataFrame:
        """底背离：股价新低，但MACD未新低"""
        df = self.df
        low_min = df['low'].rolling(window).min()
        macd_min = df['macd'].rolling(window).min()
        low_new_low = (df['low'] < low_min.shift(1))
        macd_new_low = (df['macd'] < macd_min.shift(1))
        df['macd_divergence'] = (low_new_low & ~macd_new_low).astype(int)
        df['macd_divergence'] = df['macd_divergence'].fillna(0)
        return df[['macd_divergence']]

    def big_order_ratio(self) -> pd.DataFrame:
        """大单净买入占成交额比例（使用正负资金流近似）"""
        df = self.df
        net_big = df['positive_flow'] - df['negative_flow']
        df['big_order_ratio'] = net_big / df['amount_x']
        df['big_order_ratio'] = df['big_order_ratio'].fillna(0)
        return df[['big_order_ratio']]

    def lhb_strength_5d(self) -> pd.DataFrame:
        """龙虎榜近5日净买入强度（相对流通市值）"""
        df = self.df
        # 假设 net_amount 为龙虎榜净买入额，若无则可用 l_buy - l_sell
        df['lhb_strength_5d'] = df['net_amount'].rolling(5).sum() / df['circ_mv']
        df['lhb_strength_5d'] = df['lhb_strength_5d'].fillna(0)
        return df[['lhb_strength_5d']]

    def vol_breakout(self, vol_col: str = 'vol', base_window: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
        """成交量突破：当日成交量 > base_window日均量的multiplier倍"""
        df = self.df
        avg_vol = df[vol_col].rolling(base_window).mean()
        df['vol_breakout'] = (df[vol_col] > avg_vol * multiplier).astype(int)
        df['vol_breakout'] = df['vol_breakout'].fillna(0)
        return df[['vol_breakout']]

    def volatility_20d(self, close_col: str = 'close_x', window: int = 20) -> pd.DataFrame:
        """20日年化波动率"""
        df = self.df
        ret = df[close_col].pct_change()
        vol = ret.rolling(window).std() * np.sqrt(252)
        df['volatility_20d'] = vol
        df['volatility_20d'] = df['volatility_20d'].fillna(0)
        return df[['volatility_20d']]

    def reversal_5d(self, close_col: str = 'close_x', period: int = 5) -> pd.DataFrame:
        """5日反转：过去5日收益取负（反转效应）"""
        df = self.df
        chg = df[close_col].pct_change(periods=period)
        df['reversal_5d'] = -chg
        df['reversal_5d'] = df['reversal_5d'].fillna(0)
        return df[['reversal_5d']]

    def high_low_spread(self, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close_x') -> pd.DataFrame:
        """日内振幅 = (最高-最低)/收盘价"""
        df = self.df
        df['high_low_spread'] = (df[high_col] - df[low_col]) / df[close_col]
        df['high_low_spread'] = df['high_low_spread'].fillna(0)
        return df[['high_low_spread']]

    def update_factor(self, factor_list: list = None):
        """
        批量更新 factor 到线下数据中
        :param factor_list: 可选，因子名字符串列表
        :return: 处理后的 DataFrame
        """
        if factor_list is None:
            calcu_list = [
                self.label, self.macd, self.kdj, self.mfi, self.rsi,
                self.cci, self.force_index, self.vwap, self.mtm_margin,
                self.macd_air_refuel, self.macd_divergence, self.big_order_ratio,
                self.lhb_strength_5d, self.vol_breakout, self.volatility_20d,
                self.reversal_5d, self.high_low_spread, self.gross_margin,
                self.debt_ratio, self.roe_ttm, self.revenue_growth_yoy, self.profit_growth_yoy,
                self.accruals
            ]
        else:
            calcu_list = []
            for factor in factor_list:
                if hasattr(self, factor):
                    calcu_list.append(getattr(self, factor))
                else:
                    print(f"警告：{factor} 不是有效属性")  # 或者跳过/抛异常

        for calcu in calcu_list:
            calcu()
        self.df.to_csv(self.path, index=False)
        return True


if __name__ == '__main__':

    series_path = "../data/series/"
    fin_df = pd.read_csv("../data/financial.csv")
    for file in glob.glob(series_path + "*.csv"):
        factor_manager = FactorManager(file, fin_df)
        result = factor_manager.update_factor()
        print(result, file)