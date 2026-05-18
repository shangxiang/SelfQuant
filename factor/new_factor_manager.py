import numpy as np
import pandas as pd
import glob
import os


# ================== 1. 财务因子预处理 ==================

def build_financial_features(fin_df: pd.DataFrame) -> dict:
    """
    对全量财务数据做一次性预处理，计算衍生因子，返回按股票代码索引的字典。

    财务数据以「报告期（end_date）」为单位存储，但对日线数据对齐时需要用
    「实际披露日（f_ann_date）」，避免使用未来信息（point-in-time 原则）。
    后续每只股票日线数据通过 merge_asof 向前匹配最近一次披露的财务因子。

    Parameters
    ----------
    fin_df : pd.DataFrame
        合并后的财务宽表，需含 ts_code、end_date、f_ann_date 及各财务指标列

    Returns
    -------
    dict  {ts_code: DataFrame}
        每只股票对应一个以 f_ann_date 为索引、含 6 个因子列的 DataFrame
    """
    fin = fin_df.copy()
    fin['end_date']   = pd.to_datetime(fin['end_date'],   format='%Y%m%d')
    fin['f_ann_date'] = pd.to_datetime(fin['f_ann_date'], format='%Y%m%d')

    fin_features = {}

    for code, group in fin.groupby('ts_code'):
        g = group.sort_values('end_date').copy()
        if g.empty:
            continue

        # 同一报告期可能因更正重新披露，保留最新的一条，确保 end_date 索引唯一
        g = g.drop_duplicates(subset='end_date', keep='last')

        # 用于同比计算：将 end_date 向前偏移一年，映射到去年同期的财务数据
        g['end_yoy'] = g['end_date'] - pd.DateOffset(years=1)
        rev_map = g.set_index('end_date')['total_revenue']
        np_map  = g.set_index('end_date')['n_income']
        g['rev_prev'] = g['end_yoy'].map(rev_map)   # 去年同期营收
        g['np_prev']  = g['end_yoy'].map(np_map)    # 去年同期净利润

        # ---- ① 毛利率 = (营收 - 营业成本) / 营收 ----
        # 衡量企业产品的盈利能力，排除三费等期间费用的影响
        rev  = g['total_revenue']
        cost = g['oper_cost']
        g['gross_margin'] = ((rev - cost) / rev.replace(0, np.nan)).fillna(0)

        # ---- ② 资产负债率 = 总负债 / 总资产 ----
        # 衡量财务杠杆水平，过高说明偿债压力大
        assets      = g['total_assets']
        liabilities = g['total_liab']
        g['debt_ratio'] = (liabilities / assets.replace(0, np.nan)).fillna(0)

        # ---- ③ ROE TTM（滚动4个季度） ----
        # 用最近4个报告期的净利润之和 / 平均股东权益，避免季节性波动
        g['np_sum4'] = g['n_income_attr_p'].rolling(window=4, min_periods=1).sum()
        g['eq_avg4'] = g['total_hldr_eqy_exc_min_int'].rolling(window=4, min_periods=1).mean()
        g['roe_ttm'] = (g['np_sum4'] / g['eq_avg4'].replace(0, np.nan)).fillna(0)

        # ---- ④ 营收同比增长率 ----
        # 以去年同期为基准，规避季节性因素；基期为 0 时输出 0.0 防止除零
        rev_cur  = g['total_revenue']
        rev_prev = g['rev_prev'].fillna(0)
        g['revenue_growth_yoy'] = np.where(
            rev_prev != 0,
            (rev_cur - rev_prev) / rev_prev.abs(),
            0.0
        )

        # ---- ⑤ 净利润同比增长率 ----
        np_cur  = g['n_income']
        np_prev = g['np_prev'].fillna(0)
        g['profit_growth_yoy'] = np.where(
            np_prev != 0,
            (np_cur - np_prev) / np_prev.abs(),
            0.0
        )

        # ---- ⑥ 应计项目（Accruals） ----
        # 用于衡量盈余质量：NOA 变动 / 平均资产，正值偏大说明利润含水量高
        # NOA = (总资产 - 货币资金) - (总负债 - 短期借款)
        # 即剔除金融性资产/负债后的经营性净资产
        def calc_noa(row):
            return (row['total_assets'] - row['money_cap']) - (
                    row['total_liab']   - row['st_borr'])

        g['noa']         = g.apply(calc_noa, axis=1)
        g['noa_prev']    = g['noa'].shift(1)             # 上期 NOA
        g['assets_prev'] = g['total_assets'].shift(1)    # 上期总资产
        avg_assets = (g['total_assets'] + g['assets_prev']) / 2
        g['accruals'] = np.where(
            avg_assets != 0,
            (g['noa'] - g['noa_prev']) / avg_assets,
            0.0
        )
        g['accruals'] = g['accruals'].fillna(0)

        # ---- ⑦ 资产增长率（CMA 因子代理）----
        # 总资产同比增速：当期 vs 去年同期，增速越低说明扩张越保守（正 CMA 暴露）
        assets_map = g.set_index('end_date')['total_assets']
        g['assets_prev_yoy'] = g['end_yoy'].map(assets_map)
        g['asset_growth_yoy'] = np.where(
            g['assets_prev_yoy'].notna() & (g['assets_prev_yoy'] != 0),
            (g['total_assets'] - g['assets_prev_yoy']) / g['assets_prev_yoy'].abs(),
            0.0
        )
        g['asset_growth_yoy'] = g['asset_growth_yoy'].fillna(0)

        # 只保留需要的列，写回前再做一次去重
        keep_cols = ['f_ann_date', 'end_date',
                     'gross_margin', 'debt_ratio', 'roe_ttm',
                     'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
                     'asset_growth_yoy']
        g = g[keep_cols].drop_duplicates()

        # 同一披露日（f_ann_date）保留报告期最新的那条（应对同日多份报表场景）
        g = g.sort_values(['f_ann_date', 'end_date'])
        g = g.drop_duplicates('f_ann_date', keep='last')
        g = g.set_index('f_ann_date').sort_index()

        factor_cols = ['gross_margin', 'debt_ratio', 'roe_ttm',
                       'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
                       'asset_growth_yoy']
        fin_features[code] = g[factor_cols].copy()

    return fin_features


# ================== 2. 因子管理器 ==================

class FactorManager:
    """
    单只股票的因子计算器。

    接收一个股票的日线 CSV 路径，计算全量技术因子和基本面因子后原地写回。
    所有因子方法都直接修改 self.df，由 update_factor() 统一调用。
    """

    def __init__(self, path: str, fin_features: dict):
        """
        Parameters
        ----------
        path         : str   股票日线 CSV 的完整路径（含文件名）
        fin_features : dict  由 build_financial_features() 返回的财务因子字典
        """
        self.path = path
        self.df = pd.read_csv(path)
        self.fin_features = fin_features

        # trade_date 存储为 YYYYMMDD 整数，计算时转为 datetime 便于时间运算
        self.df['trade_date'] = pd.to_datetime(
            self.df['trade_date'], format='%Y%m%d'
        )

        # 股票代码：优先从数据列读取；若无该列（如文件名即代码），则从文件名提取
        if 'ts_code' in self.df.columns:
            self.code = self.df['ts_code'].iloc[0]
        else:
            self.code = os.path.splitext(os.path.basename(self.path))[0]

    # ------------------------------------------------------------------ #
    #  基本面因子对齐                                                       #
    # ------------------------------------------------------------------ #

    def _add_financial_factors(self) -> None:
        """
        将本股票的基本面因子一次性对齐到日线，使用 merge_asof 向前匹配。

        merge_asof（direction='backward'）：对每个 trade_date，
        找到最近一个 f_ann_date <= trade_date 的财务数据行拼接过来，
        保证回测时只用到已经公开披露的财务数据，不引入未来信息。
        若无历史财务数据，对应列填 0。
        """
        feat = self.fin_features.get(self.code)
        factor_cols = ['gross_margin', 'debt_ratio', 'roe_ttm',
                       'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
                       'asset_growth_yoy']

        if feat is None or feat.empty:
            # 无财报数据的股票（如新上市未出报告期）因子全部置 0
            for col in factor_cols:
                self.df[col] = 0.0
            return

        df_sorted   = self.df.sort_values('trade_date')
        # 删除左表中已存在的因子列，防止重复运行时 merge_asof 产生 _x/_y 后缀
        df_sorted = df_sorted.drop(columns=[c for c in factor_cols if c in df_sorted.columns])
        feat_sorted = feat.reset_index().sort_values('f_ann_date')

        # 以 trade_date 为左键、f_ann_date 为右键做 asof 合并
        merged = pd.merge_asof(
            df_sorted, feat_sorted,
            left_on='trade_date', right_on='f_ann_date',
            direction='backward'   # 取 <= trade_date 的最新披露
        )

        for col in factor_cols:
            # 合并结果的行顺序与 df_sorted 一致，直接赋值给原 df
            self.df[col] = merged[col].fillna(0).values

    # ------------------------------------------------------------------ #
    #  技术因子                                                             #
    # ------------------------------------------------------------------ #

    def label(self, period: int = 5):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。"""
        df = self.df
        df['label'] = (df['close_x'].shift(-period-1) - df['close_x'].shift(-1)) / df['close_x']
        return df[['label']]

    def label_10(self, period: int = 10):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。"""
        df = self.df
        df['label_10'] = (df['close_x'].shift(-period-1) - df['close_x'].shift(-1)) / df['close_x']
        return df[['label_10']]

    def label_25(self, period: int = 25):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。"""
        df = self.df
        df['label_25'] = (df['close_x'].shift(-period-1) - df['close_x'].shift(-1)) / df['close_x']
        return df[['label_25']]

    def macd(self, price_col='close_x', fast=12, slow=26, signal=9):
        """
        MACD 三线：DIF（快慢均线差）、DEA（DIF 的信号线）、MACD（柱状值）。
        标准参数 12-26-9，使用指数移动平均（EMA）。
        """
        df = self.df
        df['ema_fast'] = df[price_col].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df[price_col].ewm(span=slow, adjust=False).mean()
        df['dif']  = df['ema_fast'] - df['ema_slow']
        df['dea']  = df['dif'].ewm(span=signal, adjust=False).mean()
        df['macd'] = 2 * (df['dif'] - df['dea'])   # 柱状值 = 2 * (DIF - DEA)
        df.drop(columns=['ema_fast', 'ema_slow'], inplace=True)
        return df[['dif', 'dea', 'macd']]

    def kdj(self, high_col='high', low_col='low', close_col='close_x',
            period=9, k_period=3, d_period=3):
        """
        KDJ 随机指标：K、D 为平滑后的超买超卖指标，J 为 K 和 D 的偏离度。
        RSV = (收盘 - N日最低) / (N日最高 - N日最低) * 100
        K = (K_prev * (period-1) + RSV) / period（加权移动平均）
        最高最低价相等时（一字板等）RSV 设为 50，避免除零。
        """
        df = self.df
        df['lowest_low']   = df[low_col].rolling(window=period, min_periods=1).min()
        df['highest_high'] = df[high_col].rolling(window=period, min_periods=1).max()
        denom = df['highest_high'] - df['lowest_low']
        rsv = pd.Series(index=df.index, dtype=float)
        mask = denom == 0
        rsv[~mask] = 100 * (df.loc[~mask, close_col] - df.loc[~mask, 'lowest_low']) / denom[~mask]
        rsv[mask]  = 50.0  # 最高最低价相等（如一字涨停/跌停），RSV 取中性值
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
        df['J'] = 3 * df['K'] - 2 * df['D']   # J 超出 [0,100] 表示极度超买/超卖
        df.drop(['lowest_low', 'highest_high'], axis=1, inplace=True)
        return df[['K', 'D', 'J']]

    def rsi(self, price_col='close_x', period=14):
        """
        RSI 相对强弱指数：衡量一段时间内涨幅占总波动的比例。
        RSI = 100 - 100 / (1 + 平均涨幅 / 平均跌幅)
        区间 [0, 100]，>70 超买，<30 超卖。avg_loss 为 0 时 rs 置 0（全部上涨）。
        """
        df = self.df
        delta    = df[price_col].diff()
        gain     = delta.clip(lower=0)         # 正收益，跌日为 0
        loss     = -delta.clip(upper=0)        # 绝对跌幅，涨日为 0
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.fillna(0)
        rs = rs.fillna(0)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)   # 数据不足时填中性值 50
        return df[['rsi']]

    def cci(self, high_col='high', low_col='low', close_col='close_x', period=14):
        """
        CCI 顺势指标：典型价格偏离其均值的程度（以平均绝对偏差为分母归一化）。
        典型价格 TP = (高 + 低 + 收) / 3；超过 ±100 通常视为超买/超卖。
        """
        df = self.df
        tp     = (df[high_col] + df[low_col] + df[close_col]) / 3
        tp_sma = tp.rolling(window=period, min_periods=period).mean()
        mad    = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (tp - tp_sma) / (0.015 * mad).replace(0, np.nan)
        df['cci'] = cci.fillna(0)
        return df[['cci']]

    def force_index(self, volume_col='vol', price_col='close_x', window=1):
        """
        强度指数：价格变动 × 成交量，衡量价格变动背后的资金驱动力。
        window=1 时为原始值；window>1 时做 EMA 平滑以减少噪音。
        """
        df = self.df
        df['raw_force_index']       = df[volume_col] * df[price_col].diff()
        df['force_index_smoothed']  = df['raw_force_index'].ewm(span=window, adjust=False).mean()
        df['force_index']           = df['force_index_smoothed'].fillna(0)
        return df[['force_index']]

    def vwap(self):
        """
        成交量加权均价（VWAP）及收盘价相对偏离度。
        VWAP = 成交额 / 成交量（Tushare 成交额单位为元，成交量单位为手=100股，
        需 ×10 换算到每股均价）。
        close_to_vwap_ratio > 0 表示收盘价高于当日均价（买方强势）。
        """
        df = self.df
        df['vwap']               = (df['amount_x'] / df['vol']) * 10
        df['close_to_vwap_ratio'] = (df['close_x'] - df['vwap']) / df['vwap']
        return df[['vwap', 'close_to_vwap_ratio']]

    def mfi(self, high_col='high', low_col='low', close_col='close_x',
            volume_col='vol', period=14):
        """
        资金流量指标（MFI）：将 RSI 的价格换为「典型价格 × 成交量」，
        衡量资金净流入强度。>80 超买，<20 超卖。
        """
        df = self.df
        typical_price   = (df[high_col] + df[low_col] + df[close_col]) / 3
        raw_money_flow  = typical_price * df[volume_col]
        price_diff      = df[close_col].diff()
        # 当日收盘价 >= 昨日：资金为正向流入，否则为负向流出
        df['positive_flow'] = np.where(price_diff >= 0, raw_money_flow, 0)
        df['negative_flow'] = np.where(price_diff <  0, raw_money_flow, 0)
        sum_pos = df['positive_flow'].rolling(window=period, min_periods=period).sum()
        sum_neg = df['negative_flow'].rolling(window=period, min_periods=period).sum()
        money_flow_ratio = sum_pos / sum_neg.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        df['mfi'] = mfi.fillna(50)
        return df[['mfi']]

    def mtm_margin(self, margin_balance_col='rzye', period=5):
        """
        融资余额动量：period 日前后融资余额的变化率，反映杠杆资金进出趋势。
        正值：融资加仓；负值：融资减仓（可能预示下行压力）。
        """
        df = self.df
        df['mtm_margin_balance_change'] = df[margin_balance_col].pct_change(periods=period, fill_method=None)
        df['mtm_margin_balance_change'] = df['mtm_margin_balance_change'].fillna(0)
        return df[['mtm_margin_balance_change']]

    # def macd_air_refuel(self, dif_col='dif', dea_col='dea', macd_col='macd'):
    #     """
    #     MACD 金叉后的「空中加油」形态：DIF 在 DEA 上方（多头排列），
    #     前一日 MACD 柱缩量或回落（短暂回踩），今日 MACD 柱放量上行。
    #     该形态视为趋势中继的做多信号。
    #     """
    #     df = self.df
    #     golden_state        = df[dif_col] > df[dea_col]          # DIF 在 DEA 上方
    #     macd_rising         = df[macd_col] > df[macd_col].shift(1)  # 今日 MACD 柱放大
    #     yesterday_macd      = df[macd_col].shift(1)
    #     yesterday_macd_slope = df[macd_col].diff().shift(1)
    #     # 前一日满足：MACD 柱 <= 0 或斜率为负（曾经回踩）
    #     was_pullback = (yesterday_macd <= 0) | (yesterday_macd_slope < 0)
    #     air_refuel = golden_state & macd_rising & was_pullback
    #     df['macd_air_refuel'] = air_refuel.astype(int).fillna(0)
    #     return df[['macd_air_refuel']]

    def macd_air_refuel(self, dif_col='dif', dea_col='dea', macd_col='macd'):
        """
        基于 MACD 柱局部谷底 + DIF 逐级抬高的做多信号。
        条件：昨天 MACD 柱是局部最低点，且该谷底的 DIF 高于上一个谷底（底背离加速反转）。
        """
        df = self.df
        # 局部谷底：昨天 MACD 柱 < 前天 且 < 今天
        is_trough = (df[macd_col].shift(1) < df[macd_col].shift(2)) & \
                    (df[macd_col].shift(1) < df[macd_col])
        df['_trough'] = is_trough.astype(int)
        df['_dif_at_trough'] = df[dif_col].where(df['_trough'] == 1)
        # 上一个谷底处的 DIF（向前填充后再 shift 取"前一个"）
        df['_prev_dif'] = df['_dif_at_trough'].ffill().shift(1)
        # 信号：当前是谷底 且 DIF 比上一个谷底高（逐级抬高）
        final = (df['_trough'] == 1) & (df[dif_col] > df['_prev_dif'])
        df['macd_air_refuel'] = final.astype(int).fillna(0)
        df.drop(['_trough', '_dif_at_trough', '_prev_dif'], axis=1, inplace=True)
        return df[['macd_air_refuel']]

    def macd_divergence(self, window=5):
        """
        MACD 底背离：股价创 window 日内新低，但 MACD 柱未创新低，
        暗示下跌动能减弱，可能出现反转，输出为 0/1 二值信号。
        """
        df = self.df
        low_min  = df['low'].rolling(window).min()
        macd_min = df['macd'].rolling(window).min()
        low_new_low  = (df['low']  < low_min.shift(1))   # 价格创新低
        macd_new_low = (df['macd'] < macd_min.shift(1))  # MACD 也创新低（非背离）
        df['macd_divergence'] = (low_new_low & ~macd_new_low).astype(int).fillna(0)
        return df[['macd_divergence']]

    def big_order_ratio(self):
        """
        大单净比率：(主动买入 - 主动卖出) / 总成交额，衡量大资金的净买卖方向。
        依赖 mfi() 已计算的 positive_flow / negative_flow 列。
        """
        df = self.df
        net_big = df['positive_flow'] - df['negative_flow']
        df['big_order_ratio'] = net_big / df['amount_x']
        df['big_order_ratio'] = df['big_order_ratio'].fillna(0)
        return df[['big_order_ratio']]

    def lhb_strength_5d(self):
        """
        龙虎榜净买入强度：5日龙虎榜净买入额 / 流通市值，
        衡量机构或游资在该股上的相对持续买入力度。
        """
        df = self.df
        df['lhb_strength_5d'] = df['net_amount'].rolling(5).sum() / df['circ_mv']
        df['lhb_strength_5d'] = df['lhb_strength_5d'].fillna(0)
        return df[['lhb_strength_5d']]

    def vol_breakout(self, vol_col='vol', base_window=20, multiplier=1.5):
        """
        成交量放量突破信号：今日成交量 > N 日均量 × multiplier 时输出 1，否则 0。
        用于识别异常放量，通常伴随重要价格突破。
        """
        df = self.df
        avg_vol = df[vol_col].rolling(base_window).mean()
        df['vol_breakout'] = (df[vol_col] > avg_vol * multiplier).astype(int).fillna(0)
        return df[['vol_breakout']]

    def volatility_20d(self, close_col='close_x', window=20):
        """
        20 日历史波动率（年化）：日收益率的滚动标准差 × √252。
        用于衡量个股风险水平，可作为风险控制因子使用。
        """
        df = self.df
        ret = df[close_col].pct_change()
        vol = ret.rolling(window).std() * np.sqrt(252)
        df['volatility_20d'] = vol.fillna(0)
        return df[['volatility_20d']]

    def reversal_5d(self, close_col='close_x', period=5):
        """
        5 日反转因子：过去 5 日累计涨幅取反。
        短期内涨幅越大，反转因子越小（预期均值回归向下），反之亦然。
        """
        df = self.df
        chg = df[close_col].pct_change(periods=period)
        df['reversal_5d'] = -chg   # 取负号：近期跌得多 → 因子值大 → 预期反弹
        df['reversal_5d'] = df['reversal_5d'].fillna(0)
        return df[['reversal_5d']]

    def high_low_spread(self, high_col='high', low_col='low', close_col='close_x'):
        """
        日内振幅：(最高价 - 最低价) / 收盘价，衡量日内波动幅度。
        振幅大通常意味着分歧加剧或流动性下降。
        """
        df = self.df
        df['high_low_spread'] = (df[high_col] - df[low_col]) / df[close_col]
        df['high_low_spread'] = df['high_low_spread'].fillna(0)
        return df[['high_low_spread']]

    def size_factor(self, mv_col='circ_mv'):
        """
        SMB 规模因子代理：流通市值取对数后取反，值越大说明规模越小。
        小市值股票在 Fama-French 框架中具有正 SMB 暴露，历史上存在小市值溢价。
        circ_mv 单位为万元，取 log 后量纲一致、分布更对称。
        """
        df = self.df
        df['size_factor'] = -np.log(df[mv_col].replace(0, np.nan)).fillna(0)
        return df[['size_factor']]

    def value_factor(self, pb_col='pb'):
        """
        HML 估值因子代理：账面市值比 = 1 / PB，值越大说明越低估（价值股）。
        高账面市值比对应正 HML 暴露，即 Fama-French "价值溢价"的来源。
        PB 为 0 或缺失时填 0，避免除零。
        """
        df = self.df
        df['value_factor'] = (1 / df[pb_col].replace(0, np.nan)).fillna(0)
        return df[['value_factor']]

    def cma_factor(self):
        """
        CMA 投资因子代理：资产增长率取反（依赖 _add_financial_factors 已对齐）。
        资产扩张越保守（增速低）= 正 CMA 暴露；激进扩张企业 CMA 暴露为负。
        """
        df = self.df
        df['cma_factor'] = -df['asset_growth_yoy']
        return df[['cma_factor']]

    def momentum_12_1(self, close_col='close_x', long_window=252, short_window=21):
        """
        动量因子（12-1 月）：过去 12 个月累计涨幅，排除最近 1 个月以规避短期反转。
        计算方式：ret_mom = (1 + ret_252) / (1 + ret_21) - 1
        即用 t-252 到 t-21 区间的净累计收益衡量中长期动量强度。
        """
        df = self.df
        ret_long  = df[close_col].pct_change(periods=long_window)
        ret_short = df[close_col].pct_change(periods=short_window)
        df['momentum_12_1'] = ((1 + ret_long) / (1 + ret_short.fillna(0)) - 1).fillna(0)
        return df[['momentum_12_1']]

    # ------------------------------------------------------------------ #
    #  高阶 / 交叉因子（必须在基础 FF 因子写入 df 之后计算）                 #
    # ------------------------------------------------------------------ #

    def smb_squared(self):
        """
        SMB² 规模非线性因子：捕捉规模效应的曲率。
        极小盘（流动性风险）和极大盘（机构抱团）可能同时跑输中盘，呈 U 形关系。
        """
        df = self.df
        df['smb_squared'] = df['size_factor'] ** 2
        return df[['smb_squared']]

    def smb_mom(self):
        """
        SMB × Mom：小盘动量交叉项。
        小市值股票套利成本高、机构回避，动量信号不易被纠正，持续性更强。
        """
        df = self.df
        df['smb_mom'] = df['size_factor'] * df['momentum_12_1']
        return df[['smb_mom']]

    def smb_squared_mom(self):
        """
        SMB² × Mom：规模非线性与动量的三阶交叉。
        捕捉极小盘动量效应相对中盘/大盘的额外溢价（非线性叠加）。
        """
        df = self.df
        df['smb_squared_mom'] = df['size_factor'] ** 2 * df['momentum_12_1']
        return df[['smb_squared_mom']]

    def hml_rmw(self):
        """
        HML × RMW："质量价值"交叉因子。
        高账面市值比（低估）叠加高盈利能力（ROE TTM），双重筛选排除"价值陷阱"。
        roe_ttm 已由 _add_financial_factors 对齐到日线，无需重新计算。
        """
        df = self.df
        df['hml_rmw'] = df['value_factor'] * df['roe_ttm']
        return df[['hml_rmw']]

    def smb_hml(self):
        """
        SMB × HML：小市值价值股交叉项。
        同时具备规模溢价和价值溢价的双重特征，历史上超额收益显著。
        """
        df = self.df
        df['smb_hml'] = df['size_factor'] * df['value_factor']
        return df[['smb_hml']]

    def vol_mom(self):
        """
        波动率 × 动量交叉项：捕捉动量崩溃的条件风险。
        高波动 + 强动量的股票在市场急速反转时暴跌更猛（Daniel & Moskowitz 2016）。
        volatility_20d 和 momentum_12_1 均须先于本方法执行。
        """
        df = self.df
        df['vol_mom'] = df['volatility_20d'] * df['momentum_12_1']
        return df[['vol_mom']]

    # ------------------------------------------------------------------ #
    #  统一计算入口                                                         #
    # ------------------------------------------------------------------ #

    def update_factor(self, factor_list: list = None) -> bool:
        """
        计算全量因子并将结果写回原 CSV 文件。

        执行顺序：
          1. 基本面因子对齐（merge_asof）
          2. 技术量价因子逐个计算（各方法直接修改 self.df）
          3. 将 trade_date 恢复为 YYYYMMDD 整数格式后写盘

        Parameters
        ----------
        factor_list : list[str] | None
            指定要计算的因子方法名列表，None 表示计算全部因子

        Returns
        -------
        bool  始终返回 True（便于调用方做批量计数）
        """
        # 1. 基本面因子：通过 merge_asof 按披露日对齐到日线
        self._add_financial_factors()

        # 2. 技术量价因子
        if factor_list is None:
            calcu_list = [
                self.label, self.label_10, self.label_25,
                self.macd, self.kdj, self.mfi, self.rsi,
                self.cci, self.force_index, self.vwap, self.mtm_margin,
                self.macd_air_refuel, self.macd_divergence, self.big_order_ratio,
                self.lhb_strength_5d, self.vol_breakout, self.volatility_20d,
                self.reversal_5d, self.high_low_spread,
                # Fama-French 风格因子（size/value 读日线列，cma 读已对齐的财务列）
                self.size_factor, self.value_factor, self.cma_factor, self.momentum_12_1,
                # 高阶/交叉因子：依赖上方 FF 因子列已写入 df，必须置于其后
                self.smb_squared, self.smb_mom, self.smb_squared_mom,
                self.hml_rmw, self.smb_hml, self.vol_mom,
            ]
        else:
            calcu_list = []
            for f in factor_list:
                if hasattr(self, f):
                    calcu_list.append(getattr(self, f))
                else:
                    print(f"警告：{f} 不是有效的因子方法名")

        for calc in calcu_list:
            calc()

        # 写盘前将 trade_date 从 datetime 恢复为原始整数格式（YYYYMMDD）
        self.df['trade_date'] = self.df['trade_date'].dt.strftime('%Y%m%d').astype(int)
        self.df.to_csv(self.path, index=False)
        return True


# ================== 3. 主程序 ==================

if __name__ == '__main__':
    # 路径相对于本文件所在目录的上层（项目根），无论从哪里执行都正确
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    series_path = os.path.join(_base, "data", "series") + os.sep
    fin_df = pd.read_csv(os.path.join(_base, "data", "financial.csv"))

    print("预处理财务数据...")
    fin_features = build_financial_features(fin_df)
    print(f"完成，共 {len(fin_features)} 只股票。")

    # 只匹配股票代码文件（6位数字.交易所.csv），排除 daily_basic_data.csv 等合并产物
    files = glob.glob(series_path + "[0-9]*.csv")
    total = len(files)
    for idx, file in enumerate(files, 1):
        print(f"\r处理 {idx}/{total}: {file}", end='', flush=True)
        fm = FactorManager(file, fin_features)
        fm.update_factor()
    print("\n全部完成！")
