import numpy as np
import pandas as pd
import glob
import os

# ================== 1. 全局预处理 financial.csv ==================
def build_financial_features(fin_df):
    fin = fin_df.copy()
    fin['end_date'] = pd.to_datetime(fin['end_date'], format='%Y%m%d')
    fin['f_ann_date'] = pd.to_datetime(fin['f_ann_date'], format='%Y%m%d')

    fin_features = {}

    for code, group in fin.groupby('ts_code'):
        g = group.sort_values('end_date').copy()
        if g.empty:
            continue

        # ★ 修复1：同一 end_date 只保留最新披露的一条（去重），确保索引唯一
        g = g.drop_duplicates(subset='end_date', keep='last')

        # ★ 去年同期匹配
        g['end_yoy'] = g['end_date'] - pd.DateOffset(years=1)
        rev_map = g.set_index('end_date')['total_revenue']
        np_map  = g.set_index('end_date')['net_profit']
        g['rev_prev'] = g['end_yoy'].map(rev_map)
        g['np_prev']  = g['end_yoy'].map(np_map)

        # ---- ① 毛利率 ----
        rev = g['total_revenue']
        cost = g['oper_cost']
        g['gross_margin'] = ((rev - cost) / rev.replace(0, np.nan)).fillna(0)

        # ---- ② 负债率 ----
        assets = g['total_assets']
        liabilities = g['total_liab']   # 保持与 debt_ratio 默认参数一致
        g['debt_ratio'] = (liabilities / assets.replace(0, np.nan)).fillna(0)

        # ---- ③ ROE TTM（最近4个报告期） ----
        g['np_sum4'] = g['n_income_attr_p'].rolling(window=4, min_periods=1).sum()
        g['eq_avg4'] = g['total_hldr_eqy_exc_min_int'].rolling(window=4, min_periods=1).mean()
        g['roe_ttm'] = ((g['np_sum4'] / g['eq_avg4'].replace(0, np.nan)).fillna(0))

        # ---- ④ 营收同比增长 ----
        rev_cur = g['total_revenue']
        rev_prev = g['rev_prev'].fillna(0)
        g['revenue_growth_yoy'] = np.where(
            rev_prev != 0,
            (rev_cur - rev_prev) / rev_prev.abs(),
            0.0
        )

        # ---- ⑤ 净利润同比增长 ----
        np_cur = g['net_profit']
        np_prev = g['np_prev'].fillna(0)
        g['profit_growth_yoy'] = np.where(
            np_prev != 0,
            (np_cur - np_prev) / np_prev.abs(),
            0.0
        )

        # ---- ⑥ 应计项目 ----
        def calc_noa(row):
            return (row['total_assets'] - row['money_cap']) - (
                    row['total_liab'] - row['st_borr'])
        g['noa'] = g.apply(calc_noa, axis=1)
        g['noa_prev'] = g['noa'].shift(1)
        g['assets_prev'] = g['total_assets'].shift(1)
        avg_assets = (g['total_assets'] + g['assets_prev']) / 2
        g['accruals'] = np.where(
            avg_assets != 0,
            (g['noa'] - g['noa_prev']) / avg_assets,
            0.0
        )
        g['accruals'] = g['accruals'].fillna(0)

        # 保留因子列
        keep_cols = ['f_ann_date', 'end_date',
                     'gross_margin', 'debt_ratio', 'roe_ttm',
                     'revenue_growth_yoy', 'profit_growth_yoy', 'accruals']
        g = g[keep_cols].drop_duplicates()

        # ★ 同一 f_ann_date 只保留报告期最新的那条（应对同日多份报表的场景）
        g = g.sort_values(['f_ann_date', 'end_date'])
        g = g.drop_duplicates('f_ann_date', keep='last')
        g = g.set_index('f_ann_date').sort_index()

        factor_cols = ['gross_margin', 'debt_ratio', 'roe_ttm',
                       'revenue_growth_yoy', 'profit_growth_yoy', 'accruals']
        fin_features[code] = g[factor_cols].copy()

    return fin_features


# ================== 2. 优化后的 FactorManager ==================
class FactorManager:
    def __init__(self, path, fin_features):
        self.path = path
        self.df = pd.read_csv(path)
        self.fin_features = fin_features

        # ---- 正确解析 trade_date，格式 YYYYMMDD ----
        self.df['trade_date'] = pd.to_datetime(
            self.df['trade_date'], format='%Y%m%d'
        )

        # 股票代码：优先从列获取，否则由文件名提取
        if 'ts_code' in self.df.columns:
            self.code = self.df['ts_code'].iloc[0]
        else:
            self.code = os.path.splitext(os.path.basename(self.path))[0]

    def _add_financial_factors(self):
        """将本股票的基本面因子一次性对齐到日线"""
        feat = self.fin_features.get(self.code)
        factor_cols = ['gross_margin', 'debt_ratio', 'roe_ttm',
                       'revenue_growth_yoy', 'profit_growth_yoy', 'accruals']
        if feat is None or feat.empty:
            for col in factor_cols:
                self.df[col] = 0.0
            return

        df_sorted = self.df.sort_values('trade_date')
        feat_sorted = feat.reset_index().sort_values('f_ann_date')

        merged = pd.merge_asof(
            df_sorted, feat_sorted,
            left_on='trade_date', right_on='f_ann_date',
            direction='backward'
        )

        for col in factor_cols:
            self.df[col] = merged[col].fillna(0).values

    # -------- 技术因子（与原代码完全一致） --------
    def label(self, period=5):
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
        df['lowest_low'] = df[low_col].rolling(window=period, min_periods=1).min()
        df['highest_high'] = df[high_col].rolling(window=period, min_periods=1).max()
        denom = df['highest_high'] - df['lowest_low']
        rsv = pd.Series(index=df.index, dtype=float)
        mask = denom == 0
        rsv[~mask] = 100 * (df.loc[~mask, close_col] - df.loc[~mask, 'lowest_low']) / denom[~mask]
        rsv[mask] = 50.0
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

    def rsi(self, price_col='close_x', period=14):
        df = self.df
        delta = df[price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.fillna(0)
        rs = rs.fillna(0)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df[['rsi']]

    def cci(self, high_col='high', low_col='low', close_col='close_x', period=14):
        df = self.df
        tp = (df[high_col] + df[low_col] + df[close_col]) / 3
        tp_sma = tp.rolling(window=period, min_periods=period).mean()
        mad = tp.rolling(window=period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - tp_sma) / (0.015 * mad).replace(0, np.nan)
        df['cci'] = cci.fillna(0)
        return df[['cci']]

    def force_index(self, volume_col='vol', price_col='close_x', window=1):
        df = self.df
        df['raw_force_index'] = df[volume_col] * df[price_col].diff()
        df['force_index_smoothed'] = df['raw_force_index'].ewm(span=window, adjust=False).mean()
        df['force_index'] = df['force_index_smoothed'].fillna(0)
        return df[['force_index']]

    def vwap(self):
        df = self.df
        df['vwap'] = (df['amount_x'] / df['vol']) * 10
        df['close_to_vwap_ratio'] = (df['close_x'] - df['vwap']) / df['vwap']
        return df[['vwap', 'close_to_vwap_ratio']]

    def mfi(self, high_col='high', low_col='low', close_col='close_x',
            volume_col='vol', period=14):
        df = self.df
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        raw_money_flow = typical_price * df[volume_col]
        price_diff = df[close_col].diff()
        df['positive_flow'] = np.where(price_diff >= 0, raw_money_flow, 0)
        df['negative_flow'] = np.where(price_diff < 0, raw_money_flow, 0)
        sum_positive_flow = df['positive_flow'].rolling(window=period, min_periods=period).sum()
        sum_negative_flow = df['negative_flow'].rolling(window=period, min_periods=period).sum()
        money_flow_ratio = sum_positive_flow / sum_negative_flow.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        df['mfi'] = mfi.fillna(50)
        return df[['mfi']]

    def mtm_margin(self, margin_balance_col='rzye', period=5):
        df = self.df
        df['mtm_margin_balance_change'] = df[margin_balance_col].pct_change(periods=period)
        df['mtm_margin_balance_change'] = df['mtm_margin_balance_change'].fillna(0)
        return df[['mtm_margin_balance_change']]

    def macd_air_refuel(self, dif_col='dif', dea_col='dea', macd_col='macd'):
        df = self.df
        golden_state = df[dif_col] > df[dea_col]
        macd_rising = df[macd_col] > df[macd_col].shift(1)
        yesterday_macd = df[macd_col].shift(1)
        yesterday_macd_slope = df[macd_col].diff().shift(1)
        was_pullback = (yesterday_macd <= 0) | (yesterday_macd_slope < 0)
        air_refuel = golden_state & macd_rising & was_pullback
        df['macd_air_refuel'] = air_refuel.astype(int)
        df['macd_air_refuel'] = df['macd_air_refuel'].fillna(0)
        return df[['macd_air_refuel']]

    def macd_divergence(self, window=5):
        df = self.df
        low_min = df['low'].rolling(window).min()
        macd_min = df['macd'].rolling(window).min()
        low_new_low = (df['low'] < low_min.shift(1))
        macd_new_low = (df['macd'] < macd_min.shift(1))
        df['macd_divergence'] = (low_new_low & ~macd_new_low).astype(int)
        df['macd_divergence'] = df['macd_divergence'].fillna(0)
        return df[['macd_divergence']]

    def big_order_ratio(self):
        df = self.df
        net_big = df['positive_flow'] - df['negative_flow']
        df['big_order_ratio'] = net_big / df['amount_x']
        df['big_order_ratio'] = df['big_order_ratio'].fillna(0)
        return df[['big_order_ratio']]

    def lhb_strength_5d(self):
        df = self.df
        df['lhb_strength_5d'] = df['net_amount'].rolling(5).sum() / df['circ_mv']
        df['lhb_strength_5d'] = df['lhb_strength_5d'].fillna(0)
        return df[['lhb_strength_5d']]

    def vol_breakout(self, vol_col='vol', base_window=20, multiplier=1.5):
        df = self.df
        avg_vol = df[vol_col].rolling(base_window).mean()
        df['vol_breakout'] = (df[vol_col] > avg_vol * multiplier).astype(int)
        df['vol_breakout'] = df['vol_breakout'].fillna(0)
        return df[['vol_breakout']]

    def volatility_20d(self, close_col='close_x', window=20):
        df = self.df
        ret = df[close_col].pct_change()
        vol = ret.rolling(window).std() * np.sqrt(252)
        df['volatility_20d'] = vol
        df['volatility_20d'] = df['volatility_20d'].fillna(0)
        return df[['volatility_20d']]

    def reversal_5d(self, close_col='close_x', period=5):
        df = self.df
        chg = df[close_col].pct_change(periods=period)
        df['reversal_5d'] = -chg
        df['reversal_5d'] = df['reversal_5d'].fillna(0)
        return df[['reversal_5d']]

    def high_low_spread(self, high_col='high', low_col='low', close_col='close_x'):
        df = self.df
        df['high_low_spread'] = (df[high_col] - df[low_col]) / df[close_col]
        df['high_low_spread'] = df['high_low_spread'].fillna(0)
        return df[['high_low_spread']]

    def update_factor(self, factor_list=None):
        # 1. 基本面因子对齐
        self._add_financial_factors()

        # 2. 技术量价因子
        if factor_list is None:
            calcu_list = [
                self.label, self.macd, self.kdj, self.mfi, self.rsi,
                self.cci, self.force_index, self.vwap, self.mtm_margin,
                self.macd_air_refuel, self.macd_divergence, self.big_order_ratio,
                self.lhb_strength_5d, self.vol_breakout, self.volatility_20d,
                self.reversal_5d, self.high_low_spread
            ]
        else:
            calcu_list = []
            for f in factor_list:
                if hasattr(self, f):
                    calcu_list.append(getattr(self, f))
                else:
                    print(f"警告：{f} 不是有效属性")

        for calc in calcu_list:
            calc()

        # ---- 写回 CSV 前，将 trade_date 恢复为原始整数格式 YYYYMMDD ----
        self.df['trade_date'] = self.df['trade_date'].dt.strftime('%Y%m%d').astype(int)

        self.df.to_csv(self.path, index=False)
        return True


# ================== 3. 主程序 ==================
if __name__ == '__main__':
    series_path = "../data/series/"
    fin_df = pd.read_csv("../data/financial.csv")

    print("预处理财务数据...")
    fin_features = build_financial_features(fin_df)
    print(f"完成，共 {len(fin_features)} 只股票。")

    files = glob.glob(series_path + "*.csv")
    total = len(files)
    for idx, file in enumerate(files, 1):
        print(f"\r处理 {idx}/{total}: {file}", end='', flush=True)
        fm = FactorManager(file, fin_features)
        fm.update_factor()
    print("\n全部完成！")