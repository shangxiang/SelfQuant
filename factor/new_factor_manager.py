import multiprocessing
import numpy as np
import pandas as pd
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


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

    # 增量日期更新时向前携带的历史行数（覆盖最长滚动窗口 252 日 + 缓冲）
    _ROLLING_LOOKBACK: int = 260

    # 列名 → 产出该列的方法名。用于"列维度增量"检测：若列不存在则触发对应方法重算。
    # 财务因子（_add_financial_factors）每次均在全量 df 上重跑，不纳入此字典。
    factor_to_fun: dict = {
        # ── 标签 ──────────────────────────────────────────────────────────
        'label':   'label',
        'label_1': 'label_1',
        'label_3': 'label_3',
        'label_10': 'label_10',
        'label_25': 'label_25',
        # ── MACD ─────────────────────────────────────────────────────────
        'dif': 'macd', 'dea': 'macd', 'macd': 'macd',
        # ── KDJ ──────────────────────────────────────────────────────────
        'K': 'kdj', 'D': 'kdj', 'J': 'kdj',
        # ── MFI（positive_flow / negative_flow 是 big_order_ratio 的中间依赖）
        'mfi': 'mfi', 'positive_flow': 'mfi', 'negative_flow': 'mfi',
        # ── 单列指标 ──────────────────────────────────────────────────────
        'rsi':                    'rsi',
        'cci':                    'cci',
        'force_index':            'force_index',
        'vwap':                   'vwap',
        'close_to_vwap_ratio':    'vwap',
        'mtm_margin_balance_change': 'mtm_margin',
        'macd_air_refuel':        'macd_air_refuel',
        'macd_divergence':        'macd_divergence',
        'big_order_ratio':        'big_order_ratio',
        'lhb_strength_5d':        'lhb_strength_5d',
        'vol_breakout':           'vol_breakout',
        'volatility_20d':         'volatility_20d',
        'reversal_5d':            'reversal_5d',
        'high_low_spread':        'high_low_spread',
        # ── Fama-French 风格因子 ──────────────────────────────────────────
        'size_factor':     'size_factor',
        'value_factor':    'value_factor',
        'cma_factor':      'cma_factor',
        'momentum_12_1':   'momentum_12_1',
        # ── 交叉因子 ──────────────────────────────────────────────────────
        'smb_squared':     'smb_squared',
        'smb_mom':         'smb_mom',
        'smb_squared_mom': 'smb_squared_mom',
        'hml_rmw':         'hml_rmw',
        'smb_hml':         'smb_hml',
        'vol_mom':         'vol_mom',
        # ── 中短期动量 / 技术形态 ─────────────────────────────────────────
        'ret_10d':             'ret_10d',
        'ret_20d':             'ret_20d',
        'ret_60d':             'ret_60d',
        'dist_52w_high':       'dist_52w_high',
        'close_ma20_ratio':    'close_ma20_ratio',
        'up_day_ratio_20':     'up_day_ratio_20',
        'vol_price_corr_20d':  'vol_price_corr_20d',
        'adx':                 'adx',
        # ── 高频痕迹 ──────────────────────────────────────────────────────
        'turnover_amplitude_ratio': 'turnover_amplitude_ratio',
        'long_shadow_freq':         'long_shadow_freq',
        'doji_freq':                'doji_freq',
        'intraday_drawdown':        'intraday_drawdown',
        'gap_vs_range_ratio':       'gap_vs_range_ratio',
        # ── 多项式形状因子 ────────────────────────────────────────────────
        'poly_close_a1': 'poly_shape', 'poly_close_a2': 'poly_shape',
        'poly_vol_a1':   'poly_shape', 'poly_vol_a2':   'poly_shape',
        # ── 时序差分特征（factor_time_series 的默认输出）──────────────────
        **{
            f'{f}_chg_{p}d': 'factor_time_series'
            for f in ('K', 'D', 'J', 'rsi', 'macd', 'adx',
                      'volatility_20d', 'turnover_rate_x',
                      'reversal_5d', 'momentum_12_1', 'rzye')
            for p in (5, 10)
        },
        # ── Alpha101 因子 ──────────────────────────────────────────────────
        **{f'alpha101_{i}': f'alpha101_{i}' for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 33, 34, 41, 52, 53, 54, 57, 101]},
        # ── Size 规模因子 ──────────────────────────────────────────────────
        'size': 'size', 'float_size': 'float_size',
        # ── Value 价值因子 ─────────────────────────────────────────────────
        'earnings_to_price': 'earnings_to_price', 'book_to_market': 'book_to_market',
        'ocf_to_market': 'ocf_to_market', 'fcf_to_market': 'fcf_to_market',
        'sales_to_market': 'sales_to_market',
        # ── Reversal 反转因子 ──────────────────────────────────────────────
        'small_cap_reversal_21d': 'small_cap_reversal_21d', 'price_dist': 'price_dist',
        # ── Momentum 动量因子（补充）──────────────────────────────────────
        'return_5d': 'return_5d', 'return_21d': 'return_21d', 'return_42d': 'return_42d',
        'return_63d': 'return_63d', 'return_126d': 'return_126d', 'return_252d': 'return_252d',
        'ma_20d': 'ma_20d', 'price_position_ir_60d': 'price_position_ir_60d',
        'rsrs': 'rsrs', 'days_down_up': 'days_down_up',
        # ── Risk 风险因子 ──────────────────────────────────────────────────
        'return_std_21d': 'return_std_21d', 'return_std_42d': 'return_std_42d',
        'return_std_63d': 'return_std_63d', 'return_std_126d': 'return_std_126d',
        'return_std_252d': 'return_std_252d',
        'sharpe_60d': 'sharpe_60d', 'sharpe_750d': 'sharpe_750d',
        'adjusted_sharpe_750d': 'adjusted_sharpe_750d',
        'high_low_21d': 'high_low_21d', 'high_low_42d': 'high_low_42d',
        'high_low_63d': 'high_low_63d', 'high_low_126d': 'high_low_126d',
        'high_low_252d': 'high_low_252d',
        'days_beyond_upper_lower_21d': 'days_beyond_upper_lower_21d',
        'log_price': 'log_price',
        # ── Liquidity 流动性因子 ───────────────────────────────────────────
        'avg_turnover_5d': 'avg_turnover_5d', 'avg_turnover_10d': 'avg_turnover_10d',
        'avg_turnover_20d': 'avg_turnover_20d', 'amount_ma_20d': 'amount_ma_20d',
        'turnover_ma_20d': 'turnover_ma_20d', 'sum_abs_rtn_amount_20d': 'sum_abs_rtn_amount_20d',
        'std_turnover_21d': 'std_turnover_21d', 'avg_turnover_21d': 'avg_turnover_21d',
        'std_turnover_42d': 'std_turnover_42d', 'avg_turnover_42d': 'avg_turnover_42d',
        'std_turnover_63d': 'std_turnover_63d', 'avg_turnover_63d': 'avg_turnover_63d',
        'std_turnover_126d': 'std_turnover_126d', 'avg_turnover_126d': 'avg_turnover_126d',
        'std_turnover_252d': 'std_turnover_252d', 'avg_turnover_252d': 'avg_turnover_252d',
        'bias_turn_21d_252d': 'bias_turn_21d_252d', 'bias_std_turn_21d_252d': 'bias_std_turn_21d_252d',
        'bias_turn_42d_252d': 'bias_turn_42d_252d', 'bias_turn_63d_252d': 'bias_turn_63d_252d',
        'bias_turn_126d_252d': 'bias_turn_126d_252d',
        'bias_turn_21d_504d': 'bias_turn_21d_504d', 'bias_std_turn_21d_504d': 'bias_std_turn_21d_504d',
        'bias_turn_42d_504d': 'bias_turn_42d_504d', 'bias_std_turn_42d_504d': 'bias_std_turn_42d_504d',
        'bias_turn_63d_504d': 'bias_turn_63d_504d', 'bias_std_turn_63d_504d': 'bias_std_turn_63d_504d',
        'bias_turn_126d_504d': 'bias_turn_126d_504d', 'bias_std_turn_126d_504d': 'bias_std_turn_126d_504d',
        'turnover_ma_20d_120d': 'turnover_ma_20d_120d',
        # ── Quality 质量因子 ──────────────────────────────────────────────
        'roe_ttm': 'roe_ttm', 'roa_ttm': 'roa_ttm', 'gross_margin': 'gross_margin',
        'net_margin': 'net_margin', 'debt_to_assets': 'debt_to_assets',
        'current_ratio': 'current_ratio', 'quick_ratio': 'quick_ratio',
        'cash_flow_to_debt': 'cash_flow_to_debt', 'accruals': 'accruals',
        'earnings_quality': 'earnings_quality',
        # ── Growth 成长因子 ───────────────────────────────────────────────
        'revenue_growth_yoy': 'revenue_growth_yoy', 'profit_growth_yoy': 'profit_growth_yoy',
        'asset_growth_yoy': 'asset_growth_yoy', 'roe_growth_yoy': 'roe_growth_yoy',
        'eps_growth_yoy': 'eps_growth_yoy', 'revenue_growth_qoq': 'revenue_growth_qoq',
        'profit_growth_qoq': 'profit_growth_qoq', 'gross_margin_growth': 'gross_margin_growth',
        'net_margin_growth': 'net_margin_growth', 'ocf_growth_yoy': 'ocf_growth_yoy',
    }

    def __init__(self, path: str, fin_features: dict):
        """
        Parameters
        ----------
        path         : str   股票日线文件的完整路径（含文件名），支持 .csv 和 .parquet
        fin_features : dict  由 build_financial_features() 返回的财务因子字典
        """
        self.path = path
        # 支持 Parquet 和 CSV 格式
        if path.endswith('.parquet'):
            self.df = pd.read_parquet(path)
        else:
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

        # 计算后复权价格列，供所有因子计算使用
        self._compute_hfq_prices()

    # ------------------------------------------------------------------ #
    #  后复权价格计算                                                       #
    # ------------------------------------------------------------------ #

    def _compute_hfq_prices(self) -> None:
        """
        根据复权因子计算后复权价格，供因子计算使用。

        后复权价格 = 不复权价格 × 复权因子。
        后复权价格序列保证了区间收益率的真实连续性（包含分红送股等公司行为），
        使得基于价格变化率的因子（MACD、RSI、动量等）不受除权缺口的干扰。

        生成的列：close_hfq, open_hfq, high_hfq, low_hfq
        若 adj_factor 列不存在，则后复权价格等同于不复权价格。
        """
        if 'adj_factor' not in self.df.columns:
            # 无复权因子数据，后复权价格退化为不复权价格
            self.df['close_hfq'] = self.df['close_x']
            self.df['open_hfq']  = self.df['open']
            self.df['high_hfq']  = self.df['high']
            self.df['low_hfq']   = self.df['low']
            return

        adj = self.df['adj_factor']
        self.df['close_hfq'] = self.df['close_x'] * adj
        self.df['open_hfq']  = self.df['open']  * adj
        self.df['high_hfq']  = self.df['high']  * adj
        self.df['low_hfq']   = self.df['low']   * adj

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
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。使用后复权价格。"""
        df = self.df
        df['label'] = (df['close_hfq'].shift(-period-1) - df['close_hfq'].shift(-1)) / df['close_hfq'].shift(-1)
        return df[['label']]

    def label_1(self, period: int = 1):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。使用后复权价格。"""
        df = self.df
        df['label_1'] = (df['close_hfq'].shift(-period-1) - df['close_hfq'].shift(-1)) / df['close_hfq'].shift(-1)
        return df[['label_1']]

    def label_3(self, period: int = 3):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。使用后复权价格。"""
        df = self.df
        df['label_3'] = (df['close_hfq'].shift(-period-1) - df['close_hfq'].shift(-1)) / df['close_hfq'].shift(-1)
        return df[['label_3']]

    def label_10(self, period: int = 10):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。使用后复权价格。"""
        df = self.df
        df['label_10'] = (df['close_hfq'].shift(-period-1) - df['close_hfq'].shift(-1)) / df['close_hfq'].shift(-1)
        return df[['label_10']]

    def label_25(self, period: int = 25):
        """预测标签：未来 period 日的涨跌幅（用于模型训练，实盘时末尾为 NaN）。使用后复权价格。"""
        df = self.df
        df['label_25'] = (df['close_hfq'].shift(-period-1) - df['close_hfq'].shift(-1)) / df['close_hfq'].shift(-1)
        return df[['label_25']]

    def macd(self, price_col='close_hfq', fast=12, slow=26, signal=9):
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

    def kdj(self, high_col='high_hfq', low_col='low_hfq', close_col='close_hfq',
            period=9, k_period=3, d_period=3):
        """
        KDJ 随机指标：K、D 为平滑后的超买超卖指标，J 为 K 和 D 的偏离度。
        RSV = (收盘 - N日最低) / (N日最高 - N日最低) * 100
        K = EWM(RSV, alpha=1/k_period)，D = EWM(K, alpha=1/d_period)
        最高最低价相等时（一字板等）RSV 设为 50，避免除零。
        """
        df = self.df
        lowest_low   = df[low_col].rolling(window=period, min_periods=1).min()
        highest_high = df[high_col].rolling(window=period, min_periods=1).max()
        denom = highest_high - lowest_low
        rsv = pd.Series(50.0, index=df.index, dtype=float)
        mask = denom != 0
        rsv[mask] = 100 * (df.loc[mask, close_col] - lowest_low[mask]) / denom[mask]

        # EWM 等价于原始递推：alpha = 1/k_period 时 new = alpha*x + (1-alpha)*prev
        alpha_k = 1.0 / k_period
        alpha_d = 1.0 / d_period
        df['K'] = rsv.ewm(alpha=alpha_k, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=alpha_d, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        return df[['K', 'D', 'J']]

    def rsi(self, price_col='close_hfq', period=14):
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

    def cci(self, high_col='high_hfq', low_col='low_hfq', close_col='close_hfq', period=14):
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

    def force_index(self, volume_col='vol', price_col='close_hfq', window=1):
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
        df['close_to_vwap_ratio'] = (df['close_hfq'] - df['vwap']) / df['vwap']
        return df[['vwap', 'close_to_vwap_ratio']]

    def mfi(self, high_col='high_hfq', low_col='low_hfq', close_col='close_hfq',
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
        low_min  = df['low_hfq'].rolling(window).min()
        macd_min = df['macd'].rolling(window).min()
        low_new_low  = (df['low_hfq']  < low_min.shift(1))   # 价格创新低
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

    def volatility_20d(self, close_col='close_hfq', window=20):
        """
        20 日历史波动率（年化）：日收益率的滚动标准差 × √252。
        用于衡量个股风险水平，可作为风险控制因子使用。
        """
        df = self.df
        ret = df[close_col].pct_change()
        vol = ret.rolling(window).std() * np.sqrt(252)
        df['volatility_20d'] = vol.fillna(0)
        return df[['volatility_20d']]

    def reversal_5d(self, close_col='close_hfq', period=5):
        """
        5 日反转因子：过去 5 日累计涨幅取反。
        短期内涨幅越大，反转因子越小（预期均值回归向下），反之亦然。
        """
        df = self.df
        chg = df[close_col].pct_change(periods=period)
        df['reversal_5d'] = -chg   # 取负号：近期跌得多 → 因子值大 → 预期反弹
        df['reversal_5d'] = df['reversal_5d'].fillna(0)
        return df[['reversal_5d']]

    def high_low_spread(self, high_col='high_hfq', low_col='low_hfq', close_col='close_hfq'):
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

    def momentum_12_1(self, close_col='close_hfq', long_window=252, short_window=21):
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
    #  中短期动量 / 技术形态因子                                             #
    # ------------------------------------------------------------------ #

    def ret_10d(self, close_col='close_hfq'):
        """10 日价格动量：捕捉短期延续效应，补充 reversal_5d 与 momentum_12_1 之间的空白。"""
        df = self.df
        df['ret_10d'] = df[close_col].pct_change(periods=10).fillna(0)
        return df[['ret_10d']]

    def ret_20d(self, close_col='close_hfq'):
        """20 日价格动量：月度级别趋势，与 reversal 和 12-1 动量互补。"""
        df = self.df
        df['ret_20d'] = df[close_col].pct_change(periods=20).fillna(0)
        return df[['ret_20d']]

    def ret_60d(self, close_col='close_hfq'):
        """60 日价格动量：季度级别中期趋势。"""
        df = self.df
        df['ret_60d'] = df[close_col].pct_change(periods=60).fillna(0)
        return df[['ret_60d']]

    def dist_52w_high(self, close_col='close_hfq', window=252):
        """
        距 52 周高点的距离：(收盘价 / 252日最高收盘价) - 1，值域 (-∞, 0]。
        接近 52 周高点的股票往往处于强势趋势（George & Hwang 2004 动量解释）。
        """
        df = self.df
        rolling_max = df[close_col].rolling(window=window, min_periods=20).max()
        df['dist_52w_high'] = (df[close_col] / rolling_max.replace(0, np.nan) - 1).fillna(0)
        return df[['dist_52w_high']]

    def close_ma20_ratio(self, close_col='close_hfq', window=20):
        """
        收盘价相对 20 日均线偏离度：(close / MA20) - 1。
        正值表示价格在均线上方（偏强），负值在下方（偏弱），捕捉短期均值回归或趋势延续。
        """
        df = self.df
        ma20 = df[close_col].rolling(window=window, min_periods=5).mean()
        df['close_ma20_ratio'] = (df[close_col] / ma20.replace(0, np.nan) - 1).fillna(0)
        return df[['close_ma20_ratio']]

    def up_day_ratio_20(self, close_col='close_hfq', window=20):
        """
        20 日上涨天数占比：滚动 20 日内收盘价上涨的交易日比例。
        衡量趋势一致性，区别于单纯的累计涨幅（避免大涨小跌噪音）。
        """
        df = self.df
        up_flag = (df[close_col].diff() > 0).astype(float)
        df['up_day_ratio_20'] = up_flag.rolling(window=window, min_periods=5).mean().fillna(0)
        return df[['up_day_ratio_20']]

    def vol_price_corr_20d(self, close_col='close_hfq', vol_col='vol', window=20):
        """
        量价相关性（20 日）：价格日收益率与成交量日变化率的滚动相关系数。
        正值（量价齐升/同步下跌）通常为趋势延续信号；负值（量价背离）暗示反转。
        """
        df = self.df
        price_ret = df[close_col].pct_change()
        vol_chg   = df[vol_col].pct_change()
        df['vol_price_corr_20d'] = price_ret.rolling(window=window, min_periods=10).corr(vol_chg).fillna(0)
        return df[['vol_price_corr_20d']]

    def adx(self, high_col='high_hfq', low_col='low_hfq', close_col='close_hfq', period=14):
        """
        平均趋向指数（ADX）：Wilder 方法，衡量趋势强度（不含方向）。
        ADX > 25 通常认为趋势显著；<20 为盘整。
        使用 EWM（alpha=1/period）近似 Wilder 平滑，与标准定义等价。
        """
        df = self.df
        high, low, close = df[high_col], df[low_col], df[close_col]
        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        up_move   = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm  = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        plus_dm[(up_move > down_move) & (up_move > 0)]     = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)]  = down_move

        alpha = 1.0 / period
        atr      = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0)
        return df[['adx']]

    # ------------------------------------------------------------------ #
    #  高频痕迹因子                                                         #
    # ------------------------------------------------------------------ #

    def turnover_amplitude_ratio(self, turnover_col='turnover_rate_x',
                                  high_col='high_hfq', low_col='low_hfq', window=20):
        """
        换手率振幅比：换手率 / (最高价/最低价 - 1) 的20日均值。
        高频刷单放大成交但压缩价格波动，该比率异常放大是做市算法的典型痕迹。
        振幅为0时跳过，避免除零。
        """
        df = self.df
        amplitude = (df[high_col] / df[low_col].replace(0, np.nan) - 1).replace(0, np.nan)
        daily_ratio = df[turnover_col] / amplitude
        df['turnover_amplitude_ratio'] = (
            daily_ratio.rolling(window=window, min_periods=10).mean().fillna(0)
        )
        return df[['turnover_amplitude_ratio']]

    def long_shadow_freq(self, open_col='open_hfq', high_col='high_hfq',
                          low_col='low_hfq', close_col='close_hfq', window=20):
        """
        长影线频率：过去20日出现长影线（上/下影线 > 实体×3）的天数占比。
        高频算法试探盘口深度后迅速撤退，在日K线上留下极长影线。
        """
        df = self.df
        body         = (df[close_col] - df[open_col]).abs()
        upper_shadow = df[high_col] - df[[open_col, close_col]].max(axis=1)
        lower_shadow = df[[open_col, close_col]].min(axis=1) - df[low_col]
        long_shadow  = (upper_shadow > body * 3) | (lower_shadow > body * 3)
        df['long_shadow_freq'] = (
            long_shadow.astype(float).rolling(window=window, min_periods=10).mean().fillna(0)
        )
        return df[['long_shadow_freq']]

    def doji_freq(self, open_col='open_hfq', high_col='high_hfq',
                   low_col='low_hfq', close_col='close_hfq', window=20, threshold=0.2):
        """
        十字星频率：过去20日实体占比（|收-开| / (最高-最低)）< 0.2 的天数占比。
        高频拉锯使收盘价反复回到开盘价附近，十字星频现是算法博弈的"指纹"。
        """
        df = self.df
        body       = (df[close_col] - df[open_col]).abs()
        total_range = (df[high_col] - df[low_col]).replace(0, np.nan)
        body_ratio  = (body / total_range).fillna(0)
        is_doji     = (body_ratio < threshold).astype(float)
        df['doji_freq'] = (
            is_doji.rolling(window=window, min_periods=10).mean().fillna(0)
        )
        return df[['doji_freq']]

    def intraday_drawdown(self, high_col='high_hfq', close_col='close_hfq', window=20):
        """
        日内回撤幅度：过去20日 (最高价 - 收盘价) / 最高价 的均值。
        高频突然撤单引发的"冲高回落"痕迹，均值越大说明盘中瞬间崩盘越频繁。
        """
        df = self.df
        daily_drawdown = (
            (df[high_col] - df[close_col]) / df[high_col].replace(0, np.nan)
        )
        df['intraday_drawdown'] = (
            daily_drawdown.rolling(window=window, min_periods=10).mean().fillna(0)
        )
        return df[['intraday_drawdown']]

    def gap_vs_range_ratio(self, open_col='open_hfq', high_col='high_hfq',
                            low_col='low_hfq', close_col='close_hfq', window=20):
        """
        隔夜跳空/日内波动比：20日平均隔夜跳空幅度 / 20日平均日内振幅。
        高频策略主要在盘中活动，使日内波动远大于隔夜跳空，该比值越低说明盘中
        算法干扰越强。
        """
        df = self.df
        gap           = (df[open_col] - df[close_col].shift(1)).abs() \
                        / df[close_col].shift(1).replace(0, np.nan)
        intraday_rng  = (df[high_col] / df[low_col].replace(0, np.nan) - 1)
        gap_mean      = gap.rolling(window=window, min_periods=10).mean()
        range_mean    = intraday_rng.rolling(window=window, min_periods=10).mean().replace(0, np.nan)
        df['gap_vs_range_ratio'] = (gap_mean / range_mean).fillna(0)
        return df[['gap_vs_range_ratio']]

    # ------------------------------------------------------------------ #
    #  Alpha101 因子（世坤经典因子）                                        #
    # ------------------------------------------------------------------ #

    def _ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        """时序排名：当前值在过去window天中的百分位排名（0-1）"""
        return series.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    def _ts_argmax(self, series: pd.Series, window: int) -> pd.Series:
        """时序最大值位置：过去window天中最大值的天数索引"""
        return series.rolling(window).apply(lambda x: x.argmax(), raw=True)

    def _ts_argmin(self, series: pd.Series, window: int) -> pd.Series:
        """时序最小值位置：过去window天中最小值的天数索引"""
        return series.rolling(window).apply(lambda x: x.argmin(), raw=True)

    def _signed_power(self, series: pd.Series, power: float) -> pd.Series:
        """带符号的幂运算：保留符号的幂次"""
        return np.sign(series) * (series.abs() ** power)

    def alpha101_1(self):
        """Alpha101-1: 基于负收益时段波动放大的极值位置排名因子"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        std20 = returns.rolling(20).std()
        # IF(Returns < 0, StdDev(Returns, 20), Close)
        condition = np.where(returns < 0, std20, df['close_hfq'])
        # Ts_ArgMax(SignedPower(..., 2), 5)
        argmax = self._ts_argmax(self._signed_power(pd.Series(condition, index=df.index), 2), 5)
        # Rank(...) - 0.5
        df['alpha101_1'] = (argmax.rank(pct=True) - 0.5).fillna(0.5)
        return df[['alpha101_1']]

    def alpha101_2(self):
        """Alpha101-2: 开盘价排名与成交量排名相关性的负值"""
        df = self.df
        # -1 * Correlation(Rank(Delta(Log(Volume), 2)), Rank((Close - Open) / Open), 6)
        log_vol_delta = np.log(df['vol']).diff(2)
        close_open_ratio = (df['close_hfq'] - df['open_hfq']) / df['open_hfq'].replace(0, np.nan)
        corr = log_vol_delta.rank(pct=True).rolling(6).corr(close_open_ratio.rank(pct=True))
        df['alpha101_2'] = (-corr).fillna(0)
        return df[['alpha101_2']]

    def alpha101_3(self):
        """Alpha101-3: 开盘价排名与成交量排名相关性的负值"""
        df = self.df
        # -1 * Correlation(Rank(Open), Rank(Volume), 10)
        corr = df['open_hfq'].rank(pct=True).rolling(10).corr(df['vol'].rank(pct=True))
        df['alpha101_3'] = (-corr).fillna(0)
        return df[['alpha101_3']]

    def alpha101_4(self):
        """Alpha101-4: 最低价排名的9日时间序列排名的负值"""
        df = self.df
        # -1 * Ts_Rank(Rank(Low), 9)
        low_rank = df['low_hfq'].rank(pct=True)
        df['alpha101_4'] = (-self._ts_rank(low_rank, 9)).fillna(0.5)
        return df[['alpha101_4']]

    def alpha101_5(self):
        """Alpha101-5: 开盘价偏离VWAP排名与收盘价偏离VWAP排名绝对值的组合"""
        df = self.df
        vwap = df['vwap'] if 'vwap' in df.columns else (df['amount_x'] / df['vol'] * 10)
        vwap10 = vwap.rolling(10).mean()
        # Rank(Open - (Sum(VWAP, 10) / 10)) * (-1 * Abs(Rank(Close - VWAP)))
        part1 = (df['open_hfq'] - vwap10).rank(pct=True)
        part2 = -((df['close_hfq'] - vwap).rank(pct=True)).abs()
        df['alpha101_5'] = (part1 * part2).fillna(0)
        return df[['alpha101_5']]

    def alpha101_6(self):
        """Alpha101-6: 开盘价与成交量10日相关性的负值"""
        df = self.df
        # -1 * Correlation(Open, Volume, 10)
        corr = df['open_hfq'].rolling(10).corr(df['vol'])
        df['alpha101_6'] = (-corr).fillna(0)
        return df[['alpha101_6']]

    def alpha101_7(self):
        """Alpha101-7: 成交量条件判断因子"""
        df = self.df
        adv20 = df['vol'].rolling(20).mean()
        delta_close7 = df['close_hfq'].diff(7)
        # (ADV20 < Volume) ? (-1 * Ts_Rank(Abs(Delta(Close, 7)), 60) * Sign(Delta(Close, 7))) : -1
        condition = adv20 < df['vol']
        ts_rank = self._ts_rank(delta_close7.abs(), 60)
        sign = np.sign(delta_close7)
        df['alpha101_7'] = pd.Series(np.where(condition, -ts_rank * sign, -1), index=df.index).fillna(-1)
        return df[['alpha101_7']]

    def alpha101_8(self):
        """Alpha101-8: 5日开盘价与收益率乘积和相对其10日延迟值之差的排名负值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        sum_open5 = df['open_hfq'].rolling(5).sum()
        sum_ret5 = returns.rolling(5).sum()
        product = sum_open5 * sum_ret5
        # -1 * Rank((Sum(Open, 5) * Sum(Returns, 5)) - Delay(..., 10))
        delta = product - product.shift(10)
        df['alpha101_8'] = (-delta.rank(pct=True)).fillna(0)
        return df[['alpha101_8']]

    def alpha101_9(self):
        """Alpha101-9: 收盘价变化的条件判断因子"""
        df = self.df
        delta_close = df['close_hfq'].diff(1)
        ts_min5 = delta_close.rolling(5).min()
        ts_max5 = delta_close.rolling(5).max()
        # (0 < Ts_Min(Delta(Close, 1), 5)) ? Delta(Close, 1) :
        #   ((Ts_Max(Delta(Close, 1), 5) < 0) ? Delta(Close, 1) : -1 * Delta(Close, 1))
        condition1 = ts_min5 > 0
        condition2 = ts_max5 < 0
        result = np.where(condition1, delta_close,
                         np.where(condition2, delta_close, -delta_close))
        df['alpha101_9'] = pd.Series(result, index=df.index).fillna(0)
        return df[['alpha101_9']]

    def alpha101_10(self):
        """Alpha101-10: 收盘价变化的排名条件判断因子"""
        df = self.df
        delta_close = df['close_hfq'].diff(1)
        ts_min4 = delta_close.rolling(4).min()
        ts_max4 = delta_close.rolling(4).max()
        condition1 = ts_min4 > 0
        condition2 = ts_max4 < 0
        result = np.where(condition1, delta_close,
                         np.where(condition2, delta_close, -delta_close))
        df['alpha101_10'] = pd.Series(result, index=df.index).rank(pct=True).fillna(0.5)
        return df[['alpha101_10']]

    def alpha101_11(self):
        """Alpha101-11: VWAP与收盘价差值的时序极值排名"""
        df = self.df
        vwap = df['vwap'] if 'vwap' in df.columns else (df['amount_x'] / df['vol'] * 10)
        diff = vwap - df['close_hfq']
        ts_max3 = diff.rolling(3).max()
        ts_min3 = diff.rolling(3).min()
        delta_vol3 = df['vol'].diff(3)
        # (Rank(Ts_Max(VWAP - Close, 3)) + Rank(Ts_Min(VWAP - Close, 3))) * Rank(Delta(Volume, 3))
        df['alpha101_11'] = ((ts_max3.rank(pct=True) + ts_min3.rank(pct=True)) *
                             delta_vol3.rank(pct=True)).fillna(0)
        return df[['alpha101_11']]

    def alpha101_12(self):
        """Alpha101-12: 成交量变化符号与收盘价变化的乘积"""
        df = self.df
        # Sign(Delta(Volume, 1)) * (-1 * Delta(Close, 1))
        vol_sign = np.sign(df['vol'].diff(1))
        delta_close = df['close_hfq'].diff(1)
        df['alpha101_12'] = (vol_sign * -delta_close).fillna(0)
        return df[['alpha101_12']]

    def alpha101_13(self):
        """Alpha101-13: 收盘价排名与成交量排名的协方差排名负值"""
        df = self.df
        # -1 * Rank(Covariance(Rank(Close), Rank(Volume), 5))
        cov = df['close_hfq'].rank(pct=True).rolling(5).cov(df['vol'].rank(pct=True))
        df['alpha101_13'] = (-cov.rank(pct=True)).fillna(0)
        return df[['alpha101_13']]

    def alpha101_14(self):
        """Alpha101-14: 收益率变化排名与开盘价成交量相关性的乘积负值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        delta_ret3 = returns.diff(3)
        corr = df['open_hfq'].rolling(10).corr(df['vol'])
        # -1 * Rank(Delta(Returns, 3)) * Correlation(Open, Volume, 10)
        df['alpha101_14'] = (-delta_ret3.rank(pct=True) * corr).fillna(0)
        return df[['alpha101_14']]

    def alpha101_15(self):
        """Alpha101-15: 最高价排名与成交量排名相关性的排名累加和负值"""
        df = self.df
        # -1 * Sum(Rank(Correlation(Rank(High), Rank(Volume), 3)), 3)
        corr = df['high_hfq'].rank(pct=True).rolling(3).corr(df['vol'].rank(pct=True))
        df['alpha101_15'] = (-corr.rank(pct=True).rolling(3).sum()).fillna(0)
        return df[['alpha101_15']]

    def alpha101_16(self):
        """Alpha101-16: 最高价排名与成交量排名的协方差排名负值"""
        df = self.df
        # -1 * Rank(Covariance(Rank(High), Rank(Volume), 5))
        cov = df['high_hfq'].rank(pct=True).rolling(5).cov(df['vol'].rank(pct=True))
        df['alpha101_16'] = (-cov.rank(pct=True)).fillna(0)
        return df[['alpha101_16']]

    def alpha101_17(self):
        """Alpha101-17: 收盘价排名、二阶差分排名、成交量相对ADV20排名的组合"""
        df = self.df
        adv20 = df['vol'].rolling(20).mean()
        delta_close = df['close_hfq'].diff(1)
        delta2_close = delta_close.diff(1)
        # (-1 * Rank(Ts_Rank(Close, 10))) * Rank(Delta(Delta(Close, 1), 1)) * Rank(Ts_Rank(Volume / ADV20, 5))
        part1 = -self._ts_rank(df['close_hfq'], 10).rank(pct=True)
        part2 = delta2_close.rank(pct=True)
        part3 = (df['vol'] / adv20.replace(0, np.nan)).rank(pct=True)
        part3 = self._ts_rank(part3, 5)
        df['alpha101_17'] = (part1 * part2 * part3).fillna(0)
        return df[['alpha101_17']]

    def alpha101_18(self):
        """Alpha101-18: 收盘价与开盘价差值的波动、相关性组合排名负值"""
        df = self.df
        diff = df['close_hfq'] - df['open_hfq']
        std5 = diff.abs().rolling(5).std()
        corr10 = df['close_hfq'].rolling(10).corr(df['open_hfq'])
        # -1 * Rank(StdDev(Abs(Close - Open), 5) + (Close - Open) + Correlation(Close, Open, 10))
        df['alpha101_18'] = (-(std5 + diff + corr10).rank(pct=True)).fillna(0)
        return df[['alpha101_18']]

    def alpha101_19(self):
        """Alpha101-19: 复杂的趋势反转信号（简化版）"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        sum_ret250 = returns.rolling(250).sum()
        delta_close7 = df['close_hfq'].diff(7)
        # 简化实现：基于趋势强度和反转信号
        trend_strength = np.sign(df['close_hfq'] - df['close_hfq'].shift(7))
        reversal = -np.sign(delta_close7)
        df['alpha101_19'] = (trend_strength * (1 + sum_ret250.rank(pct=True)) +
                             reversal * df['close_hfq'].rank(pct=True)).fillna(0)
        return df[['alpha101_19']]

    def alpha101_20(self):
        """Alpha101-20: 开盘价与昨日高低价差值的排名乘积负值"""
        df = self.df
        # -1 * Rank(Open - Delay(High, 1)) * Rank(Open - Delay(Close, 1)) * Rank(Open - Delay(Low, 1))
        part1 = (df['open_hfq'] - df['high_hfq'].shift(1)).rank(pct=True)
        part2 = (df['open_hfq'] - df['close_hfq'].shift(1)).rank(pct=True)
        part3 = (df['open_hfq'] - df['low_hfq'].shift(1)).rank(pct=True)
        df['alpha101_20'] = (-part1 * part2 * part3).fillna(0)
        return df[['alpha101_20']]

    def alpha101_22(self):
        """Alpha101-22: 最高价与成交量相关性变化与收盘价波动的乘积负值"""
        df = self.df
        corr5 = df['high_hfq'].rolling(5).corr(df['vol'])
        delta_corr5 = corr5.diff(5)
        std20 = df['close_hfq'].rolling(20).std()
        # -1 * Delta(Correlation(High, Volume, 5), 5) * Rank(StdDev(Close, 20))
        df['alpha101_22'] = (-delta_corr5 * std20.rank(pct=True)).fillna(0)
        return df[['alpha101_22']]

    def alpha101_23(self):
        """Alpha101-23: 最高价突破20日均值的条件因子"""
        df = self.df
        sum_high20 = df['high_hfq'].rolling(20).sum() / 20
        delta_high2 = df['high_hfq'].diff(2)
        # (Sum(High, 20) / 20 < High) ? -1 * Delta(High, 2) : 0
        condition = sum_high20 < df['high_hfq']
        df['alpha101_23'] = pd.Series(np.where(condition, -delta_high2, 0), index=df.index).fillna(0)
        return df[['alpha101_23']]

    def alpha101_25(self):
        """Alpha101-25: 收益率、成交量、VWAP、高低价差的排名乘积"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        adv20 = df['vol'].rolling(20).mean()
        vwap = df['vwap'] if 'vwap' in df.columns else (df['amount_x'] / df['vol'] * 10)
        # Rank(-1 * Returns * ADV20 * VWAP * (High - Close))
        product = -returns * adv20 * vwap * (df['high_hfq'] - df['close_hfq'])
        df['alpha101_25'] = product.rank(pct=True).fillna(0.5)
        return df[['alpha101_25']]

    def alpha101_33(self):
        """Alpha101-33: 开盘价与收盘价比值的排名"""
        df = self.df
        # Rank(-1 * (1 - Open / Close))
        ratio = 1 - df['open_hfq'] / df['close_hfq'].replace(0, np.nan)
        df['alpha101_33'] = (-ratio).rank(pct=True).fillna(0.5)
        return df[['alpha101_33']]

    def alpha101_34(self):
        """Alpha101-34: 收益率波动比率与收盘价变化的排名组合"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        std2 = returns.rolling(2).std()
        std5 = returns.rolling(5).std()
        delta_close = df['close_hfq'].diff(1)
        # Rank(1 - Rank(StdDev(Returns, 2) / StdDev(Returns, 5)) + 1 - Rank(Delta(Close, 1)))
        ratio = std2 / std5.replace(0, np.nan)
        df['alpha101_34'] = ((1 - ratio.rank(pct=True)) + (1 - delta_close.rank(pct=True))).rank(pct=True).fillna(0.5)
        return df[['alpha101_34']]

    def alpha101_41(self):
        """Alpha101-41: 最高价最低价几何平均与VWAP的差值"""
        df = self.df
        vwap = df['vwap'] if 'vwap' in df.columns else (df['amount_x'] / df['vol'] * 10)
        # (High * Low) ^ 0.5 - VWAP
        df['alpha101_41'] = ((df['high_hfq'] * df['low_hfq']) ** 0.5 - vwap).fillna(0)
        return df[['alpha101_41']]

    def alpha101_52(self):
        """Alpha101-52: 最低价变化、收益率差、成交量排名的组合"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        sum_ret240 = returns.rolling(240).sum()
        sum_ret20 = returns.rolling(20).sum()
        ts_min_low5 = df['low_hfq'].rolling(5).min()
        # ((-1 * Ts_Min(Low, 5) + Delay(Ts_Min(Low, 5), 5)) * Rank((Sum(Returns, 240) - Sum(Returns, 20)) / 220)) * Ts_Rank(Volume, 5)
        part1 = -ts_min_low5 + ts_min_low5.shift(5)
        part2 = ((sum_ret240 - sum_ret20) / 220).rank(pct=True)
        part3 = self._ts_rank(df['vol'], 5)
        df['alpha101_52'] = (part1 * part2 * part3).fillna(0)
        return df[['alpha101_52']]

    def alpha101_53(self):
        """Alpha101-53: 收盘价在日内区间的位置变化"""
        df = self.df
        # -1 * Delta(((Close - Low) - (High - Close)) / (Close - Low), 9)
        numerator = (df['close_hfq'] - df['low_hfq']) - (df['high_hfq'] - df['close_hfq'])
        denominator = (df['close_hfq'] - df['low_hfq']).replace(0, np.nan)
        ratio = numerator / denominator
        df['alpha101_53'] = (-ratio.diff(9)).fillna(0)
        return df[['alpha101_53']]

    def alpha101_54(self):
        """Alpha101-54: 开盘价与收盘价的幂次比值"""
        df = self.df
        # (-1 * (Low - Close) * Open^5) / ((Low - High) * Close^5)
        numerator = -1 * (df['low_hfq'] - df['close_hfq']) * (df['open_hfq'] ** 5)
        denominator = (df['low_hfq'] - df['high_hfq']) * (df['close_hfq'] ** 5)
        df['alpha101_54'] = (numerator / denominator.replace(0, np.nan)).fillna(0)
        return df[['alpha101_54']]

    def alpha101_57(self):
        """Alpha101-57: 收盘价与VWAP差值与收盘价排名的线性衰减比值"""
        df = self.df
        vwap = df['vwap'] if 'vwap' in df.columns else (df['amount_x'] / df['vol'] * 10)
        ts_argmax = self._ts_argmax(df['close_hfq'], 30)
        # -(Close - VWAP) / Decay_Linear(Rank(Ts_ArgMax(Close, 30)), 2)
        decay = ts_argmax.rank(pct=True).ewm(span=2, adjust=False).mean()
        df['alpha101_57'] = (-(df['close_hfq'] - vwap) / decay.replace(0, np.nan)).fillna(0)
        return df[['alpha101_57']]

    def alpha101_101(self):
        """Alpha101-101: 收盘价与开盘价差值占日内振幅的比例"""
        df = self.df
        # (Close - Open) / ((High - Low) + 0.001)
        numerator = df['close_hfq'] - df['open_hfq']
        denominator = (df['high_hfq'] - df['low_hfq']) + 0.001
        df['alpha101_101'] = (numerator / denominator).fillna(0)
        return df[['alpha101_101']]

    # ------------------------------------------------------------------ #
    #  Size 规模因子                                                        #
    # ------------------------------------------------------------------ #

    def size(self):
        """
        总市值因子：计算总市值的负对数,用于衡量公司规模。
        Size = -log(TotalShares * ClosePrice / 1e6)
        该因子值为负对数形式,值越小表示市值越大。
        """
        df = self.df
        if 'total_mv' in df.columns:
            # total_mv单位为万元,转换为亿元
            df['size'] = -np.log(df['total_mv'] / 100).fillna(0)
        else:
            df['size'] = 0
        return df[['size']]

    def float_size(self):
        """
        流通市值因子：计算流通市值的负对数,用于衡量公司可交易部分的规模。
        FloatSize = -log(FloatShares * ClosePrice / 1e6)
        该因子值为负对数形式,值越小表示流通市值越大。
        """
        df = self.df
        if 'circ_mv' in df.columns:
            # circ_mv单位为万元,转换为亿元
            df['float_size'] = -np.log(df['circ_mv'] / 100).fillna(0)
        else:
            df['float_size'] = 0
        return df[['float_size']]

    # ------------------------------------------------------------------ #
    #  Value 价值因子                                                       #
    # ------------------------------------------------------------------ #

    def earnings_to_price(self):
        """
        市盈率倒数(E/P)：归母净利润TTM / 市值
        earnings_to_price = EPS_TTM / ClosePrice
        越高说明股票越"便宜"(价值越高)。
        """
        df = self.df
        if 'eps' in df.columns and 'close_x' in df.columns:
            df['earnings_to_price'] = (df['eps'] / df['close_x'].replace(0, np.nan)).fillna(0)
        else:
            df['earnings_to_price'] = 0
        return df[['earnings_to_price']]

    def book_to_market(self):
        """
        账面市值比(B/M)：账面价值 / 市值
        book_to_market = (归母股东权益 + 递延所得税资产) / (收盘价 × 总股本)
        越高说明股票越"便宜"(价值越高)。
        """
        df = self.df
        if 'total_hldr_eqy_exc_min_int' in df.columns and 'total_mv' in df.columns:
            # total_mv单位为万元,转换为元; total_hldr_eqy_exc_min_int单位为元
            market_cap = df['total_mv'] * 10000
            df['book_to_market'] = (df['total_hldr_eqy_exc_min_int'] / market_cap.replace(0, np.nan)).fillna(0)
        else:
            df['book_to_market'] = 0
        return df[['book_to_market']]

    def ocf_to_market(self):
        """
        经营现金流市值比：NetOperateCashFlow_TTM / 市值
        越高说明每单位市值对应的经营活动现金流越多。
        """
        df = self.df
        if 'n_cashflow_act' in df.columns and 'total_mv' in df.columns:
            market_cap = df['total_mv'] * 10000
            df['ocf_to_market'] = (df['n_cashflow_act'] / market_cap.replace(0, np.nan)).fillna(0)
        else:
            df['ocf_to_market'] = 0
        return df[['ocf_to_market']]

    def fcf_to_market(self):
        """
        自由现金流市值比：自由现金流TTM / 市值
        fcf_to_market = (NOCF_TTM - SICO_TTM) / (ClosePrice × TotalShares)
        越高说明公司的自由现金流创造能力越强。
        """
        df = self.df
        if 'free_cashflow' in df.columns and 'total_mv' in df.columns:
            market_cap = df['total_mv'] * 10000
            df['fcf_to_market'] = (df['free_cashflow'] / market_cap.replace(0, np.nan)).fillna(0)
        else:
            df['fcf_to_market'] = 0
        return df[['fcf_to_market']]

    def sales_to_market(self):
        """
        营业收入市值比：营业收入Q / 市值
        越高说明每单位市值对应的营业收入越多。
        """
        df = self.df
        if 'total_revenue' in df.columns and 'total_mv' in df.columns:
            market_cap = df['total_mv'] * 10000
            df['sales_to_market'] = (df['total_revenue'] / market_cap.replace(0, np.nan)).fillna(0)
        else:
            df['sales_to_market'] = 0
        return df[['sales_to_market']]

    # ------------------------------------------------------------------ #
    #  Reversal 反转因子                                                    #
    # ------------------------------------------------------------------ #

    def small_cap_reversal_21d(self, window=21):
        """
        小盘反转因子：选市值最小的股票,取过去21日累计收益的反转信号。
        市值越小、前期涨幅越低的股票得分越高。
        """
        df = self.df
        returns = df['close_hfq'].pct_change()
        cum_return = (1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)
        if 'circ_mv' in df.columns:
            # 市值越小、收益越低,得分越高
            df['small_cap_reversal_21d'] = (-cum_return / df['circ_mv'].replace(0, np.nan)).fillna(0)
        else:
            df['small_cap_reversal_21d'] = -cum_return.fillna(0)
        return df[['small_cap_reversal_21d']]

    def price_dist(self, window=0):
        """
        价格距离因子：计算股价与其下一个整数(或10、100的倍数)的距离。
        捕捉价格的心理整数关口效应。
        """
        df = self.df
        price = df['close_hfq']

        def calc_dist(p):
            if p < 10:
                return np.ceil(p) - p
            elif p < 100:
                return np.ceil(p / 10) * 10 - p
            else:
                return np.ceil(p / 100) * 100 - p

        df['price_dist'] = price.apply(calc_dist)
        if window > 0:
            df['price_dist'] = df['price_dist'].rolling(window).mean()
        return df[['price_dist']]

    # ------------------------------------------------------------------ #
    #  Momentum 动量因子(补充)                                              #
    # ------------------------------------------------------------------ #

    def return_5d(self, window=5):
        """5日累计收益率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_5d'] = ((1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)).fillna(0)
        return df[['return_5d']]

    def return_21d(self, window=21):
        """21日累计收益率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_21d'] = ((1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)).fillna(0)
        return df[['return_21d']]

    def return_42d(self, window=42):
        """42日累计收益率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_42d'] = ((1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)).fillna(0)
        return df[['return_42d']]

    def return_63d(self, window=63):
        """63日累计收益率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_63d'] = ((1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)).fillna(0)
        return df[['return_63d']]

    def return_126d(self, window=126):
        """126日累计收益率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_126d'] = ((1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)).fillna(0)
        return df[['return_126d']]

    def return_252d(self, window=252):
        """252日累计收益率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_252d'] = ((1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=False)).fillna(0)
        return df[['return_252d']]

    def ma_20d(self, window=20):
        """20日移动平均线"""
        df = self.df
        df['ma_20d'] = df['close_hfq'].rolling(window).mean().fillna(0)
        return df[['ma_20d']]

    def price_position_ir_60d(self, window=60):
        """
        价格位置动量因子：计算过去60日(收盘-开盘)/(最高-最低)比率的信息比率。
        Factor = Mean(Ratio, 60) / StdDev(Ratio, 60)
        """
        df = self.df
        ratio = (df['close_hfq'] - df['open_hfq']) / (df['high_hfq'] - df['low_hfq']).replace(0, np.nan)
        mean_ratio = ratio.rolling(window).mean()
        std_ratio = ratio.rolling(window).std()
        df['price_position_ir_60d'] = (mean_ratio / std_ratio.replace(0, np.nan)).fillna(0)
        return df[['price_position_ir_60d']]

    def rsrs(self, regress_window=18, zscore_window=200):
        """
        RSRS指标：通过回归最高价和最低价得到斜率,再对斜率进行标准化。
        1. Slope_t = Beta from OLS(Low ~ High), N=18
        2. RSRS_t = Z-Score(Slope_{t-M+1:t}), M=200
        """
        df = self.df
        # 滚动回归计算斜率
        slopes = []
        for i in range(len(df)):
            if i < regress_window - 1:
                slopes.append(np.nan)
                continue
            high = df['high_hfq'].iloc[i-regress_window+1:i+1].values
            low = df['low_hfq'].iloc[i-regress_window+1:i+1].values
            # 停牌日会留下 NaN，直接喂给 polyfit 会触发 "SVD did not converge"
            ok = np.isfinite(high) & np.isfinite(low)
            if ok.sum() < 2 or np.std(high[ok]) == 0:
                slopes.append(np.nan)
                continue
            try:
                slope = float(np.polyfit(high[ok], low[ok], 1)[0])
            except Exception:
                slope = np.nan
            slopes.append(slope)

        slope_series = pd.Series(slopes, index=df.index)
        # Z-Score标准化
        mean_slope = slope_series.rolling(zscore_window).mean()
        std_slope = slope_series.rolling(zscore_window).std()
        df['rsrs'] = ((slope_series - mean_slope) / std_slope.replace(0, np.nan)).fillna(0)
        return df[['rsrs']]

    def days_down_up(self):
        """
        连续涨跌天数因子：计算连续上涨天数与连续下跌天数之差的绝对值减1。
        Factor = |ConsecutiveUp - ConsecutiveDown - 1|
        """
        df = self.df
        returns = df['close_hfq'].pct_change()

        # 计算连续上涨天数
        up_streak = []
        current_up = 0
        for r in returns:
            if pd.isna(r):
                up_streak.append(0)
                continue
            if r > 0:
                current_up += 1
            else:
                current_up = 0
            up_streak.append(current_up)

        # 计算连续下跌天数
        down_streak = []
        current_down = 0
        for r in returns:
            if pd.isna(r):
                down_streak.append(0)
                continue
            if r < 0:
                current_down += 1
            else:
                current_down = 0
            down_streak.append(current_down)

        df['days_down_up'] = (pd.Series(up_streak, index=df.index) -
                              pd.Series(down_streak, index=df.index)).abs() - 1
        df['days_down_up'] = df['days_down_up'].fillna(0)
        return df[['days_down_up']]

    # ------------------------------------------------------------------ #
    #  Risk 风险因子                                                        #
    # ------------------------------------------------------------------ #

    def return_std_21d(self, window=21):
        """21日收益率标准差"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_std_21d'] = returns.rolling(window).std().fillna(0)
        return df[['return_std_21d']]

    def return_std_42d(self, window=42):
        """42日收益率标准差"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_std_42d'] = returns.rolling(window).std().fillna(0)
        return df[['return_std_42d']]

    def return_std_63d(self, window=63):
        """63日收益率标准差"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_std_63d'] = returns.rolling(window).std().fillna(0)
        return df[['return_std_63d']]

    def return_std_126d(self, window=126):
        """126日收益率标准差"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_std_126d'] = returns.rolling(window).std().fillna(0)
        return df[['return_std_126d']]

    def return_std_252d(self, window=252):
        """252日收益率标准差"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        df['return_std_252d'] = returns.rolling(window).std().fillna(0)
        return df[['return_std_252d']]

    def sharpe_60d(self, window=60):
        """60日夏普比率：Mean(Return, 60) / StdDev(Return, 60)"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        mean_ret = returns.rolling(window).mean()
        std_ret = returns.rolling(window).std()
        df['sharpe_60d'] = (mean_ret / std_ret.replace(0, np.nan)).fillna(0)
        return df[['sharpe_60d']]

    def sharpe_750d(self, window=750):
        """750日夏普比率"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        mean_ret = returns.rolling(window).mean()
        std_ret = returns.rolling(window).std()
        df['sharpe_750d'] = (mean_ret / std_ret.replace(0, np.nan)).fillna(0)
        return df[['sharpe_750d']]

    def adjusted_sharpe_750d(self, window=750):
        """750日调整夏普率：Mean / Std^4,对高波动性惩罚更重"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        mean_ret = returns.rolling(window).mean()
        std_ret = returns.rolling(window).std()
        df['adjusted_sharpe_750d'] = (mean_ret / (std_ret ** 4).replace(0, np.nan)).fillna(0)
        return df[['adjusted_sharpe_750d']]

    def high_low_21d(self, window=21):
        """21日净值曲线最高点与最低点的比值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        net_value = (1 + returns).cumprod()
        rolling_max = net_value.rolling(window).max()
        rolling_min = net_value.rolling(window).min()
        df['high_low_21d'] = (rolling_max / rolling_min.replace(0, np.nan)).fillna(1)
        return df[['high_low_21d']]

    def high_low_42d(self, window=42):
        """42日净值曲线最高点与最低点的比值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        net_value = (1 + returns).cumprod()
        rolling_max = net_value.rolling(window).max()
        rolling_min = net_value.rolling(window).min()
        df['high_low_42d'] = (rolling_max / rolling_min.replace(0, np.nan)).fillna(1)
        return df[['high_low_42d']]

    def high_low_63d(self, window=63):
        """63日净值曲线最高点与最低点的比值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        net_value = (1 + returns).cumprod()
        rolling_max = net_value.rolling(window).max()
        rolling_min = net_value.rolling(window).min()
        df['high_low_63d'] = (rolling_max / rolling_min.replace(0, np.nan)).fillna(1)
        return df[['high_low_63d']]

    def high_low_126d(self, window=126):
        """126日净值曲线最高点与最低点的比值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        net_value = (1 + returns).cumprod()
        rolling_max = net_value.rolling(window).max()
        rolling_min = net_value.rolling(window).min()
        df['high_low_126d'] = (rolling_max / rolling_min.replace(0, np.nan)).fillna(1)
        return df[['high_low_126d']]

    def high_low_252d(self, window=252):
        """252日净值曲线最高点与最低点的比值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        net_value = (1 + returns).cumprod()
        rolling_max = net_value.rolling(window).max()
        rolling_min = net_value.rolling(window).min()
        df['high_low_252d'] = (rolling_max / rolling_min.replace(0, np.nan)).fillna(1)
        return df[['high_low_252d']]

    def days_beyond_upper_lower_21d(self, window=21):
        """
        21日内价格超越均值±标准差的天数之差。
        Factor = Upper - Lower
        """
        df = self.df
        mean_price = df['close_hfq'].rolling(window).mean()
        std_price = df['close_hfq'].rolling(window).std()
        upper = mean_price + std_price
        lower = mean_price - std_price

        beyond_upper = (df['close_hfq'] > upper).astype(int).rolling(window).sum()
        beyond_lower = (df['close_hfq'] < lower).astype(int).rolling(window).sum()
        df['days_beyond_upper_lower_21d'] = (beyond_upper - beyond_lower).fillna(0)
        return df[['days_beyond_upper_lower_21d']]

    def log_price(self):
        """收盘价的自然对数"""
        df = self.df
        df['log_price'] = np.log(df['close_hfq'].replace(0, np.nan)).fillna(0)
        return df[['log_price']]

    # ------------------------------------------------------------------ #
    #  Liquidity 流动性因子                                                 #
    # ------------------------------------------------------------------ #

    def avg_turnover_5d(self, window=5):
        """5日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_5d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_5d'] = 0
        return df[['avg_turnover_5d']]

    def avg_turnover_10d(self, window=10):
        """10日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_10d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_10d'] = 0
        return df[['avg_turnover_10d']]

    def avg_turnover_20d(self, window=20):
        """20日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_20d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_20d'] = 0
        return df[['avg_turnover_20d']]

    def amount_ma_20d(self, window=20):
        """20日成交额移动平均"""
        df = self.df
        if 'amount_x' in df.columns:
            df['amount_ma_20d'] = df['amount_x'].rolling(window).mean().fillna(0)
        else:
            df['amount_ma_20d'] = 0
        return df[['amount_ma_20d']]

    def turnover_ma_20d(self, window=20):
        """20日成交量与流通市值比率的移动平均"""
        df = self.df
        if 'vol' in df.columns and 'circ_mv' in df.columns:
            vol_cap_ratio = df['vol'] / df['circ_mv'].replace(0, np.nan)
            df['turnover_ma_20d'] = -vol_cap_ratio.rolling(window).mean().fillna(0)
        else:
            df['turnover_ma_20d'] = 0
        return df[['turnover_ma_20d']]

    def sum_abs_rtn_amount_20d(self, window=20):
        """20日累计绝对收益率与累计成交额的比值"""
        df = self.df
        returns = df['close_hfq'].pct_change()
        abs_ret_sum = returns.abs().rolling(window).sum()
        if 'amount_x' in df.columns:
            amount_sum = df['amount_x'].rolling(window).sum()
            df['sum_abs_rtn_amount_20d'] = (abs_ret_sum / amount_sum.replace(0, np.nan)).fillna(0)
        else:
            df['sum_abs_rtn_amount_20d'] = 0
        return df[['sum_abs_rtn_amount_20d']]

    def std_turnover_21d(self, window=21):
        """21日换手率滚动标准差"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['std_turnover_21d'] = df['turnover_rate_x'].rolling(window).std().fillna(0)
        else:
            df['std_turnover_21d'] = 0
        return df[['std_turnover_21d']]

    def avg_turnover_21d(self, window=21):
        """21日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_21d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_21d'] = 0
        return df[['avg_turnover_21d']]

    def std_turnover_42d(self, window=42):
        """42日换手率滚动标准差"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['std_turnover_42d'] = df['turnover_rate_x'].rolling(window).std().fillna(0)
        else:
            df['std_turnover_42d'] = 0
        return df[['std_turnover_42d']]

    def avg_turnover_42d(self, window=42):
        """42日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_42d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_42d'] = 0
        return df[['avg_turnover_42d']]

    def std_turnover_63d(self, window=63):
        """63日换手率滚动标准差"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['std_turnover_63d'] = df['turnover_rate_x'].rolling(window).std().fillna(0)
        else:
            df['std_turnover_63d'] = 0
        return df[['std_turnover_63d']]

    def avg_turnover_63d(self, window=63):
        """63日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_63d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_63d'] = 0
        return df[['avg_turnover_63d']]

    def std_turnover_126d(self, window=126):
        """126日换手率滚动标准差"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['std_turnover_126d'] = df['turnover_rate_x'].rolling(window).std().fillna(0)
        else:
            df['std_turnover_126d'] = 0
        return df[['std_turnover_126d']]

    def avg_turnover_126d(self, window=126):
        """126日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_126d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_126d'] = 0
        return df[['avg_turnover_126d']]

    def std_turnover_252d(self, window=252):
        """252日换手率滚动标准差"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['std_turnover_252d'] = df['turnover_rate_x'].rolling(window).std().fillna(0)
        else:
            df['std_turnover_252d'] = 0
        return df[['std_turnover_252d']]

    def avg_turnover_252d(self, window=252):
        """252日平均换手率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            df['avg_turnover_252d'] = df['turnover_rate_x'].rolling(window).mean().fillna(0)
        else:
            df['avg_turnover_252d'] = 0
        return df[['avg_turnover_252d']]

    def bias_turn_21d_252d(self):
        """21日平均换手率与252日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(21).mean()
            ma_long = df['turnover_rate_x'].rolling(252).mean()
            df['bias_turn_21d_252d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_21d_252d'] = 0
        return df[['bias_turn_21d_252d']]

    def bias_std_turn_21d_252d(self):
        """21日换手率标准差与252日换手率标准差的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            std_short = df['turnover_rate_x'].rolling(21).std()
            std_long = df['turnover_rate_x'].rolling(252).std()
            df['bias_std_turn_21d_252d'] = (std_short / std_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_std_turn_21d_252d'] = 0
        return df[['bias_std_turn_21d_252d']]

    def bias_turn_42d_252d(self):
        """42日平均换手率与252日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(42).mean()
            ma_long = df['turnover_rate_x'].rolling(252).mean()
            df['bias_turn_42d_252d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_42d_252d'] = 0
        return df[['bias_turn_42d_252d']]

    def bias_turn_63d_252d(self):
        """63日平均换手率与252日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(63).mean()
            ma_long = df['turnover_rate_x'].rolling(252).mean()
            df['bias_turn_63d_252d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_63d_252d'] = 0
        return df[['bias_turn_63d_252d']]

    def bias_turn_126d_252d(self):
        """126日平均换手率与252日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(126).mean()
            ma_long = df['turnover_rate_x'].rolling(252).mean()
            df['bias_turn_126d_252d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_126d_252d'] = 0
        return df[['bias_turn_126d_252d']]

    def bias_turn_21d_504d(self):
        """21日平均换手率与504日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(21).mean()
            ma_long = df['turnover_rate_x'].rolling(504).mean()
            df['bias_turn_21d_504d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_21d_504d'] = 0
        return df[['bias_turn_21d_504d']]

    def bias_std_turn_21d_504d(self):
        """21日换手率标准差与504日换手率标准差的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            std_short = df['turnover_rate_x'].rolling(21).std()
            std_long = df['turnover_rate_x'].rolling(504).std()
            df['bias_std_turn_21d_504d'] = (std_short / std_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_std_turn_21d_504d'] = 0
        return df[['bias_std_turn_21d_504d']]

    def bias_turn_42d_504d(self):
        """42日平均换手率与504日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(42).mean()
            ma_long = df['turnover_rate_x'].rolling(504).mean()
            df['bias_turn_42d_504d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_42d_504d'] = 0
        return df[['bias_turn_42d_504d']]

    def bias_std_turn_42d_504d(self):
        """42日换手率标准差与504日换手率标准差的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            std_short = df['turnover_rate_x'].rolling(42).std()
            std_long = df['turnover_rate_x'].rolling(504).std()
            df['bias_std_turn_42d_504d'] = (std_short / std_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_std_turn_42d_504d'] = 0
        return df[['bias_std_turn_42d_504d']]

    def bias_turn_63d_504d(self):
        """63日平均换手率与504日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(63).mean()
            ma_long = df['turnover_rate_x'].rolling(504).mean()
            df['bias_turn_63d_504d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_63d_504d'] = 0
        return df[['bias_turn_63d_504d']]

    def bias_std_turn_63d_504d(self):
        """63日换手率标准差与504日换手率标准差的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            std_short = df['turnover_rate_x'].rolling(63).std()
            std_long = df['turnover_rate_x'].rolling(504).std()
            df['bias_std_turn_63d_504d'] = (std_short / std_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_std_turn_63d_504d'] = 0
        return df[['bias_std_turn_63d_504d']]

    def bias_turn_126d_504d(self):
        """126日平均换手率与504日平均换手率的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            ma_short = df['turnover_rate_x'].rolling(126).mean()
            ma_long = df['turnover_rate_x'].rolling(504).mean()
            df['bias_turn_126d_504d'] = (ma_short / ma_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_turn_126d_504d'] = 0
        return df[['bias_turn_126d_504d']]

    def bias_std_turn_126d_504d(self):
        """126日换手率标准差与504日换手率标准差的乖离率"""
        df = self.df
        if 'turnover_rate_x' in df.columns:
            std_short = df['turnover_rate_x'].rolling(126).std()
            std_long = df['turnover_rate_x'].rolling(504).std()
            df['bias_std_turn_126d_504d'] = (std_short / std_long.replace(0, np.nan) - 1).fillna(0)
        else:
            df['bias_std_turn_126d_504d'] = 0
        return df[['bias_std_turn_126d_504d']]

    def turnover_ma_20d_120d(self):
        """20日成交量与流通市值比率与120日的比值"""
        df = self.df
        if 'vol' in df.columns and 'circ_mv' in df.columns:
            vol_cap_ratio = df['vol'] / df['circ_mv'].replace(0, np.nan)
            ma_short = vol_cap_ratio.rolling(20).mean()
            ma_long = vol_cap_ratio.rolling(120).mean()
            df['turnover_ma_20d_120d'] = (ma_short / ma_long.replace(0, np.nan)).fillna(0)
        else:
            df['turnover_ma_20d_120d'] = 0
        return df[['turnover_ma_20d_120d']]

    # ------------------------------------------------------------------ #
    #  Quality 质量因子                                                     #
    # ------------------------------------------------------------------ #

    def roe_ttm(self):
        """TTM净资产收益率"""
        df = self.df
        if 'roe_ttm' in df.columns:
            df['roe_ttm'] = df['roe_ttm'].fillna(0)
        else:
            df['roe_ttm'] = 0
        return df[['roe_ttm']]

    def roa_ttm(self):
        """TTM总资产收益率"""
        df = self.df
        if 'total_assets' in df.columns and 'n_income' in df.columns:
            df['roa_ttm'] = (df['n_income'] / df['total_assets'].replace(0, np.nan)).fillna(0)
        else:
            df['roa_ttm'] = 0
        return df[['roa_ttm']]

    def gross_margin(self):
        """毛利率"""
        df = self.df
        if 'gross_margin' in df.columns:
            df['gross_margin'] = df['gross_margin'].fillna(0)
        else:
            df['gross_margin'] = 0
        return df[['gross_margin']]

    def net_margin(self):
        """净利率"""
        df = self.df
        if 'total_revenue' in df.columns and 'n_income' in df.columns:
            df['net_margin'] = (df['n_income'] / df['total_revenue'].replace(0, np.nan)).fillna(0)
        else:
            df['net_margin'] = 0
        return df[['net_margin']]

    def debt_to_assets(self):
        """资产负债率"""
        df = self.df
        if 'debt_ratio' in df.columns:
            df['debt_to_assets'] = df['debt_ratio'].fillna(0)
        else:
            df['debt_to_assets'] = 0
        return df[['debt_to_assets']]

    def current_ratio(self):
        """流动比率"""
        df = self.df
        if 'total_cur_assets' in df.columns and 'total_cur_liab' in df.columns:
            df['current_ratio'] = (df['total_cur_assets'] / df['total_cur_liab'].replace(0, np.nan)).fillna(0)
        else:
            df['current_ratio'] = 0
        return df[['current_ratio']]

    def quick_ratio(self):
        """速动比率"""
        df = self.df
        if 'total_cur_assets' in df.columns and 'inventories' in df.columns and 'total_cur_liab' in df.columns:
            df['quick_ratio'] = ((df['total_cur_assets'] - df['inventories']) /
                                 df['total_cur_liab'].replace(0, np.nan)).fillna(0)
        else:
            df['quick_ratio'] = 0
        return df[['quick_ratio']]

    def cash_flow_to_debt(self):
        """现金流负债比"""
        df = self.df
        if 'n_cashflow_act' in df.columns and 'total_liab' in df.columns:
            df['cash_flow_to_debt'] = (df['n_cashflow_act'] / df['total_liab'].replace(0, np.nan)).fillna(0)
        else:
            df['cash_flow_to_debt'] = 0
        return df[['cash_flow_to_debt']]

    def accruals(self):
        """应计项目"""
        df = self.df
        if 'accruals' in df.columns:
            df['accruals'] = df['accruals'].fillna(0)
        else:
            df['accruals'] = 0
        return df[['accruals']]

    def earnings_quality(self):
        """盈余质量：经营现金流/净利润"""
        df = self.df
        if 'n_cashflow_act' in df.columns and 'n_income' in df.columns:
            df['earnings_quality'] = (df['n_cashflow_act'] / df['n_income'].replace(0, np.nan)).fillna(0)
        else:
            df['earnings_quality'] = 0
        return df[['earnings_quality']]

    # ------------------------------------------------------------------ #
    #  Growth 成长因子                                                      #
    # ------------------------------------------------------------------ #

    def revenue_growth_yoy(self):
        """营收同比增长率"""
        df = self.df
        if 'revenue_growth_yoy' in df.columns:
            df['revenue_growth_yoy'] = df['revenue_growth_yoy'].fillna(0)
        else:
            df['revenue_growth_yoy'] = 0
        return df[['revenue_growth_yoy']]

    def profit_growth_yoy(self):
        """净利润同比增长率"""
        df = self.df
        if 'profit_growth_yoy' in df.columns:
            df['profit_growth_yoy'] = df['profit_growth_yoy'].fillna(0)
        else:
            df['profit_growth_yoy'] = 0
        return df[['profit_growth_yoy']]

    def asset_growth_yoy(self):
        """资产同比增长率"""
        df = self.df
        if 'asset_growth_yoy' in df.columns:
            df['asset_growth_yoy'] = df['asset_growth_yoy'].fillna(0)
        else:
            df['asset_growth_yoy'] = 0
        return df[['asset_growth_yoy']]

    def roe_growth_yoy(self):
        """ROE同比增长率"""
        df = self.df
        if 'roe_ttm' in df.columns:
            roe_current = df['roe_ttm']
            roe_prev = df['roe_ttm'].shift(252)
            df['roe_growth_yoy'] = ((roe_current - roe_prev) / roe_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['roe_growth_yoy'] = 0
        return df[['roe_growth_yoy']]

    def eps_growth_yoy(self):
        """EPS同比增长率"""
        df = self.df
        if 'eps' in df.columns:
            eps_current = df['eps']
            eps_prev = df['eps'].shift(252)
            df['eps_growth_yoy'] = ((eps_current - eps_prev) / eps_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['eps_growth_yoy'] = 0
        return df[['eps_growth_yoy']]

    def revenue_growth_qoq(self):
        """营收环比增长率"""
        df = self.df
        if 'total_revenue' in df.columns:
            rev_current = df['total_revenue']
            rev_prev = df['total_revenue'].shift(63)
            df['revenue_growth_qoq'] = ((rev_current - rev_prev) / rev_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['revenue_growth_qoq'] = 0
        return df[['revenue_growth_qoq']]

    def profit_growth_qoq(self):
        """净利润环比增长率"""
        df = self.df
        if 'n_income' in df.columns:
            profit_current = df['n_income']
            profit_prev = df['n_income'].shift(63)
            df['profit_growth_qoq'] = ((profit_current - profit_prev) / profit_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['profit_growth_qoq'] = 0
        return df[['profit_growth_qoq']]

    def gross_margin_growth(self):
        """毛利率增长率"""
        df = self.df
        if 'gross_margin' in df.columns:
            gm_current = df['gross_margin']
            gm_prev = df['gross_margin'].shift(252)
            df['gross_margin_growth'] = ((gm_current - gm_prev) / gm_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['gross_margin_growth'] = 0
        return df[['gross_margin_growth']]

    def net_margin_growth(self):
        """净利率增长率"""
        df = self.df
        if 'total_revenue' in df.columns and 'n_income' in df.columns:
            nm_current = df['n_income'] / df['total_revenue'].replace(0, np.nan)
            nm_prev = (df['n_income'].shift(252) / df['total_revenue'].shift(252).replace(0, np.nan))
            df['net_margin_growth'] = ((nm_current - nm_prev) / nm_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['net_margin_growth'] = 0
        return df[['net_margin_growth']]

    def ocf_growth_yoy(self):
        """经营现金流同比增长率"""
        df = self.df
        if 'n_cashflow_act' in df.columns:
            ocf_current = df['n_cashflow_act']
            ocf_prev = df['n_cashflow_act'].shift(252)
            df['ocf_growth_yoy'] = ((ocf_current - ocf_prev) / ocf_prev.abs().replace(0, np.nan)).fillna(0)
        else:
            df['ocf_growth_yoy'] = 0
        return df[['ocf_growth_yoy']]

    def poly_shape(self, close_col='close_hfq', vol_col='vol', window=6):
        """
        对最近 window 天的价格和成交量拟合二次多项式，提取形状因子。

        输出 4 列：
          poly_close_a1 — 价格线性系数（趋势方向与斜率）
          poly_close_a2 — 价格二次系数（趋势加速/减速，正=加速，负=减速）
          poly_vol_a1   — 成交量线性系数（量能趋势方向）
          poly_vol_a2   — 成交量二次系数（量能加速/减速）

        归一化方式：
          价格：y = close / close[window_start] - 1，以窗口首日为基准转成收益率序列
          成交量：先除以 20 日滚动均量（消除股票间量级差异），再窗口内减均值

        预计算伪逆矩阵，避免逐窗口 polyfit，效率更高。
        """
        df = self.df
        t = np.arange(window, dtype=float)
        # 设计矩阵：[t², t, 1]，形状 (window, 3)
        T = np.column_stack([t ** 2, t, np.ones(window)])
        T_pinv = np.linalg.pinv(T)   # 形状 (3, window)，预计算一次
        w_a2 = T_pinv[0]             # 提取 a2 的权重向量
        w_a1 = T_pinv[1]             # 提取 a1 的权重向量

        n = len(df)

        # ---- 价格多项式 ----
        close_arr = df[close_col].astype(float).values
        a1_close = np.zeros(n)
        a2_close = np.zeros(n)
        for i in range(window - 1, n):
            chunk = close_arr[i - window + 1: i + 1]
            if not np.all(np.isfinite(chunk)) or chunk[0] == 0:
                continue
            y = chunk / chunk[0] - 1          # 归一化为收益率序列
            a1_close[i] = w_a1 @ y
            a2_close[i] = w_a2 @ y

        df['poly_close_a1'] = a1_close
        df['poly_close_a2'] = a2_close

        # ---- 成交量多项式 ----
        vol_mean20 = df[vol_col].astype(float).rolling(20, min_periods=10).mean()
        vol_norm = (df[vol_col].astype(float) / vol_mean20.replace(0, np.nan)).fillna(1.0).values
        a1_vol = np.zeros(n)
        a2_vol = np.zeros(n)
        for i in range(window - 1, n):
            chunk = vol_norm[i - window + 1: i + 1]
            if not np.all(np.isfinite(chunk)):
                continue
            y = chunk - chunk.mean()          # 窗口内中心化，只保留形状
            a1_vol[i] = w_a1 @ y
            a2_vol[i] = w_a2 @ y

        df['poly_vol_a1'] = a1_vol
        df['poly_vol_a2'] = a2_vol

    def factor_time_series(self, factors: list = None, periods: list = None):
        """
        批量计算因子时序差分特征（必须在所有基础因子计算完成后调用）。

        对指定因子列计算 N 日差分：{factor}_chg_{N}d = factor(t) - factor(t-N)
        正值表示因子近期上升，负值表示下降，LGBM 可直接捕捉变化方向与幅度。

        Parameters
        ----------
        factors : list[str] | None  要计算差分的列名，None 时使用默认列表
        periods : list[int] | None  差分周期（交易日），None 时默认 [5, 10]
        """
        if factors is None:
            factors = [
                # 技术振荡器（KDJ、RSI）
                'K', 'D', 'J',
                'rsi',
                # 趋势指标
                'macd',
                'adx',
                # 波动率与换手
                'volatility_20d',
                'turnover_rate_x',
                # 反转与动量
                'reversal_5d',
                'momentum_12_1',
                # 融资余额（原始值变化，区别于 mtm_margin_balance_change 的百分比变化）
                'rzye',
            ]
        if periods is None:
            periods = [5, 10]

        df = self.df
        for f in factors:
            if f not in df.columns:
                continue
            series = df[f].astype(float)
            for p in periods:
                df[f'{f}_chg_{p}d'] = series.diff(p).fillna(0)

    # ------------------------------------------------------------------ #
    #  增量辅助                                                            #
    # ------------------------------------------------------------------ #

    def _get_recorded_end(self) -> 'Optional[pd.Timestamp]':
        """
        从 data/series/_date_range.csv 读取本股票的已记录截止日期。
        文件不存在或本股票无记录时返回 None（触发全量计算）。
        """
        config_path = os.path.join(os.path.dirname(self.path), '_date_range.csv')
        if not os.path.exists(config_path):
            return None
        cfg = pd.read_csv(config_path, dtype=str)
        row = cfg[cfg['ts_code'] == self.code]
        if row.empty:
            return None
        return pd.to_datetime(row['end_date'].iloc[0], format='%Y%m%d')

    def _incremental_date_update(self, calcu_list: list,
                                  recorded_end: 'pd.Timestamp') -> None:
        """
        仅对 trade_date > recorded_end 的新行运行因子计算，旧行保持原值。

        计算时向前携带 _ROLLING_LOOKBACK 行历史上下文，保证滚动窗口
        （最长 252 日动量/52 周高点等）在新行上计算正确。
        """
        full_df = self.df  # 已含 _add_financial_factors 结果

        old_mask = full_df['trade_date'] <= recorded_end
        old_df   = full_df[old_mask].copy()
        new_df   = full_df[~old_mask].copy()

        if new_df.empty:
            return

        # 取旧行末尾若干行作为滚动上下文
        context    = old_df.iloc[max(0, len(old_df) - self._ROLLING_LOOKBACK):]
        compute_df = pd.concat([context, new_df], ignore_index=True)

        self.df = compute_df
        for calc in calcu_list:
            calc()

        # 只保留计算结果中的新行，旧行值不变
        new_computed = self.df[self.df['trade_date'] > recorded_end].copy()

        self.df = pd.concat([old_df, new_computed], ignore_index=True)
        self.df.sort_values('trade_date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------ #
    #  统一计算入口                                                         #
    # ------------------------------------------------------------------ #

    def update_factor(self, factor_list: list = None) -> bool:
        """
        计算因子并将结果写回原 CSV 文件。

        增量策略（factor_list=None 时生效）：
          列维度：检查 factor_to_fun 中的列是否全部存在于文件中。
                  若有缺失，只运行缺失列对应的函数（保持依赖顺序）。
          行维度：若列完整但文件含有 _date_range.csv 记录日期之后的新行，
                  只对新行运行计算（携带 _ROLLING_LOOKBACK 行历史上下文）。
          若列完整且无新行，跳过全部技术因子计算直接写盘。

        财务因子（_add_financial_factors）每次均在全量 df 上重跑，不受增量逻辑影响。
        显式传入 factor_list 时，始终对全量 df 执行指定因子的计算。
        """
        # 1. 基本面因子：全量对齐（merge_asof 幂等，每次重跑保证财报更新能同步）
        self._add_financial_factors()

        # 2. 构建完整技术因子调用序列（维护顺序即依赖顺序）
        full_calcu_list = [
            self.label_1, self.label_3,
            self.label, self.label_10, self.label_25,
            self.macd, self.kdj, self.mfi, self.rsi,
            self.cci, self.force_index, self.vwap, self.mtm_margin,
            self.macd_air_refuel, self.macd_divergence, self.big_order_ratio,
            self.lhb_strength_5d, self.vol_breakout, self.volatility_20d,
            self.reversal_5d, self.high_low_spread,
            self.size_factor, self.value_factor, self.cma_factor, self.momentum_12_1,
            self.smb_squared, self.smb_mom, self.smb_squared_mom,
            self.hml_rmw, self.smb_hml, self.vol_mom,
            self.ret_10d, self.ret_20d, self.ret_60d,
            self.dist_52w_high, self.close_ma20_ratio,
            self.up_day_ratio_20, self.vol_price_corr_20d, self.adx,
            self.turnover_amplitude_ratio, self.long_shadow_freq,
            self.doji_freq, self.intraday_drawdown, self.gap_vs_range_ratio,
            self.poly_shape,
            # Alpha101因子
            self.alpha101_1, self.alpha101_2, self.alpha101_3, self.alpha101_4,
            self.alpha101_5, self.alpha101_6, self.alpha101_7, self.alpha101_8,
            self.alpha101_9, self.alpha101_10, self.alpha101_11, self.alpha101_12,
            self.alpha101_13, self.alpha101_14, self.alpha101_15, self.alpha101_16,
            self.alpha101_17, self.alpha101_18, self.alpha101_19, self.alpha101_20,
            self.alpha101_22, self.alpha101_23, self.alpha101_25, self.alpha101_33,
            self.alpha101_34, self.alpha101_41, self.alpha101_52, self.alpha101_53,
            self.alpha101_54, self.alpha101_57, self.alpha101_101,
            # Size因子
            self.size, self.float_size,
            # Value因子
            self.earnings_to_price, self.book_to_market, self.ocf_to_market,
            self.fcf_to_market, self.sales_to_market,
            # Reversal因子
            self.small_cap_reversal_21d, self.price_dist,
            # Momentum因子
            self.return_5d, self.return_21d, self.return_42d, self.return_63d,
            self.return_126d, self.return_252d, self.ma_20d, self.price_position_ir_60d,
            self.rsrs, self.days_down_up,
            # Risk因子
            self.return_std_21d, self.return_std_42d, self.return_std_63d,
            self.return_std_126d, self.return_std_252d,
            self.sharpe_60d, self.sharpe_750d, self.adjusted_sharpe_750d,
            self.high_low_21d, self.high_low_42d, self.high_low_63d,
            self.high_low_126d, self.high_low_252d,
            self.days_beyond_upper_lower_21d, self.log_price,
            # Liquidity因子
            self.avg_turnover_5d, self.avg_turnover_10d, self.avg_turnover_20d,
            self.amount_ma_20d, self.turnover_ma_20d, self.sum_abs_rtn_amount_20d,
            self.std_turnover_21d, self.avg_turnover_21d,
            self.std_turnover_42d, self.avg_turnover_42d,
            self.std_turnover_63d, self.avg_turnover_63d,
            self.std_turnover_126d, self.avg_turnover_126d,
            self.std_turnover_252d, self.avg_turnover_252d,
            self.bias_turn_21d_252d, self.bias_std_turn_21d_252d,
            self.bias_turn_42d_252d, self.bias_turn_63d_252d, self.bias_turn_126d_252d,
            self.bias_turn_21d_504d, self.bias_std_turn_21d_504d,
            self.bias_turn_42d_504d, self.bias_std_turn_42d_504d,
            self.bias_turn_63d_504d, self.bias_std_turn_63d_504d,
            self.bias_turn_126d_504d, self.bias_std_turn_126d_504d,
            self.turnover_ma_20d_120d,
            # Quality因子
            self.roe_ttm, self.roa_ttm, self.gross_margin, self.net_margin,
            self.debt_to_assets, self.current_ratio, self.quick_ratio,
            self.cash_flow_to_debt, self.accruals, self.earnings_quality,
            # Growth因子
            self.revenue_growth_yoy, self.profit_growth_yoy, self.asset_growth_yoy,
            self.roe_growth_yoy, self.eps_growth_yoy, self.revenue_growth_qoq,
            self.profit_growth_qoq, self.gross_margin_growth, self.net_margin_growth,
            self.ocf_growth_yoy,
            self.factor_time_series,
        ]

        if factor_list is not None:
            # 显式指定列表：全量计算
            calcu_list = []
            for f in factor_list:
                if hasattr(self, f):
                    calcu_list.append(getattr(self, f))
                else:
                    print(f"警告：{f} 不是有效的因子方法名")
            for calc in calcu_list:
                calc()
        else:
            # ── 列维度增量 ──────────────────────────────────────────────
            missing_cols = [c for c in self.factor_to_fun if c not in self.df.columns]
            if missing_cols:
                needed_funs = {self.factor_to_fun[c] for c in missing_cols}
                # 从 full_calcu_list 中过滤出缺失列对应的函数（保持原始顺序）
                calcu_list = [f for f in full_calcu_list if f.__name__ in needed_funs]
                for calc in calcu_list:
                    calc()
            else:
                # ── 日期维度增量 ────────────────────────────────────────
                recorded_end = self._get_recorded_end()
                if recorded_end is None:
                    # 无 config 记录 → 全量计算
                    for calc in full_calcu_list:
                        calc()
                elif (self.df['trade_date'] > recorded_end).any():
                    # 有新行 → 只对新行计算
                    self._incremental_date_update(full_calcu_list, recorded_end)
                # 否则：列完整、无新行 → 跳过全部技术因子计算

        # 写盘前将 trade_date 从 datetime 恢复为原始整数格式（YYYYMMDD）
        self.df['trade_date'] = self.df['trade_date'].dt.strftime('%Y%m%d').astype(int)
        # 支持 Parquet 和 CSV 格式
        if self.path.endswith('.parquet'):
            self.df.to_parquet(self.path, index=False)
        else:
            self.df.to_csv(self.path, index=False)
        return True



# ================== 3. 并行工作函数（必须在模块顶层，ProcessPoolExecutor 才能 pickle）==================

def _process_one_stock(args: tuple) -> tuple:
    """处理单只股票的因子计算，返回 (file, ts_code, start_date, end_date)。"""
    file, fin_features = args
    fm = FactorManager(file, fin_features)
    fm.update_factor()
    # update_factor() 最后将 trade_date 转回 YYYYMMDD int，直接读 min/max
    start_date = str(int(fm.df['trade_date'].min()))
    end_date   = str(int(fm.df['trade_date'].max()))
    return file, fm.code, start_date, end_date


def update_series_config(results: list, config_path: str) -> None:
    """
    将各股票的 trade_date 时间范围写入 data/series/_date_range.csv。

    config 格式（CSV）：
        ts_code, start_date, end_date
    已有记录按 ts_code 更新；新股票追加；结果按 ts_code 排序。

    Parameters
    ----------
    results     : list of (ts_code, start_date, end_date)
    config_path : str
    """
    if not results:
        return

    new_df = pd.DataFrame(results, columns=['ts_code', 'start_date', 'end_date'])

    if os.path.exists(config_path):
        existing = pd.read_csv(config_path, dtype=str)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset='ts_code', keep='last', inplace=True)
    else:
        combined = new_df

    combined.sort_values('ts_code', inplace=True)
    combined.to_csv(config_path, index=False)


# ================== 4. 主程序 ==================

if __name__ == '__main__':
    # 路径相对于本文件所在目录的上层（项目根），无论从哪里执行都正确
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    series_path = os.path.join(_base, "data", "series") + os.sep

    # 支持 Parquet 和 CSV 格式的财务数据
    financial_parquet = os.path.join(_base, "data", "financial.parquet")
    financial_csv = os.path.join(_base, "data", "financial.csv")

    if os.path.exists(financial_parquet):
        fin_df = pd.read_parquet(financial_parquet)
        print(f"从 Parquet 读取财务数据: {financial_parquet}")
    elif os.path.exists(financial_csv):
        fin_df = pd.read_csv(financial_csv)
        print(f"从 CSV 读取财务数据: {financial_csv}")
    else:
        raise FileNotFoundError("找不到财务数据文件 financial.parquet 或 financial.csv")

    print("预处理财务数据...")
    fin_features = build_financial_features(fin_df)
    print(f"完成，共 {len(fin_features)} 只股票。")

    # 优先匹配 Parquet 文件，其次 CSV 文件
    parquet_files = glob.glob(series_path + "[0-9]*.parquet")
    csv_files = glob.glob(series_path + "[0-9]*.csv")

    # 合并文件列表，Parquet 优先
    files = parquet_files if parquet_files else csv_files
    total = len(files)

    if total == 0:
        print(f"警告: 在 {series_path} 中未找到股票数据文件")
    else:
        file_format = "Parquet" if parquet_files else "CSV"
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"启动 {n_workers} 个进程并行计算因子（共 {total} 只股票，{file_format} 格式）...")

        completed = 0
        date_range_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one_stock, (f, fin_features)): f for f in files}
            for future in as_completed(futures):
                completed += 1
                print(f"\r完成 {completed}/{total}", end='', flush=True)
                try:
                    _, ts_code, start_date, end_date = future.result()
                    date_range_results.append((ts_code, start_date, end_date))
                except Exception as e:
                    print(f"\n  ✗ {futures[future]}: {e}")

        config_path = os.path.join(series_path, '_date_range.csv')
        update_series_config(date_range_results, config_path)
        print(f"\n因子计算完成，date range config 已更新 → {config_path}")
