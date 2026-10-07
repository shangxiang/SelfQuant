import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import ElasticNetCV
from typing import Optional

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class ElasticNetConfig:
    """弹性网络策略的超参数配置。"""

    def __init__(self):
        # ---- 数据路径（相对于项目根目录）----
        self.data_dir = 'data/section/'                          # 截面数据目录
        self.calendar_file = 'data/raw/trade_cal.csv'           # 交易日历
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'  # 股票列表

        # ---- 列名约定 ----
        self.stock_col = 'ts_code'   # 股票代码列
        # 预测目标列（未来 N 日收益率）
        # 'label'          = 原始总收益，包含行业/市值 beta → beta + alpha 总体最优
        # 'label_standard' = 行业市值中性化后的纯 alpha（原默认）
        # 目标是 top-K 总收益时用 'label'：中性化会主动丢掉行业轮动这块真金白银
        self.label_col = 'label'
        # label 所代表的持有期天数，用于多空收益的非重叠采样和夏普年化
        # 与 label_col 对应：'label'=5, 'label_10'=10, 'label_25'=25
        self.label_period: int = 5

        # label 在截面日 d 之后多少个交易日才能确认（即 label 最后用到的未来价格偏移量）
        # 旧 label（T→T+5）：5 日后可知，lookahead=5
        # 新 label（T+1→T+6）：6 日后可知，lookahead=6
        # _build_train_data 用此值截断训练集末端，防止未来函数
        self.label_lookahead: int = 6

        # ---- 参与建模的因子列 ----
        # 参与建模的因子列（<因子>_standard 列名，线性模型用）
        # 共 224 个，按因子族分组；想裁剪直接注释掉对应行即可
        self.factor_cols = [
            # ---- 技术 / 量价 / 资金（90）----
            'total_mv_standard',
            'macd_divergence',
            'macd_air_refuel',
            'circ_mv_standard',
            'close_x_standard',
            'close_y_standard',
            'dif_standard',
            'dea_standard',
            'macd_standard',
            'K_standard',
            'D_standard',
            'J_standard',
            'rsi_standard',
            'cci_standard',
            'close_to_vwap_ratio_standard',
            'pe_standard',
            'pe_ttm_standard',
            'pb_standard',
            'ps_standard',
            'ps_ttm_standard',
            'dv_ratio_standard',
            'dv_ttm_standard',
            'total_share_standard',
            'float_share_standard',
            'free_share_standard',
            'amount_x_standard',
            'turnover_rate_x_standard',
            'turnover_rate_f_standard',
            'volume_ratio_standard',
            'buy_sm_vol_standard',
            'buy_sm_amount_standard',
            'sell_sm_vol_standard',
            'sell_sm_amount_standard',
            'buy_md_vol_standard',
            'buy_md_amount_standard',
            'sell_md_vol_standard',
            'sell_md_amount_standard',
            'buy_lg_vol_standard',
            'buy_lg_amount_standard',
            'sell_lg_vol_standard',
            'sell_lg_amount_standard',
            'buy_elg_vol_standard',
            'buy_elg_amount_standard',
            'sell_elg_vol_standard',
            'sell_elg_amount_standard',
            'net_mf_vol_standard',
            'net_mf_amount_standard',
            'positive_flow_standard',
            'negative_flow_standard',
            'mfi_standard',
            'raw_force_index_standard',
            'force_index_smoothed_standard',
            'force_index_standard',
            'vwap_standard',
            'mtm_margin_balance_change_standard',
            'big_order_ratio_standard',
            'debt_ratio_standard',
            'adx_standard',
            'K_chg_5d_standard',
            'K_chg_10d_standard',
            'D_chg_5d_standard',
            'D_chg_10d_standard',
            'J_chg_5d_standard',
            'J_chg_10d_standard',
            'rsi_chg_5d_standard',
            'rsi_chg_10d_standard',
            'macd_chg_5d_standard',
            'macd_chg_10d_standard',
            'adx_chg_5d_standard',
            'adx_chg_10d_standard',
            'turnover_rate_x_chg_5d_standard',
            'turnover_rate_x_chg_10d_standard',
            'rzye_chg_5d_standard',
            'rzye_chg_10d_standard',
            'amount_y_standard',
            'turnover_rate_y_standard',
            'l_sell_standard',
            'l_buy_standard',
            'l_amount_standard',
            'net_amount_standard',
            'net_rate_standard',
            'amount_rate_standard',
            'rzye_standard',
            'rqye_standard',
            'rzmre_standard',
            'rqyl_standard',
            'rzche_standard',
            'rqchl_standard',
            'rqmcl_standard',
            'rzrqye_standard',
            # ---- Alpha101（31）----
            'alpha101_1_standard',
            'alpha101_2_standard',
            'alpha101_3_standard',
            'alpha101_4_standard',
            'alpha101_5_standard',
            'alpha101_6_standard',
            'alpha101_7_standard',
            'alpha101_8_standard',
            'alpha101_9_standard',
            'alpha101_10_standard',
            'alpha101_11_standard',
            'alpha101_12_standard',
            'alpha101_13_standard',
            'alpha101_14_standard',
            'alpha101_15_standard',
            'alpha101_16_standard',
            'alpha101_17_standard',
            'alpha101_18_standard',
            'alpha101_19_standard',
            'alpha101_20_standard',
            'alpha101_22_standard',
            'alpha101_23_standard',
            'alpha101_25_standard',
            'alpha101_33_standard',
            'alpha101_34_standard',
            'alpha101_41_standard',
            'alpha101_52_standard',
            'alpha101_53_standard',
            'alpha101_54_standard',
            'alpha101_57_standard',
            'alpha101_101_standard',
            # ---- 动量（21）----
            'momentum_12_1_standard',
            'ret_10d_standard',
            'ret_20d_standard',
            'ret_60d_standard',
            'momentum_12_1_chg_5d_standard',
            'momentum_12_1_chg_10d_standard',
            'return_5d_standard',
            'return_21d_standard',
            'return_42d_standard',
            'return_63d_standard',
            'return_126d_standard',
            'return_252d_standard',
            'ma_20d_standard',
            'price_position_ir_60d_standard',
            'rsrs_standard',
            'days_down_up_standard',
            'return_std_21d_standard',
            'return_std_42d_standard',
            'return_std_63d_standard',
            'return_std_126d_standard',
            'return_std_252d_standard',
            # ---- 风险/波动（14）----
            'volatility_20d_standard',
            'high_low_spread_standard',
            'volatility_20d_chg_5d_standard',
            'volatility_20d_chg_10d_standard',
            'sharpe_60d_standard',
            'sharpe_750d_standard',
            'adjusted_sharpe_750d_standard',
            'high_low_21d_standard',
            'high_low_42d_standard',
            'high_low_63d_standard',
            'high_low_126d_standard',
            'high_low_252d_standard',
            'days_beyond_upper_lower_21d_standard',
            'log_price_standard',
            # ---- 流动性（30）----
            'amount_ma_20d_standard',
            'turnover_ma_20d_standard',
            'sum_abs_rtn_amount_20d_standard',
            'avg_turnover_5d_standard',
            'avg_turnover_10d_standard',
            'avg_turnover_20d_standard',
            'std_turnover_21d_standard',
            'std_turnover_42d_standard',
            'std_turnover_63d_standard',
            'std_turnover_126d_standard',
            'std_turnover_252d_standard',
            'avg_turnover_21d_standard',
            'avg_turnover_42d_standard',
            'avg_turnover_63d_standard',
            'avg_turnover_126d_standard',
            'avg_turnover_252d_standard',
            'bias_turn_21d_252d_standard',
            'bias_std_turn_21d_252d_standard',
            'bias_turn_42d_252d_standard',
            'bias_turn_63d_252d_standard',
            'bias_turn_126d_252d_standard',
            'bias_turn_21d_504d_standard',
            'bias_std_turn_21d_504d_standard',
            'bias_turn_42d_504d_standard',
            'bias_std_turn_42d_504d_standard',
            'bias_turn_63d_504d_standard',
            'bias_std_turn_63d_504d_standard',
            'bias_turn_126d_504d_standard',
            'bias_std_turn_126d_504d_standard',
            'turnover_ma_20d_120d_standard',
            # ---- 质量/成长（9）----
            'debt_to_assets_standard',
            'gross_margin_standard',
            'roe_ttm_standard',
            'revenue_growth_yoy_standard',
            'profit_growth_yoy_standard',
            'accruals_standard',
            'asset_growth_yoy_standard',
            'roe_growth_yoy_standard',
            'gross_margin_growth_standard',
            # ---- 价值/风格（11）----
            'size_factor_standard',
            'smb_squared_standard',
            'value_factor_standard',
            'cma_factor_standard',
            'smb_mom_standard',
            'smb_squared_mom_standard',
            'hml_rmw_standard',
            'smb_hml_standard',
            'vol_mom_standard',
            'size_standard',
            'float_size_standard',
            # ---- 反转（6）----
            'reversal_5d_standard',
            'dist_52w_high_standard',
            'reversal_5d_chg_5d_standard',
            'reversal_5d_chg_10d_standard',
            'small_cap_reversal_21d_standard',
            'price_dist_standard',
            # ---- 微观结构/形态（12）----
            'close_ma20_ratio_standard',
            'up_day_ratio_20_standard',
            'vol_price_corr_20d_standard',
            'turnover_amplitude_ratio_standard',
            'long_shadow_freq_standard',
            'doji_freq_standard',
            'intraday_drawdown_standard',
            'gap_vs_range_ratio_standard',
            'poly_close_a1_standard',
            'poly_close_a2_standard',
            'poly_vol_a1_standard',
            'poly_vol_a2_standard',
        ]

        # ---- 模型超参数 ----
        # 滚动训练窗口，单位：交易日
        # 越大则训练数据越多但对近期市场反应越慢
        self.window = 80

        # ElasticNetCV 的 L1 比例搜索网格
        # 0 = 纯 Ridge（L2），1 = 纯 Lasso（L1），中间值为混合
        # 特征从 80 涨到 237 后，CV 成本 = len(l1_ratio_grid) × n_alphas × cv，
        # 原 6×100×5 = 3000 次坐标下降，单次 fit 超过 5 分钟。收到 3×50×3 = 450
        # 后降到约 1 分钟；L1 比例只需覆盖「偏 L2 / 混合 / 偏 L1」三档即可。
        self.l1_ratio_grid = [0.1, 0.5, 0.9]

        # 交叉验证折数，用于选择最优超参数
        self.cv = 3

        # alpha 搜索路径长度（ElasticNetCV 默认 100）
        self.n_alphas = 50

        # 是否对每日更新的因子权重做指数移动平滑（EMA）
        # 可减少模型在相邻交易日之间的权重跳变，使仓位更稳定
        self.smooth_weights = False

        # EMA 平滑系数：新权重 = alpha * 新值 + (1-alpha) * 旧值
        # alpha 越小，历史权重影响越大，变化越平滑
        self.smooth_alpha = 0.2

        # ---- 行业 / 市场 beta 特征 ----
        # 由 strategy/market_features.py 在 DataLoader 层动态附加，无需重跑标准化。
        # 目标若是 top-K 总收益，行业轮动是真实可赚的钱，必须显式喂给模型。

        # ---- 因子集 ----
        # 'curated' = 只用上面手工挑选的列表
        # 'all'     = 手工列表 ∪ 数据中发现的全部因子（Alpha101 / 动量 / 风险 /
        #             流动性 / 质量成长 / 价值等此前未被启用的族会一并进来）
        # 线性模型默认用 curated：ElasticNetCV 要对 6 个 l1_ratio × 100 个 alpha
        # × 5 折做坐标下降，237 个特征下单次 fit 会超过 5 分钟（树模型则无此问题，
        # 211 个特征只要 9~20 秒）。想试全量可改成 'all'，但建议同时调小
        # l1_ratio_grid 和 cv。
        self.factor_set: str = 'curated'
        # 'raw' = 原始因子列名（树模型）；'standard' = <因子>_standard（线性模型）
        self.factor_style: str = 'standard'
        self.add_market_features: bool = True
        self.market_feature_zscore: bool = True   # 截面 z-score，线性模型必需
        self.factor_cols += [
            'ind_ret_1d', 'ind_ret_5d', 'ind_ret_20d',
            'ind_up_5d', 'ind_up_20d',
            'ind_flow_5d', 'ind_flow_20d',
            'ind_mom_rank', 'ind_pe_z',
            'mkt_ret_5d', 'mkt_ret_20d', 'mkt_up_1d', 'mkt_up_5d',
        ]

        # ---- 训练样本抽样 ----
        # label_period=5 时相邻交易日的 label 共享未来价格，时间上不独立。
        # sample_step=5 可让训练样本近似无重叠（配合更大的 window 使用）。
        self.sample_step: int = 1
        # ElasticNet 用的是 _standard（已做行业+市值中性化）列，属于纯 alpha；
        # 再补上未被中性化的规模类列，配合上面的行业/市场特征，
        # 让线性模型能同时学到 alpha 与 beta 两部分。
        self.factor_cols += ['circ_mv_standard']


        # 去重（防御）：factor_cols 若出现重名，pandas 的 df[cols] 会返回多列，
        # 触发 "Columns must be same length as key"
        _seen = set()
        self.factor_cols = [c for c in self.factor_cols
                            if not (c in _seen or _seen.add(c))]

        # 数据中并不存在的列（会被填成全 0，对线性模型是纯噪声）
        self.factor_cols = [c for c in self.factor_cols if c != 'lhb_strength_5d_standard']

class ElasticNetStrategy(BaseStrategy):
    """
    基于弹性网络（ElasticNet）的滚动窗口因子选股策略。

    每个交易日调用 fit() 重新训练模型，调用 generate_signals() 对全量股票打分。
    二者由引擎显式调用，不在内部互相触发，以支持信号生成与成交时间的错位。
    """

    def __init__(self, config: ElasticNetConfig, data_loader):
        """
        Parameters
        ----------
        config      : ElasticNetConfig  策略配置
        data_loader : DataLoader        截面数据加载器（依赖注入）
        """
        self.cfg = config
        self.loader = data_loader
        # 当前训练得到的因子权重向量，shape = (len(factor_cols),)
        # None 表示尚未训练，generate_signals() 会据此返回 None
        self._weights: Optional[np.ndarray] = None

    def reset(self) -> None:
        """清空已训练权重，确保每次独立回测从零开始。"""
        self._weights = None

    def _build_train_data(self, today_idx: int):
        """
        构建滚动窗口训练集。

        滚动窗口为 [today_idx - window + 1, today_idx]，但跳过最后 5 个交易日：
        label 定义为未来 5 日收益，最近 5 天的 label 尚未实现，不能用于训练。

        Parameters
        ----------
        today_idx : int  当日在全量交易日历中的索引

        Returns
        -------
        X : np.ndarray | None  因子矩阵，shape = (n_samples, n_factors)
        y : np.ndarray | None  标签向量，shape = (n_samples,)
        """
        all_dates = self.loader.get_trading_dates()
        X_list, y_list = [], []

        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            # 跳过最近 label_lookahead 天：label = (close[d+6]-close[d+1])/close[d+1]，
            # 需要 d+6 的价格才能确认，故 d+lookahead > today 的样本不能用于训练
            if j + self.cfg.label_lookahead > today_idx:
                continue
            if self.cfg.sample_step > 1 and (today_idx - j) % self.cfg.sample_step != 0:
                continue

            t_date = all_dates[j]
            df = self.loader.get_data(t_date)
            if df is None:
                continue

            if self.cfg.stock_col not in df.columns or self.cfg.label_col not in df.columns:
                continue

            sub = df[[self.cfg.stock_col, self.cfg.label_col]].copy()
            sub = sub.dropna(subset=[self.cfg.label_col])
            if sub.empty:
                continue

            # 因子列缺失时填 0（历史数据不足导致的缺列，如动量因子在数据起始期）
            for fc in self.cfg.factor_cols:
                sub[fc] = df[fc].astype(float) if fc in df.columns else 0.0
            sub[self.cfg.factor_cols] = sub[self.cfg.factor_cols].fillna(0.0)
            X_list.append(sub[self.cfg.factor_cols].values)
            y_list.append(sub[self.cfg.label_col].astype(float).values)

        if not X_list:
            return None, None
        return np.vstack(X_list), np.concatenate(y_list)

    def fit(self, date_str: str) -> bool:
        """
        以 date_str 为基准日，用滚动窗口内的历史数据训练 ElasticNetCV。
        训练成功后更新 self._weights（可选做 EMA 平滑）。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期

        Returns
        -------
        bool  True = 训练成功，False = 非交易日或数据样本不足（< 30 条）
        """
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            print("今日并非交易日")
            return False

        today_idx = all_dates.index(date_str)
        X, y = self._build_train_data(today_idx)
        if X is None or len(y) < 30:
            # 样本量过少时，交叉验证结果不可靠，跳过该日
            return False

        X = X.astype(np.float64)
        y = y.astype(np.float64)
        # 切分必须打乱：训练集是按日期堆叠的，若用未打乱的 KFold，
        # 第 1 折会变成「用后 80% 的日期训练、前 20% 的日期验证」，
        # 等于拿未来预测过去 —— CV 分数极差，ElasticNetCV 会直接选到最大正则，
        # 把所有系数压成 0（实测：改之前 94 个权重全为 0，改之后 R2 约 3%）。
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=self.cfg.cv, shuffle=True, random_state=42)

        # n_jobs=1：单线程跑 CV，避免多进程时 Gram 矩阵浮点精度校验失败
        # precompute='auto'（默认）：保留 Gram 矩阵缓存，坐标下降 O(p) 而非 O(n*p)
        model = ElasticNetCV(
            l1_ratio=self.cfg.l1_ratio_grid,
            cv=cv,
            n_alphas=self.cfg.n_alphas,
            max_iter=5000,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X, y)
        raw_weights = model.coef_  # shape = (n_factors,)

        if self.cfg.smooth_weights and self._weights is not None:
            # EMA 平滑：新权重 = alpha * 新值 + (1-alpha) * 历史权重
            # 避免相邻交易日因子权重剧烈跳变导致换手率过高
            self._weights = (self.cfg.smooth_alpha * raw_weights +
                             (1 - self.cfg.smooth_alpha) * self._weights)
        else:
            self._weights = raw_weights
        return True

    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        用当前因子权重对 date_str 截面内所有股票打分。

        打分公式：score = X @ weights（线性因子模型）
        不使用 label 列，可直接用于实盘信号生成。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期

        Returns
        -------
        DataFrame[ts_code, score] 按 score 降序，或 None（权重未训练/数据缺失）
        """
        if self._weights is None:
            return None

        df = self.loader.get_data(date_str)
        if df is None:
            return None

        if self.cfg.stock_col not in df.columns:
            return None

        df_valid = df[[self.cfg.stock_col]].copy()
        # 因子列缺失时填 0（历史数据不足导致的缺列，如动量因子在数据起始期）
        for fc in self.cfg.factor_cols:
            df_valid[fc] = df[fc].astype(float) if fc in df.columns else 0.0
        df_valid[self.cfg.factor_cols] = df_valid[self.cfg.factor_cols].fillna(0.0)

        # 矩阵乘法：每只股票的各因子值与权重向量做内积，得到综合打分
        scores = df_valid[self.cfg.factor_cols].values.dot(self._weights)
        result = df_valid[[self.cfg.stock_col]].copy()
        result['score'] = scores
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        纯信号质量评估：Rank IC + 五分组多空收益。

        每日调用 fit() + generate_signals()，再与 label 拼接计算相关性。
        注意：此处 label 仅用于评估，不参与训练，不存在未来函数问题。

        Parameters
        ----------
        start_date : str  评估开始日期 YYYYMMDD
        end_date   : str  评估结束日期 YYYYMMDD

        Returns
        -------
        dict 包含：
            ic_df       : DataFrame[date, ic]   每日 Rank IC
            ls_df       : DataFrame[date, long_ret, short_ret, spread]  多空收益
            weights_df  : DataFrame[date, factor...]  每日因子权重
        """
        self.reset()
        all_dates = self.loader.get_trading_dates()
        dates = [d for d in all_dates if start_date <= d <= end_date]

        ic_series = []           # [(date, rank_ic), ...]
        long_short_returns = []  # [(date, long_ret, short_ret, spread), ...]
        weights_history = {}     # {date: {factor: weight}}

        for today in dates:
            ok = self.fit(today)
            if not ok:
                continue
            weights_history[today] = dict(zip(self.cfg.factor_cols, self._weights))

            signals = self.generate_signals(today)
            if signals is None or signals.empty:
                continue

            # 取当日 label（未来5日收益）用于 IC 计算
            df_today = self.loader.get_data(today)
            if df_today is None or self.cfg.label_col not in df_today.columns:
                continue
            signals = signals.merge(
                df_today[[self.cfg.stock_col, self.cfg.label_col]],
                on=self.cfg.stock_col, how='left'
            )
            signals = signals.dropna(subset=[self.cfg.label_col])
            if signals.empty:
                continue

            # Rank IC：打分排名与收益排名的 Spearman 相关系数
            ic = signals['score'].corr(signals[self.cfg.label_col], method='spearman')
            ic_series.append((today, ic))

            # 五分组：按打分分为 Q1（最低）到 Q5（最高），计算各组平均收益
            signals['rank'] = signals['score'].rank(pct=True)
            signals['group'] = pd.cut(
                signals['rank'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            )
            group_ret = signals.groupby('group')[self.cfg.label_col].mean()
            long_ret = group_ret.get('Q5', np.nan)   # 最高分组（多头）
            short_ret = group_ret.get('Q1', np.nan)  # 最低分组（空头）
            long_short_returns.append((today, long_ret, short_ret, long_ret - short_ret))

        ic_df = pd.DataFrame(ic_series, columns=['date', 'ic']).set_index('date')
        ls_df = pd.DataFrame(
            long_short_returns, columns=['date', 'long_ret', 'short_ret', 'spread']
        ).set_index('date')
        weights_df = pd.DataFrame(weights_history).T.sort_index()

        print("===== Rank IC 统计 =====")
        print(f"IC 均值: {ic_df['ic'].mean():.4f}")
        print(f"IC 标准差: {ic_df['ic'].std():.4f}")
        print(f"IR: {ic_df['ic'].mean() / ic_df['ic'].std():.4f}")
        print(f"IC>0 比例: {(ic_df['ic'] > 0).mean():.2%}")

        # 多空收益统计：label 是 N 日收益，相邻行高度重叠（重叠 N-1 天）
        # 直接对全量 ls_df 求累计/夏普会产生约 N 倍的虚高
        # 修复：用非重叠采样（每 label_period 行取一次），再以 N 日为基准年化
        n = self.cfg.label_period
        ls_nonoverlap = ls_df.iloc[::n]
        ann_obs = 252 / n                                  # 一年内独立观测次数
        cum_spread = (1 + ls_nonoverlap['spread']).prod() - 1
        sp = ls_nonoverlap['spread']
        sharpe = (sp.mean() / sp.std()) * np.sqrt(ann_obs) if sp.std() != 0 else 0.0

        print(f"\n===== 多空收益统计（非重叠采样，持有期={n}d）=====")
        print(ls_nonoverlap.describe())
        print(f"多空累计收益: {cum_spread:.4%}")
        print(f"年化夏普:     {sharpe:.4f}")
        print("\n===== 因子权重均值 =====")
        print(weights_df.mean().sort_values(ascending=False))

        result = {'ic_df': ic_df, 'ls_df': ls_df, 'weights_df': weights_df}
        self.report_dump(result, label_period=n)
        return result
