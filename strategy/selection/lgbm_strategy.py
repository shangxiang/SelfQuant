import numpy as np
import pandas as pd
import warnings
from typing import Optional

import lightgbm as lgb

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class LGBMConfig:
    """LightGBM 截面选股策略超参数配置。"""

    def __init__(self):
        # ---- 数据路径 ----
        self.data_dir        = 'data/section/'
        self.calendar_file   = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'

        # ---- 列名 ----
        self.stock_col  = 'ts_code'
        self.label_col  = 'label'  # 中性化标准化后的 5 日收益；训练时会截面 rank 归一化
        self.label_period: int = 5
        self.label_lookahead: int = 6   # label 需要 T+6 价格才能确认
        # self.label_period: int = 10
        # self.label_lookahead: int = 11   # label 需要 T+6 价格才能确认

        # ---- 因子列（树模型不需要标准化，直接用原始值）----
        # 参与建模的因子列（原始列名，树模型用）
        # 共 198 个，按因子族分组；想裁剪直接注释掉对应行即可
        self.factor_cols = [
            # ---- 技术 / 量价 / 资金（32）----
            'pe_ttm',
            'pb',
            'dv_ttm',
            'net_mf_amount',
            'turnover_rate_x',
            'volume_ratio',
            'total_mv',
            'rzye',
            'debt_ratio',
            'dif',
            'dea',
            'macd',
            'K',
            'D',
            'J',
            'positive_flow',
            'negative_flow',
            'mfi',
            'rsi',
            'cci',
            'raw_force_index',
            'force_index_smoothed',
            'force_index',
            'vwap',
            'close_to_vwap_ratio',
            'mtm_margin_balance_change',
            'macd_air_refuel',
            'macd_divergence',
            'big_order_ratio',
            # 'lhb_strength_5d',
            'vol_breakout',
            'adx',
            # ---- Alpha101（31）----
            'alpha101_1',
            'alpha101_2',
            'alpha101_3',
            'alpha101_4',
            'alpha101_5',
            'alpha101_6',
            'alpha101_7',
            'alpha101_8',
            'alpha101_9',
            'alpha101_10',
            'alpha101_11',
            'alpha101_12',
            'alpha101_13',
            'alpha101_14',
            'alpha101_15',
            'alpha101_16',
            'alpha101_17',
            'alpha101_18',
            'alpha101_19',
            'alpha101_20',
            'alpha101_22',
            'alpha101_23',
            'alpha101_25',
            'alpha101_33',
            'alpha101_34',
            'alpha101_41',
            'alpha101_52',
            'alpha101_53',
            'alpha101_54',
            'alpha101_57',
            'alpha101_101',
            # ---- 动量（19）----
            'momentum_12_1',
            'ret_10d',
            'ret_20d',
            'ret_60d',
            'return_5d',
            'return_21d',
            'return_42d',
            'return_63d',
            'return_126d',
            'return_252d',
            'ma_20d',
            'price_position_ir_60d',
            'rsrs',
            'days_down_up',
            'return_std_21d',
            'return_std_42d',
            'return_std_63d',
            'return_std_126d',
            'return_std_252d',
            # ---- 风险/波动（12）----
            'volatility_20d',
            'high_low_spread',
            'sharpe_60d',
            'sharpe_750d',
            'adjusted_sharpe_750d',
            'high_low_21d',
            'high_low_42d',
            'high_low_63d',
            'high_low_126d',
            'high_low_252d',
            'days_beyond_upper_lower_21d',
            'log_price',
            # ---- 流动性（30）----
            'avg_turnover_5d',
            'avg_turnover_10d',
            'avg_turnover_20d',
            'amount_ma_20d',
            'turnover_ma_20d',
            'sum_abs_rtn_amount_20d',
            'std_turnover_21d',
            'avg_turnover_21d',
            'std_turnover_42d',
            'avg_turnover_42d',
            'std_turnover_63d',
            'avg_turnover_63d',
            'std_turnover_126d',
            'avg_turnover_126d',
            'std_turnover_252d',
            'avg_turnover_252d',
            'bias_turn_21d_252d',
            'bias_std_turn_21d_252d',
            'bias_turn_42d_252d',
            'bias_turn_63d_252d',
            'bias_turn_126d_252d',
            'bias_turn_21d_504d',
            'bias_std_turn_21d_504d',
            'bias_turn_42d_504d',
            'bias_std_turn_42d_504d',
            'bias_turn_63d_504d',
            'bias_std_turn_63d_504d',
            'bias_turn_126d_504d',
            'bias_std_turn_126d_504d',
            'turnover_ma_20d_120d',
            # ---- 质量/成长（20）----
            'gross_margin',
            'roe_ttm',
            'revenue_growth_yoy',
            'profit_growth_yoy',
            'accruals',
            'asset_growth_yoy',
            # 'roa_ttm',
            # 'net_margin',
            'debt_to_assets',
            # 'current_ratio',
            # 'quick_ratio',
            # 'cash_flow_to_debt',
            # 'earnings_quality',
            'roe_growth_yoy',
            # 'eps_growth_yoy',
            # 'revenue_growth_qoq',
            # 'profit_growth_qoq',
            'gross_margin_growth',
            # 'net_margin_growth',
            # 'ocf_growth_yoy',
            # ---- 价值/风格（16）----
            'size_factor',
            'value_factor',
            'cma_factor',
            'smb_squared',
            'smb_mom',
            'smb_squared_mom',
            'hml_rmw',
            'smb_hml',
            'vol_mom',
            'size',
            'float_size',
            # 'earnings_to_price',
            # 'book_to_market',
            # 'ocf_to_market',
            # 'fcf_to_market',
            # 'sales_to_market',
            # ---- 反转（4）----
            'reversal_5d',
            'dist_52w_high',
            'small_cap_reversal_21d',
            'price_dist',
            # ---- 微观结构/形态（12）----
            'close_ma20_ratio',
            'up_day_ratio_20',
            'vol_price_corr_20d',
            'turnover_amplitude_ratio',
            'long_shadow_freq',
            'doji_freq',
            'intraday_drawdown',
            'gap_vs_range_ratio',
            'poly_close_a1',
            'poly_close_a2',
            'poly_vol_a1',
            'poly_vol_a2',
            # ---- 时序差分（22）----
            'K_chg_5d',
            'K_chg_10d',
            'D_chg_5d',
            'D_chg_10d',
            'J_chg_5d',
            'J_chg_10d',
            'rsi_chg_5d',
            'rsi_chg_10d',
            'macd_chg_5d',
            'macd_chg_10d',
            'adx_chg_5d',
            'adx_chg_10d',
            'volatility_20d_chg_5d',
            'volatility_20d_chg_10d',
            'turnover_rate_x_chg_5d',
            'turnover_rate_x_chg_10d',
            'reversal_5d_chg_5d',
            'reversal_5d_chg_10d',
            'momentum_12_1_chg_5d',
            'momentum_12_1_chg_10d',
            'rzye_chg_5d',
            'rzye_chg_10d',
        ]

        # ---- 市值过滤（单位：万元；None 表示不过滤）----
        # 30亿 = 300_000 万元
        self.min_mv: Optional[int] = 0
        # ---- ST 过滤（True = 过滤掉 ST 股，默认开启）----
        self.filter_st: bool = True

        # ---- 滚动训练窗口（交易日数）----
        # 滚动训练窗口（交易日数）
        # 实测（5 个基准日 × 3 种窗口，211 特征）：window=40 的 RankIC 只有 0.025，
        # 80~150 天能到 0.038~0.052。label_period=5 时 40 天窗口在「时间维度」上
        # 只有 8 个独立样本，太薄。取 120 兼顾样本量与时效性。
        self.window: int = 60

        # ---- 时间衰减权重（半衰期，单位：交易日；None 表示不使用）----
        # 权重公式：w = 2^(-(today_idx - j) / weight_halflife)
        # 60 表示距今 60 个交易日的样本权重为最新样本的一半
        self.weight_halflife: Optional[int] = 20

        # ---- LightGBM 超参数 ----
        # self.lgbm_params = {
        #     'objective':        'regression',
        #     'metric':           'rmse',
        #     'n_estimators':     100,
        #     'learning_rate':    0.05,
        #     'max_depth':        4,
        #     'num_leaves':       15,       # < 2^max_depth，控制过拟合
        #     'min_child_samples': 100,      # 每叶最少样本，防止截面过拟合
        #     'subsample':        0.8,      # 行采样
        #     'colsample_bytree': 0.6,      # 特征采样
        #     'reg_alpha':        0.1,
        #     'reg_lambda':       5.0,
        #     'n_jobs':           -1,
        #     'verbose':          -1,
        #     'random_state':     42,
        # }
        # 以下取值由 _tune_hp2.py 的网格诊断确定（211 特征 / 5 个基准日平均）：
        #   较原值（70/0.05/3/7/0.6/5.0）RankIC 0.0385 → 0.0478，ICIR 0.480 → 0.519
        # 特征数从 55 涨到 211 后，colsample 必须下调：0.6 时每棵树看 127 个特征，
        # 而其中换手率族/动量族高度同源，树之间会趋同、集成多样性尽失。
        self.lgbm_params = {
            'objective':         'regression',
            'metric':            'rmse',
            'n_estimators':      300,
            'learning_rate':     0.03,   # 树多了就相应降 lr，保持总学习量
            'max_depth':         4,      # 原 3 层配 211 特征偏欠拟合
            'num_leaves':        15,
            'min_child_samples': 300,
            'subsample':         0.8,
            'colsample_bytree':  0.3,    # 原 0.6；每棵树约看 42 个特征
            'reg_alpha':         0.1,
            'reg_lambda':        1.0,    # 原 5.0；实测 1.0 的 top10 收益更高
            'n_jobs':            -1,
            'verbose':           -1,
            'random_state':      42,
        }

        # 分位值预测
        # self.lgbm_params = {
        #     'objective':         'quantile',
        #     'alpha':             0.99,        # 预测第 90 分位：找 top-10% 的股票
        #     'metric':            'quantile', # 评估指标与目标函数保持一致
        #     'n_estimators':      100,        # 分位数梯度是阶跃函数，比 RMSE 更噪，需要更多树
        #     'learning_rate':     0.03,       # 同理，适当降低学习率稳定收敛
        #     'max_depth':         3,
        #     'num_leaves':        7,
        #     'min_child_samples': 300,
        #     'subsample':         0.8,
        #     'colsample_bytree':  0.6,
        #     'reg_alpha':         0.1,
        #     'reg_lambda':        5.0,
        #     'n_jobs':            -1,
        #     'verbose':           -1,
        #     'random_state':      42,
        # }

        # 是否使用早停（需要验证集）；False 则跑满 n_estimators
        self.early_stopping: bool = False
        self.early_stopping_rounds: int = 20

        # ---- Top-k 样本加权 ----
        # 对截面内 rank 前 top_weight_pct 的股票额外放大损失权重
        # 迫使模型优先把头部预测准，而非整体 RMSE 最小化
        # None 表示不使用（与原始等权等价）
        # 只买 top5~top10，所以加权对象必须是真正的头部（3000 只里的 0.5%~1%），
        # 而不是 top 10%（300 只）—— 后者会让梯度浪费在根本不会买的股票上
        self.top_weight_pct:    float = 0.01  # top 1% 的股票加权
        self.top_weight_factor: float = 3.0   # 额外权重倍数（1.0=不加权，3.0=3倍权重）

        # ---- 行业 / 市场 beta 特征 ----
        # 由 strategy/market_features.py 在 DataLoader 层动态附加，无需重跑标准化。
        # 目标若是 top-K 总收益，行业轮动是真实可赚的钱，必须显式喂给模型。

        # ---- 因子集 ----
        # 'curated' = 只用上面手工挑选的列表
        # 'all'     = 手工列表 ∪ 数据中发现的全部因子（Alpha101 / 动量 / 风险 /
        #             流动性 / 质量成长 / 价值等此前未被启用的族会一并进来）
        self.factor_set: str = 'curated'
        # 'raw' = 原始因子列名（树模型）；'standard' = <因子>_standard（线性模型）
        self.factor_style: str = 'raw'
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


        # 去重（防御）：factor_cols 若出现重名，pandas 的 df[cols] 会返回多列，
        # 触发 "Columns must be same length as key"
        _seen = set()
        self.factor_cols = [c for c in self.factor_cols
                            if not (c in _seen or _seen.add(c))]


class LGBMStrategy(BaseStrategy):
    """
    基于 LightGBM 的滚动窗口截面选股策略。

    与 ElasticNetStrategy 的核心区别：
      1. 树模型不需要特征标准化，直接使用原始因子值。
      2. 训练标签为截面 rank 百分位（→ [-0.5, 0.5]），
         消除极端收益的影响，提高跨期稳定性。
      3. 打分为模型预测的 rank 值，越高表示预期涨幅越大。
      4. feature_importances_ 替代 coef_ 作为因子贡献分析。

    接口与 ElasticNetStrategy 完全兼容，可无缝替换引擎中的策略对象。
    """

    def __init__(self, config: LGBMConfig, data_loader):
        self.cfg    = config
        self.loader = data_loader
        self._model: Optional[lgb.LGBMRegressor] = None
        # 滚动平均特征重要度，用于 simple_backtest 的权重输出
        self._importances: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._model       = None
        self._importances = None

    # ------------------------------------------------------------------
    # 内部：构建训练集
    # ------------------------------------------------------------------
    def _build_train_data(self, today_idx: int):
        """
        构建截面 rank 归一化的训练集。

        label 转换：每个截面日内，对原始涨幅做 rank(pct=True) - 0.5，
        使 label 分布在 [-0.5, 0.5]，消除极端值影响。

        Returns
        -------
        X : np.ndarray or None  (n_samples, n_features)
        y : np.ndarray or None  (n_samples,)  rank 归一化后的标签
        """
        all_dates = self.loader.get_trading_dates()
        X_list, y_list, dist_list, rank_w_list = [], [], [], []

        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            if j + self.cfg.label_lookahead > today_idx:
                continue
            # 样本抽样：让训练样本在时间上近似无重叠
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
            if len(sub) < 10:
                continue

            for fc in self.cfg.factor_cols:
                sub[fc] = df[fc].astype(float) if fc in df.columns else np.nan

            raw_label = sub[self.cfg.label_col].astype(float)
            # 截面 rank 归一化：[-0.5, 0.5]，每日独立计算
            sub['_y'] = raw_label.rank(pct=True) - 0.5
            # 截面内 winsorize（去极端值）
            # p1, p99 = raw_label.quantile(0.01), raw_label.quantile(0.99)
            # sub['_y'] = raw_label.clip(p1, p99)

            sub = sub.dropna(subset=['_y'])
            sub[self.cfg.factor_cols] = sub[self.cfg.factor_cols].fillna(0.0)

            n_rows = len(sub)
            # top-k 样本权重：头部股票额外放大，迫使模型优先拟合头部排序
            if self.cfg.top_weight_factor > 1.0:
                threshold = sub['_y'].quantile(1.0 - self.cfg.top_weight_pct)
                row_w = np.where(sub['_y'].values >= threshold,
                                 self.cfg.top_weight_factor, 1.0)
            else:
                row_w = np.ones(n_rows, dtype=np.float64)
            X_list.append(sub[self.cfg.factor_cols].values.astype(np.float64))
            y_list.append(sub['_y'].values.astype(np.float64))
            dist_list.append(np.full(n_rows, today_idx - j, dtype=np.float64))
            rank_w_list.append(row_w)

        if not X_list:
            return None, None, None, None
        return np.vstack(X_list), np.concatenate(y_list), np.concatenate(dist_list), np.concatenate(rank_w_list)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, date_str: str) -> bool:
        """
        以 date_str 为基准日，用滚动窗口历史数据训练 LGBMRegressor。

        Returns
        -------
        bool  True = 训练成功，False = 非交易日或样本不足
        """
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            return False

        today_idx = all_dates.index(date_str)
        X, y, dist, rank_w = self._build_train_data(today_idx)
        if X is None or len(y) < 50:
            return False

        # 时间衰减权重 × top-k 样本权重（逐元素相乘）
        if self.cfg.weight_halflife is not None:
            sample_weight = np.power(2.0, -dist / self.cfg.weight_halflife) * rank_w
        else:
            sample_weight = rank_w if self.cfg.top_weight_factor > 1.0 else None

        model = lgb.LGBMRegressor(**self.cfg.lgbm_params)
        model.fit(X, y, sample_weight=sample_weight)
        self._model = model

        # EMA 平滑特征重要度（稳定分析用，不影响预测）
        imp = model.feature_importances_.astype(float)
        if self._importances is None:
            self._importances = imp
        else:
            self._importances = 0.3 * imp + 0.7 * self._importances

        return True

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------
    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        用当前模型对 date_str 截面内所有股票打分。

        打分 = 模型预测的截面 rank 百分位（越高越好）。

        Returns
        -------
        DataFrame[ts_code, score] 按 score 降序，或 None
        """
        if self._model is None:
            return None

        df = self.loader.get_data(date_str)
        if df is None or self.cfg.stock_col not in df.columns:
            return None

        df_valid = df[[self.cfg.stock_col]].copy()
        for fc in self.cfg.factor_cols:
            df_valid[fc] = df[fc].astype(float) if fc in df.columns else np.nan

        X = df_valid[self.cfg.factor_cols].values.astype(np.float64)
        # _standard 列均值为 0，缺失填 0
        X = np.nan_to_num(X, nan=0.0)

        scores = self._model.predict(X)
        result = df_valid[[self.cfg.stock_col]].copy()
        result['score'] = scores
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # simple_backtest（与 ElasticNetStrategy 接口一致）
    # ------------------------------------------------------------------
    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        Rank IC + 五分组多空收益评估。

        Returns
        -------
        dict: ic_df, ls_df, weights_df（特征重要度时间序列）
        """
        self.reset()
        all_dates = self.loader.get_trading_dates()
        dates     = [d for d in all_dates if start_date <= d <= end_date]

        ic_series          = []
        long_short_returns = []
        topn_returns       = []   # top-N 实际收益序列
        importance_history = {}

        TOP_NS = [10, 50, 100]   # 追踪的头部持仓档位

        for today in dates:
            ok = self.fit(today)
            if not ok:
                continue
            importance_history[today] = dict(
                zip(self.cfg.factor_cols, self._model.feature_importances_)
            )

            signals = self.generate_signals(today)
            if signals is None or signals.empty:
                continue

            df_today = self.loader.get_data(today)
            if df_today is None or self.cfg.label_col not in df_today.columns:
                continue

            signals = signals.merge(
                df_today[[self.cfg.stock_col, self.cfg.label_col]],
                on=self.cfg.stock_col, how='left'
            ).dropna(subset=[self.cfg.label_col])
            if signals.empty:
                continue

            ic = signals['score'].corr(signals[self.cfg.label_col], method='spearman')
            ic_series.append((today, ic))

            # 五分组多空
            signals['rank']  = signals['score'].rank(pct=True)
            signals['group'] = pd.cut(
                signals['rank'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            )
            group_ret = signals.groupby('group', observed=False)[self.cfg.label_col].mean()
            long_ret  = group_ret.get('Q5', np.nan)
            short_ret = group_ret.get('Q1', np.nan)
            long_short_returns.append((today, long_ret, short_ret, long_ret - short_ret))

            # top-N 实际收益（按 score 降序，取前 N 只的 label 均值）
            row = {'date': today}
            mkt_ret = signals[self.cfg.label_col].mean()
            row['mkt'] = mkt_ret
            for k in TOP_NS:
                row[f'top{k}'] = signals.head(k)[self.cfg.label_col].mean()
                row[f'top{k}_alpha'] = row[f'top{k}'] - mkt_ret
            topn_returns.append(row)

        ic_df = pd.DataFrame(ic_series, columns=['date', 'ic']).set_index('date')
        ls_df = pd.DataFrame(
            long_short_returns, columns=['date', 'long_ret', 'short_ret', 'spread']
        ).set_index('date')
        topn_df = pd.DataFrame(topn_returns).set_index('date')
        weights_df = pd.DataFrame(importance_history).T.sort_index()

        print("===== Rank IC 统计 =====")
        print(f"IC 均值:  {ic_df['ic'].mean():.4f}")
        print(f"IC 标准差:{ic_df['ic'].std():.4f}")
        ic_std = ic_df['ic'].std()
        print(f"IR:       {ic_df['ic'].mean() / ic_std:.4f}" if ic_std > 0 else "IR: N/A")
        print(f"IC>0 比例:{(ic_df['ic'] > 0).mean():.2%}")

        n  = self.cfg.label_period
        ls_nonoverlap = ls_df.iloc[::n]
        ann_obs = 252 / n
        cum_spread = (1 + ls_nonoverlap['spread']).prod() - 1
        sp = ls_nonoverlap['spread']
        sharpe = (sp.mean() / sp.std()) * np.sqrt(ann_obs) if sp.std() != 0 else 0.0

        print(f"\n===== 多空收益统计（非重叠，持有期={n}d）=====")
        print(ls_nonoverlap.describe())
        print(f"多空累计收益: {cum_spread:.4%}")
        print(f"年化夏普:     {sharpe:.4f}")

        # ---- top-N 头部收益统计 ----
        topn_nonoverlap = topn_df.iloc[::n]
        print(f"\n===== Top-N 头部多头收益统计（非重叠，持有期={n}d）=====")
        print(f"{'':12s}  {'均值':>8s}  {'胜率':>7s}  {'累计':>9s}  {'年化夏普':>8s}  {'超额均值':>9s}")
        for k in TOP_NS:
            col = f'top{k}'
            acol = f'top{k}_alpha'
            s = topn_nonoverlap[col]
            a = topn_nonoverlap[acol]
            cum = (1 + s).prod() - 1
            sr  = (s.mean() / s.std()) * np.sqrt(ann_obs) if s.std() != 0 else 0.0
            win = (s > 0).mean()
            print(f"  top{k:<8d}  {s.mean():8.4%}  {win:7.2%}  {cum:9.4%}  {sr:8.4f}  {a.mean():9.4%}")

        print("\n===== 特征重要度均值（Top 15）=====")
        print(weights_df.mean().sort_values(ascending=False).head(15))

        # ---- 因子效果分析：IC 差分 + top-N 收益差分 ----
        # top10_diff = 因子活跃时 top-10实际收益均值 − 未使用时 top-10收益均值
        # 比 Q5（top20%）更贴近实盘持仓，是更直接的评估维度
        n_dates      = len(weights_df)
        factor_analysis: dict = {}
        for col in weights_df.columns:
            active_mask  = weights_df[col] > 0
            active_dates = weights_df.index[active_mask]
            zero_dates   = weights_df.index[~active_mask]
            active_rate  = active_mask.mean()
            zero_n       = int((~active_mask).sum())
            reliable     = zero_n >= 10

            ic_active = ic_df['ic'].reindex(active_dates).mean()
            ic_zero   = ic_df['ic'].reindex(zero_dates).mean()
            ic_diff   = float(ic_active - ic_zero) if (reliable and pd.notna(ic_active) and pd.notna(ic_zero)) else np.nan

            # Q5（top20%）差分——保留作参照
            lr_active = ls_df['long_ret'].reindex(active_dates).mean()
            lr_zero   = ls_df['long_ret'].reindex(zero_dates).mean()
            lr_diff   = float(lr_active - lr_zero) if (reliable and pd.notna(lr_active) and pd.notna(lr_zero)) else np.nan

            # top-10 差分——最贴近实盘
            t10_active = topn_df['top10'].reindex(active_dates).mean()
            t10_zero   = topn_df['top10'].reindex(zero_dates).mean()
            t10_diff   = float(t10_active - t10_zero) if (reliable and pd.notna(t10_active) and pd.notna(t10_zero)) else np.nan

            # top-100 差分
            t100_active = topn_df['top100'].reindex(active_dates).mean()
            t100_zero   = topn_df['top100'].reindex(zero_dates).mean()
            t100_diff   = float(t100_active - t100_zero) if (reliable and pd.notna(t100_active) and pd.notna(t100_zero)) else np.nan

            factor_analysis[col] = {
                'mean_importance': float(weights_df[col].mean()),
                'active_rate':     float(active_rate),
                'zero_n':          zero_n,
                'ic_diff':         ic_diff,
                'q5_diff':         lr_diff,
                'top10_diff':      t10_diff,
                'top100_diff':     t100_diff,
            }

        fa_df = pd.DataFrame(factor_analysis).T

        ic_thr  = 0.01
        # 核心判定改为 top10_diff（最贴近实盘）
        reliable   = fa_df['zero_n'] >= 10
        harmful    = fa_df[reliable & (fa_df['ic_diff'] < -ic_thr) & (fa_df['top10_diff'] < 0)].sort_values('top10_diff')
        tradeoff   = fa_df[reliable & (fa_df['ic_diff'] < -ic_thr) & (fa_df['top10_diff'] >= 0)].sort_values('top10_diff', ascending=False)
        positive   = fa_df[reliable & (fa_df['ic_diff'] >  ic_thr)].sort_values('top10_diff', ascending=False)
        unreliable = fa_df[~reliable].sort_values('mean_importance', ascending=False)

        print(f"\n===== 因子效果分类（IC差分 + top-N收益差分，ic阈值={ic_thr}）=====")
        print(f"注：zero_n < 10 的因子对比组过小，ic_diff 不可信，单独列出\n")
        print(f"正面因子（ic_diff > +{ic_thr}，按 top10_diff 排序，共 {len(positive)} 个）：")
        for f, row in positive.iterrows():
            print(f"  {f:<40s}  ic_diff={row['ic_diff']:+.4f}  top10={row['top10_diff']:+.5f}  top100={row['top100_diff']:+.5f}  imp={row['mean_importance']:.1f}")
        print(f"\n真正有害（ic_diff<-{ic_thr} 且 top10_diff<0，共 {len(harmful)} 个，建议删除）：")
        for f, row in harmful.iterrows():
            print(f"  {f:<40s}  ic_diff={row['ic_diff']:+.4f}  top10={row['top10_diff']:+.5f}  top100={row['top100_diff']:+.5f}  imp={row['mean_importance']:.1f}")
        if not tradeoff.empty:
            print(f"\n以排序换头部收益（ic_diff负但top10_diff正，共 {len(tradeoff)} 个，谨慎删除）：")
            for f, row in tradeoff.iterrows():
                print(f"  {f:<40s}  ic_diff={row['ic_diff']:+.4f}  top10={row['top10_diff']:+.5f}  top100={row['top100_diff']:+.5f}  imp={row['mean_importance']:.1f}")
        if not unreliable.empty:
            print(f"\n对比组不足（zero_n < 10，ic_diff 不可信，共 {len(unreliable)} 个，不建议基于此删除）：")
            for f, row in unreliable.iterrows():
                print(f"  {f:<40s}  active={row['active_rate']:.0%}  zero_n={int(row['zero_n'])}  imp={row['mean_importance']:.1f}")

        result = {'ic_df': ic_df, 'ls_df': ls_df, 'topn_df': topn_df,
                  'weights_df': weights_df, 'factor_analysis_df': fa_df}
        self.report_dump(result, label_period=n)
        return result
