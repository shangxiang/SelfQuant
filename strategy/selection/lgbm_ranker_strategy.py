import numpy as np
import pandas as pd
import warnings
from typing import Optional, List

import lightgbm as lgb

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class LGBMRankerConfig:
    """LightGBM LambdaRank 截面选股策略超参数配置。"""

    def __init__(self):
        # ---- 数据路径 ----
        self.data_dir        = 'data/section/'
        self.calendar_file   = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'

        # ---- 列名 ----
        self.stock_col      = 'ts_code'
        self.label_col      = 'label'
        self.label_period   : int = 5
        self.label_lookahead: int = 6

        # ---- 相关度分档数（LambdaRank 标签范围 0 ~ n_relevance_levels-1）----
        # 5档：Q1=0, Q2=1, Q3=2, Q4=3, Q5=4
        # LightGBM 默认 gain 公式 2^label-1：label=4 → gain=15，label=0 → gain=0
        # 这使模型把梯度集中在正确排出 Top-20% 上，与 Top-10 选股目标对齐
        self.n_relevance_levels: int = 10

        # ---- 因子列（树模型直接使用原始值，无需标准化）----
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

        # ---- 滚动训练窗口（交易日数）----
        self.window: int = 120

        # ---- 时间衰减权重（半衰期；None 表示不使用）----
        self.weight_halflife: Optional[int] = None

        # ---- LGBMRanker 超参数 ----
        # label_gain=[0,1,2,3,4]：线性 gain 替代默认指数 [0,1,3,7,15]，
        # 避免 Top-20% 因涨停股极值而获得 15x 过大梯度权重
        # lambdarank_truncation_level=10：只优化前 10 名的排序，
        # 与 Top-10 选股目标完全对齐，后面 3590 只的排序不影响梯度
        self.lgbm_params = {
            'objective'                 : 'lambdarank',
            'metric'                    : 'ndcg',
            'eval_at'                   : [5, 10],   # NDCG@5 / @10，与 top-10 选股目标对齐
            # 与 lgbm_strategy 同一套诊断结论（211 特征下 colsample 0.6 会让树趋同）
            'n_estimators'              : 300,
            'learning_rate'             : 0.03,
            'max_depth'                 : 4,
            'num_leaves'                : 15,
            'min_child_samples'         : 300,
            'subsample'                 : 0.8,
            'colsample_bytree'          : 0.3,   # 与 lgbm_strategy 对齐；实测 0.3 优于 0.2
            'reg_alpha'                 : 0.1,
            'reg_lambda'                : 1.0,
            'lambdarank_truncation_level': 10,
            'label_gain'                : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'n_jobs'                    : -1,
            'verbose'                   : -1,
            'random_state'              : 42,
            'eval_at'                   : [5, 10],
        }

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


class LGBMRankerStrategy(BaseStrategy):
    """
    基于 LightGBM LambdaRank 的滚动窗口截面选股策略。

    与 LGBMStrategy（regression）的核心区别：
      1. 使用 LGBMRanker 直接优化 NDCG@20，而非 RMSE，
         目标函数与 Top-10 选股的「排序」需求对齐更紧密。
      2. 训练标签为组内五分位相关度（0-4 整数），每个截面为一个「查询组」。
         LambdaRank 的 gain=2^label-1，Top-20% 股票的梯度权重是 Bottom-20% 的 15 倍。
      3. 需要向 fit() 传入 group 数组（每组样本数），LightGBM 据此划分配对边界。
      4. 时间衰减权重（sample_weight）仍然有效，逐样本施加。

    接口与 LGBMStrategy 完全兼容，可无缝替换引擎中的策略对象。
    """

    def __init__(self, config: LGBMRankerConfig, data_loader):
        self.cfg    = config
        self.loader = data_loader
        self._model: Optional[lgb.LGBMRanker] = None
        self._importances: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._model       = None
        self._importances = None

    # ------------------------------------------------------------------
    # 内部：构建训练集
    # ------------------------------------------------------------------
    def _build_train_data(self, today_idx: int):
        """
        构建 LambdaRank 训练集。

        Label 转换（每个截面独立）：
          原始涨幅 → rank(method='first') → qcut 分 5 档 → 整数标签 [0,4]
          rank(method='first') 打破平局，保证 qcut 分组不报错。

        Group 构建：
          每个截面日的股票数记为一个 group，按时间顺序追加。
          LightGBM 的 group 数组格式为每组样本数（非边界索引）。

        Returns
        -------
        X      : np.ndarray  (n_samples, n_features)
        y      : np.ndarray  (n_samples,)  整数相关度 [0, n_relevance_levels-1]
        groups : list[int]   每个截面的样本数（LightGBM group 参数）
        dist   : np.ndarray  (n_samples,)  距今交易日数（用于时间衰减权重）
        """
        all_dates = self.loader.get_trading_dates()
        X_list, y_list, dist_list, groups = [], [], [], []

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
            # 至少需要 n_relevance_levels 只股票才能分档
            if len(sub) < self.cfg.n_relevance_levels * 10:
                continue

            for fc in self.cfg.factor_cols:
                sub[fc] = df[fc].astype(float) if fc in df.columns else np.nan

            raw_label = sub[self.cfg.label_col].astype(float)

            # 截尾后再排名：先 winsorize 到 90 分位，消除连续涨停股的极值干扰。
            # 涨停股排名仍在头部，但不再因极端收益值独占 label=4 的梯度预算。
            winsor_upper = raw_label.quantile(0.90)
            winsor_lower = raw_label.quantile(0.10)
            clipped_label = raw_label.clip(lower=winsor_lower, upper=winsor_upper)

            # 组内五分位相关度：rank(method='first') 保证唯一，qcut 均等分档
            try:
                labels = pd.qcut(
                    clipped_label.rank(method='first'),
                    q=self.cfg.n_relevance_levels,
                    labels=list(range(self.cfg.n_relevance_levels))
                ).astype(np.int32)
            except ValueError:
                continue

            sub['_y'] = labels
            sub = sub.dropna(subset=['_y'])
            sub[self.cfg.factor_cols] = sub[self.cfg.factor_cols].fillna(0.0)

            n_rows = len(sub)
            if n_rows == 0:
                continue

            X_list.append(sub[self.cfg.factor_cols].values.astype(np.float64))
            y_list.append(sub['_y'].values)
            dist_list.append(np.full(n_rows, today_idx - j, dtype=np.float64))
            groups.append(n_rows)

        if not X_list:
            return None, None, None, None

        return (
            np.vstack(X_list),
            np.concatenate(y_list),
            groups,
            np.concatenate(dist_list),
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, date_str: str) -> bool:
        """
        以 date_str 为基准日，用滚动窗口历史数据训练 LGBMRanker。

        Returns
        -------
        bool  True = 训练成功，False = 非交易日或样本不足
        """
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            return False

        today_idx = all_dates.index(date_str)
        X, y, groups, dist = self._build_train_data(today_idx)
        if X is None or len(groups) < 5:
            return False

        if self.cfg.weight_halflife is not None:
            sample_weight = np.power(2.0, -dist / self.cfg.weight_halflife)
        else:
            sample_weight = None

        model = lgb.LGBMRanker(**self.cfg.lgbm_params)
        model.fit(X, y, group=groups, sample_weight=sample_weight)
        self._model = model

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

        打分 = LambdaRank 预测的相关度分数，越高越好。

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
        X = np.nan_to_num(X, nan=0.0)

        scores = self._model.predict(X)
        result = df_valid[[self.cfg.stock_col]].copy()
        result['score'] = scores
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # simple_backtest
    # ------------------------------------------------------------------
    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        Rank IC + 五分组多空收益 + NDCG@10 评估。

        Returns
        -------
        dict: ic_df, ls_df, weights_df, ndcg_df
        """
        self.reset()
        all_dates = self.loader.get_trading_dates()
        dates     = [d for d in all_dates if start_date <= d <= end_date]

        ic_series          = []
        long_short_returns = []
        importance_history = {}
        ndcg_series        = []
        
        # 因子IC分析
        factor_ic_series = {fc: [] for fc in self.cfg.factor_cols}

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

            # Rank IC（与实际涨幅的 Spearman 相关）
            ic = signals['score'].corr(signals[self.cfg.label_col], method='spearman')
            ic_series.append((today, ic))

            # 计算每个因子的 IC
            for fc in self.cfg.factor_cols:
                if fc in signals.columns:
                    factor_ic = signals[fc].corr(signals[self.cfg.label_col], method='spearman')
                    factor_ic_series[fc].append((today, factor_ic))

            # NDCG@10：以实际涨幅排名作为相关度
            try:
                true_relevance = signals[self.cfg.label_col].rank(method='first').values
                pred_scores    = signals['score'].values
                # sklearn.metrics.ndcg_score 期望 shape (1, n_samples)
                from sklearn.metrics import ndcg_score as sk_ndcg
                ndcg10 = sk_ndcg(
                    true_relevance.reshape(1, -1),
                    pred_scores.reshape(1, -1),
                    k=10
                )
                ndcg_series.append((today, ndcg10))
            except Exception:
                pass

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

        ic_df = pd.DataFrame(ic_series, columns=['date', 'ic']).set_index('date')
        ls_df = pd.DataFrame(
            long_short_returns, columns=['date', 'long_ret', 'short_ret', 'spread']
        ).set_index('date')
        weights_df = pd.DataFrame(importance_history).T.sort_index()
        ndcg_df    = pd.DataFrame(ndcg_series, columns=['date', 'ndcg10']).set_index('date')

        print("===== Rank IC 统计 =====")
        print(f"IC 均值:  {ic_df['ic'].mean():.4f}")
        print(f"IC 标准差:{ic_df['ic'].std():.4f}")
        ic_std = ic_df['ic'].std()
        print(f"IR:       {ic_df['ic'].mean() / ic_std:.4f}" if ic_std > 0 else "IR: N/A")
        print(f"IC>0 比例:{(ic_df['ic'] > 0).mean():.2%}")

        if not ndcg_df.empty:
            print(f"\n===== NDCG@10 统计 =====")
            print(f"均值: {ndcg_df['ndcg10'].mean():.4f}  |  中位数: {ndcg_df['ndcg10'].median():.4f}")

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

        print("\n===== 特征重要度均值（Top 15）=====")
        print(weights_df.mean().sort_values(ascending=False).head(15))

        # ---- 因子 IC 分析 ----
        factor_ic_records = []
        for fc in self.cfg.factor_cols:
            if factor_ic_series[fc]:
                ic_vals = pd.DataFrame(factor_ic_series[fc], columns=['date', 'ic']).set_index('date')['ic']
                ic_mean = ic_vals.mean()
                ic_std = ic_vals.std()
                ir = ic_mean / ic_std if ic_std > 0 else 0.0
                ic_pos_ratio = (ic_vals > 0).mean()
                factor_ic_records.append({
                    'factor': fc,
                    'IC_mean': ic_mean,
                    'IC_std': ic_std,
                    'IR': ir,
                    'IC>0_ratio': ic_pos_ratio
                })
        
        factor_ic_df = pd.DataFrame(factor_ic_records).sort_values('IC_mean', ascending=False)
        
        print("\n===== 因子 IC 统计（Top 15）=====")
        print(factor_ic_df.head(15).to_string(index=False))
        
        # 输出到文件
        factor_ic_df.to_csv('factor_ic_analysis.csv', index=False)
        print("\n因子 IC 分析已输出到 factor_ic_analysis.csv")

        result = {
            'ic_df'     : ic_df,
            'ls_df'     : ls_df,
            'weights_df': weights_df,
            'ndcg_df'   : ndcg_df,
            'factor_ic_df': factor_ic_df,
        }
        self.report_dump(result, label_period=n)
        return result
