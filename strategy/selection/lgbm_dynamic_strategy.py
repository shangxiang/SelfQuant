import numpy as np
import pandas as pd
import warnings
from typing import Optional
from itertools import product

import lightgbm as lgb

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class LGBMDynamicConfig:
    """LightGBM 动态超参数截面选股策略配置。

    与 LGBMConfig 的核心区别：
      - 超参数不写死，而是定义搜索空间
      - 定期（默认每月）用过去 N 个月数据做网格搜索，选 Rank IC 最高的参数
      - 搜索到的最优参数用于未来 M 个月的预测
    """

    def __init__(self):
        # ---- 数据路径 ----
        self.data_dir        = 'data/section/'
        self.calendar_file   = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'

        # ---- 列名 ----
        self.stock_col  = 'ts_code'
        self.label_col  = 'label'
        self.label_period: int = 5
        self.label_lookahead: int = 6

        # ---- 因子列（与 LGBM 原版一致）----
        self.factor_cols = [
            # 估值类
            'pe_ttm', 'pb', 'dv_ttm',
            # 技术类
            'macd', 'cci', 'force_index_smoothed', 'net_mf_amount',
            'K', 'D', 'J', 'rsi', 'turnover_rate_x', 'volume_ratio',
            'positive_flow', 'negative_flow', 'total_mv',
            'volatility_20d', 'reversal_5d',
            # 二值信号
            'macd_divergence', 'macd_air_refuel',
            # 基本面类
            'gross_margin', 'debt_ratio', 'roe_ttm',
            'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
            'mfi', 'vwap', 'close_to_vwap_ratio',
            'mtm_margin_balance_change', 'big_order_ratio', 'rzye',
            # Fama-French 风格因子
            'size_factor', 'smb_squared', 'value_factor', 'cma_factor',
            'asset_growth_yoy', 'momentum_12_1',
            # 中短期动量 / 技术形态因子
            'ret_10d', 'dist_52w_high', 'close_ma20_ratio',
            'up_day_ratio_20', 'vol_price_corr_20d', 'adx',
            # 高频痕迹因子
            'turnover_amplitude_ratio', 'gap_vs_range_ratio',
            # 时序变化因子
            'K_chg_5d', 'K_chg_10d', 'D_chg_5d', 'D_chg_10d',
            'J_chg_5d', 'J_chg_10d', 'rsi_chg_10d',
            'macd_chg_5d', 'macd_chg_10d', 'adx_chg_5d',
            'volatility_20d_chg_5d', 'volatility_20d_chg_10d',
            'turnover_rate_x_chg_5d', 'turnover_rate_x_chg_10d',
            'reversal_5d_chg_5d', 'reversal_5d_chg_10d',
            'momentum_12_1_chg_5d', 'rzye_chg_5d', 'rzye_chg_10d',
            # 多项式形状因子
            'poly_close_a1', 'poly_close_a2', 'poly_vol_a2',
        ]

        # ---- 市值过滤 / ST 过滤 ----
        self.min_mv: Optional[int] = 0
        self.filter_st: bool = True

        # ---- 滚动训练窗口 ----
        self.window: int = 40

        # ---- 时间衰减权重 ----
        self.weight_halflife: Optional[int] = 20

        # ---- Top-k 样本加权 ----
        self.top_weight_pct:    float = 0.1
        self.top_weight_factor: float = 1.0

        # ================================================================
        #  动态超参数搜索配置
        # ================================================================

        # 搜索空间（网格搜索会遍历所有组合）
        # 组合数 = 3×3×3×3×2×3×3 = 2916，需要控制
        self.param_search_space = {
            'max_depth':         [2, 3, 4],
            'num_leaves':        [5, 7, 10],
            'min_child_samples': [200, 300, 500],
            'n_estimators':      [50, 70, 100],
            'learning_rate':     [0.03, 0.05],
        }

        # 二级搜索空间（与一级交替搜索，避免组合爆炸）
        # 组合数 = 3×4 = 12
        self.param_search_space_v2 = {
            'window':           [30, 40, 60],
            'weight_halflife':  [10, 20, 30, None],
        }

        # 固定不变的参数
        self.param_fixed = {
            'objective':        'regression',
            'metric':           'rmse',
            'subsample':        0.8,
            'colsample_bytree': 0.6,
            'reg_alpha':        0.1,
            'reg_lambda':       5.0,
            'n_jobs':           -1,
            'verbose':          -1,
            'random_state':     42,
        }

        # 搜索频率（交易日数），默认 20 ≈ 1 个月
        self.tune_interval: int = 20

        # 二级搜索频率（每 N 次一级搜索后，做一次二级搜索）
        self.tune_interval_v2: int = 3  # 每 3 个月搜一次 window/halflife

        # 搜索用的历史窗口（交易日数），默认 120 ≈ 6 个月
        self.tune_lookback: int = 120

        # 验证集比例（时间序列最后 X% 用于评估参数）
        self.tune_val_ratio: float = 0.2

        # 兜底参数（搜索失败时使用）
        self.default_params = {
            'n_estimators':      70,
            'learning_rate':     0.05,
            'max_depth':         3,
            'num_leaves':        7,
            'min_child_samples': 300,
        }

        # 是否打印搜索过程
        self.verbose_tune: bool = True


class LGBMDynamicStrategy(BaseStrategy):
    """带动态超参数搜索的 LightGBM 截面选股策略。

    与 LGBMStrategy 的区别：
      - 每隔 tune_interval 个交易日，用过去 lookback 天数据做网格搜索
      - 搜索目标：验证集上的 Rank IC 均值最高
      - 搜索到的参数用于下一个周期的预测
      - 非搜索日直接使用缓存的最优参数
    """

    def __init__(self, config: LGBMDynamicConfig, data_loader):
        self.cfg    = config
        self.loader = data_loader
        self._model: Optional[lgb.LGBMRegressor] = None
        self._importances: Optional[np.ndarray] = None

        # 动态参数缓存
        self._current_params: dict = config.default_params.copy()
        self._last_tune_idx: Optional[int] = None
        self._tune_count: int = 0  # 一级搜索计数器，用于触发二级搜索

    def reset(self) -> None:
        self._model          = None
        self._importances    = None
        self._current_params = self.cfg.default_params.copy()
        self._last_tune_idx  = None

    # ------------------------------------------------------------------
    # 内部：构建训练集（与 LGBMStrategy 一致）
    # ------------------------------------------------------------------
    def _build_train_data(self, today_idx: int, window: int = None):
        if window is None:
            window = self.cfg.window

        all_dates = self.loader.get_trading_dates()
        X_list, y_list, dist_list, rank_w_list = [], [], [], []

        for j in range(max(0, today_idx - window + 1), today_idx + 1):
            if j + self.cfg.label_lookahead > today_idx:
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
            sub['_y'] = raw_label.rank(pct=True) - 0.5

            sub = sub.dropna(subset=['_y'])
            sub[self.cfg.factor_cols] = sub[self.cfg.factor_cols].fillna(0.0)

            n_rows = len(sub)
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
        return (np.vstack(X_list), np.concatenate(y_list),
                np.concatenate(dist_list), np.concatenate(rank_w_list))

    # ------------------------------------------------------------------
    # 内部：网格搜索最优参数
    # ------------------------------------------------------------------
    def _evaluate_params(self, params: dict, start_idx: int, train_end: int,
                         val_start: int, end_idx: int, window: int = None) -> float:
        """
        评估一组参数在验证集上的 Rank IC 均值。
        """
        full_params = {**self.cfg.param_fixed, **params}
        model = lgb.LGBMRegressor(**full_params)

        X, y, dist, rank_w = self._build_train_data(train_end, window=window)
        if X is None or len(y) < 50:
            return -999

        halflife = params.get('weight_halflife', self.cfg.weight_halflife)
        if halflife is not None:
            sample_weight = np.power(2.0, -dist / halflife) * rank_w
        else:
            sample_weight = rank_w if self.cfg.top_weight_factor > 1.0 else None

        model.fit(X, y, sample_weight=sample_weight)

        all_dates = self.loader.get_trading_dates()
        ics = []
        for v_idx in range(val_start, end_idx + 1):
            df = self.loader.get_data(all_dates[v_idx])
            if df is None or self.cfg.stock_col not in df.columns:
                continue
            if self.cfg.label_col not in df.columns:
                continue

            df_valid = df[[self.cfg.stock_col, self.cfg.label_col]].copy()
            for fc in self.cfg.factor_cols:
                df_valid[fc] = df[fc].astype(float) if fc in df.columns else np.nan
            df_valid = df_valid.dropna(subset=[self.cfg.label_col])
            if len(df_valid) < 10:
                continue

            X_val = df_valid[self.cfg.factor_cols].values.astype(np.float64)
            X_val = np.nan_to_num(X_val, nan=0.0)
            scores = model.predict(X_val)

            ic = pd.Series(scores).corr(df_valid[self.cfg.label_col].astype(float), method='spearman')
            if not np.isnan(ic):
                ics.append(ic)

        return np.mean(ics) if ics else -999

    def _search_best_params(self, today_idx: int, level: int = 1) -> dict:
        """
        网格搜索最优参数。

        Parameters
        ----------
        level : int
            1 = 搜索模型超参数（max_depth 等，每月执行）
            2 = 搜索 window + weight_halflife（每 3 个月执行）
        """
        all_dates = self.loader.get_trading_dates()
        lookback  = self.cfg.tune_lookback
        val_ratio = self.cfg.tune_val_ratio

        start_idx = max(0, today_idx - lookback)
        end_idx   = today_idx - self.cfg.label_lookahead
        if end_idx - start_idx < 60:
            return self.cfg.default_params.copy()

        split = int(start_idx + (end_idx - start_idx) * (1 - val_ratio))
        train_end = split
        val_start = split + 1

        if val_start >= end_idx:
            return self.cfg.default_params.copy()

        if level == 1:
            keys = list(self.cfg.param_search_space.keys())
            values = list(self.cfg.param_search_space.values())
            label = "一级（模型超参数）"
        else:
            keys = list(self.cfg.param_search_space_v2.keys())
            values = list(self.cfg.param_search_space_v2.values())
            label = "二级（window/halflife）"

        combos = [dict(zip(keys, v)) for v in product(*values)]

        if self.cfg.verbose_tune:
            print(f"  [Tune-{label}] 搜索 {len(combos)} 组参数 "
                  f"(训练: {all_dates[start_idx]}~{all_dates[train_end]}, "
                  f"验证: {all_dates[val_start]}~{all_dates[end_idx]})")

        best_ic = -999
        best_params = self.cfg.default_params.copy() if level == 1 else {}

        for params in combos:
            # 二级搜索需要合并当前模型超参数
            if level == 2:
                eval_params = {**self._current_params, **params}
                # window 影响训练集构建
                w = params.get('window', self.cfg.window)
            else:
                eval_params = params
                w = train_end - start_idx + 1

            mean_ic = self._evaluate_params(eval_params, start_idx, train_end,
                                            val_start, end_idx, window=w)

            if mean_ic > best_ic:
                best_ic = mean_ic
                best_params = params.copy()

        if self.cfg.verbose_tune:
            print(f"  [Tune-{label}] 最优 IC={best_ic:.4f}, 参数={best_params}")

        return best_params

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, date_str: str) -> bool:
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            return False

        today_idx = all_dates.index(date_str)

        # 定期搜索参数
        if (self._last_tune_idx is None or
                today_idx - self._last_tune_idx >= self.cfg.tune_interval):
            # 一级搜索：模型超参数
            self._current_params.update(self._search_best_params(today_idx, level=1))
            self._last_tune_idx = today_idx
            self._tune_count += 1

            # 每 N 次一级搜索后，触发二级搜索
            if self._tune_count % self.cfg.tune_interval_v2 == 0:
                v2_params = self._search_best_params(today_idx, level=2)
                self._current_params.update(v2_params)
                # 更新 config 中的 window/halflife，供 _build_train_data 使用
                if 'window' in v2_params:
                    self.cfg.window = v2_params['window']
                if 'weight_halflife' in v2_params:
                    self.cfg.weight_halflife = v2_params['weight_halflife']

        # 用当前最优参数训练模型
        X, y, dist, rank_w = self._build_train_data(today_idx)
        if X is None or len(y) < 50:
            return False

        # _current_params 可能包含 window/weight_halflife，需过滤
        lgbm_keys = set(self.cfg.param_fixed.keys()) | set(self.cfg.param_search_space.keys())
        lgb_params = {k: v for k, v in self._current_params.items() if k in lgbm_keys}
        full_params = {**self.cfg.param_fixed, **lgb_params}
        if self.cfg.weight_halflife is not None:
            sample_weight = np.power(2.0, -dist / self.cfg.weight_halflife) * rank_w
        else:
            sample_weight = rank_w if self.cfg.top_weight_factor > 1.0 else None

        model = lgb.LGBMRegressor(**full_params)
        model.fit(X, y, sample_weight=sample_weight)
        self._model = model

        imp = model.feature_importances_.astype(float)
        if self._importances is None:
            self._importances = imp
        else:
            self._importances = 0.3 * imp + 0.7 * self._importances

        return True

    # ------------------------------------------------------------------
    # generate_signals（与 LGBMStrategy 一致）
    # ------------------------------------------------------------------
    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
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
    # simple_backtest（与 LGBMStrategy 一致）
    # ------------------------------------------------------------------
    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        self.reset()
        all_dates = self.loader.get_trading_dates()
        dates     = [d for d in all_dates if start_date <= d <= end_date]

        ic_series          = []
        long_short_returns = []
        topn_returns       = []
        importance_history = {}

        TOP_NS = [10, 50, 100]

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

        # 因子效果分析
        n_dates = len(weights_df)
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

            lr_active = ls_df['long_ret'].reindex(active_dates).mean()
            lr_zero   = ls_df['long_ret'].reindex(zero_dates).mean()
            lr_diff   = float(lr_active - lr_zero) if (reliable and pd.notna(lr_active) and pd.notna(lr_zero)) else np.nan

            t10_active = topn_df['top10'].reindex(active_dates).mean()
            t10_zero   = topn_df['top10'].reindex(zero_dates).mean()
            t10_diff   = float(t10_active - t10_zero) if (reliable and pd.notna(t10_active) and pd.notna(t10_zero)) else np.nan

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
