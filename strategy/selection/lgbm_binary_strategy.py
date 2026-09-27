import numpy as np
import pandas as pd
import warnings
from typing import Optional

import lightgbm as lgb

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class LGBMBinaryConfig:
    """LightGBM 二分类截面选股策略超参数配置。
    
    核心思路：不做全截面排序，只做"Top-K vs 非 Top-K"的二分类。
    模型只需学习"什么是强势股"，不关心中间段排序，更贴合实际选股需求。
    """

    def __init__(self):
        # ---- 数据路径 ----
        self.data_dir        = 'data/section/'
        self.calendar_file   = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'

        # ---- 列名 ----
        self.stock_col  = 'ts_code'
        self.label_col  = 'label'  # 中性化标准化后的 5 日收益
        self.label_period: int = 5
        self.label_lookahead: int = 6

        # ---- Top-K 定义 ----
        # 二分类阈值：Top top_k_pct 的股票标记为正例（1），其余为负例（0）
        self.top_k_pct: float = 0.05  # Top 5% = 约 250 只股票

        # ---- 因子列（树模型不需要标准化，直接用原始值）----
        self.factor_cols = [
            # 估值类
            'pe_ttm',
            'pb',
            'dv_ttm',
            # 技术类
            'macd',
            'cci',
            'force_index_smoothed',
            'net_mf_amount',
            'K',
            'D',
            'J',
            'rsi',
            'turnover_rate_x',
            'volume_ratio',
            'positive_flow',
            'negative_flow',
            'total_mv',
            'volatility_20d',
            'reversal_5d',
            # 二值信号（不做标准化）
            'macd_divergence',
            'macd_air_refuel',
            # 基本面类（行业+市值中性化后标准化）
            'gross_margin',
            'debt_ratio',
            'roe_ttm',
            'revenue_growth_yoy',
            'profit_growth_yoy',
            'accruals',
            'mfi',
            'vwap',
            'close_to_vwap_ratio',
            'mtm_margin_balance_change',
            'big_order_ratio',
            'rzye',
            # Fama-French 风格因子
            'size_factor',
            'smb_squared',
            'value_factor',
            'cma_factor',
            'asset_growth_yoy',
            'momentum_12_1',
            # 中短期动量 / 技术形态因子
            'ret_10d',
            'dist_52w_high',
            'close_ma20_ratio',
            'up_day_ratio_20',
            'vol_price_corr_20d',
            'adx',
            # 高频痕迹因子
            'turnover_amplitude_ratio',
            'gap_vs_range_ratio',
            # 更多时序因子
            'K_chg_5d',
            'K_chg_10d',
            'D_chg_5d',   
            'D_chg_10d',   
            'J_chg_5d',   
            'J_chg_10d',
            'rsi_chg_10d',   
            'macd_chg_5d',  
            'macd_chg_10d',
            'adx_chg_5d',   
            'volatility_20d_chg_5d',   
            'volatility_20d_chg_10d',
            'turnover_rate_x_chg_5d',  
            'turnover_rate_x_chg_10d',
            'reversal_5d_chg_5d',   
            'reversal_5d_chg_10d',
            'momentum_12_1_chg_5d',    
            'rzye_chg_5d',  
            'rzye_chg_10d',
            # 多项式形状因子
            'poly_close_a1',
            'poly_close_a2',
            'poly_vol_a2',
        ]

        # ---- 市值过滤（单位：万元；None 表示不过滤）----
        self.min_mv: Optional[int] = 0
        # ---- ST 过滤（True = 过滤掉 ST 股，默认开启）----
        self.filter_st: bool = True

        # ---- 滚动训练窗口（交易日数）----
        self.window: int = 40

        # ---- 时间衰减权重（半衰期，单位：交易日；None 表示不使用）----
        self.weight_halflife: Optional[int] = 20

        # ---- LightGBM 二分类超参数 ----
        self.lgbm_params = {
            'objective':         'binary',
            'metric':            'auc',
            'n_estimators':      100,
            'learning_rate':     0.05,
            'max_depth':         3,
            'num_leaves':        7,
            'min_child_samples': 300,
            'subsample':         0.8,
            'colsample_bytree':  0.6,
            'reg_alpha':         0.1,
            'reg_lambda':        5.0,
            'n_jobs':            -1,
            'verbose':           -1,
            'random_state':      42,
            # scale_pos_weight 会在 fit() 中动态计算
        }

        # 是否使用早停（需要验证集）；False 则跑满 n_estimators
        self.early_stopping: bool = False
        self.early_stopping_rounds: int = 20


class LGBMBinaryStrategy(BaseStrategy):
    """
    基于 LightGBM 二分类的滚动窗口截面选股策略。

    与 LGBMStrategy（回归版）的核心区别：
      1. 目标函数：二分类（binary）而非回归（regression）
      2. 标签定义：Top-K 股票 = 1（正例），其余 = 0（负例）
      3. 评估指标：AUC 而非 RMSE，天然关注正例（Top-K）的识别能力
      4. 预测输出：概率值作为 score，越高表示越可能是强势股
    
    优势：
      - 模型只需学习"什么是强势股"，不关心中间段排序
      - 训练目标与实际选股需求（Top-K）完全对齐
      - 避免模型把算力浪费在尾部无关紧要的排序上
      - AUC 指标天然关注正例，更适合不平衡分类
    
    接口与 LGBMStrategy 完全兼容，可无缝替换引擎中的策略对象。
    """

    def __init__(self, config: LGBMBinaryConfig, data_loader):
        self.cfg    = config
        self.loader = data_loader
        self._model: Optional[lgb.LGBMClassifier] = None
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
        构建二分类训练集。

        label 转换：每个截面日内，Top top_k_pct 的股票标记为 1，其余为 0。

        Returns
        -------
        X : np.ndarray or None  (n_samples, n_features)
        y : np.ndarray or None  (n_samples,)  二分类标签 [0, 1]
        """
        all_dates = self.loader.get_trading_dates()
        X_list, y_list, dist_list = [], [], []

        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
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
            
            # 二分类标签：Top top_k_pct = 1, 其余 = 0
            threshold = raw_label.quantile(1.0 - self.cfg.top_k_pct)
            sub['_y'] = (raw_label >= threshold).astype(int)

            sub = sub.dropna(subset=['_y'])
            sub[self.cfg.factor_cols] = sub[self.cfg.factor_cols].fillna(0.0)

            n_rows = len(sub)
            X_list.append(sub[self.cfg.factor_cols].values.astype(np.float64))
            y_list.append(sub['_y'].values.astype(np.int32))
            dist_list.append(np.full(n_rows, today_idx - j, dtype=np.float64))

        if not X_list:
            return None, None, None
        return np.vstack(X_list), np.concatenate(y_list), np.concatenate(dist_list)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, date_str: str) -> bool:
        """
        以 date_str 为基准日，用滚动窗口历史数据训练 LGBMClassifier（二分类）。

        Returns
        -------
        bool  True = 训练成功，False = 非交易日或样本不足
        """
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            return False

        today_idx = all_dates.index(date_str)
        X, y, dist = self._build_train_data(today_idx)
        if X is None or len(y) < 50:
            return False

        # 计算正负样本比例，动态设置 scale_pos_weight
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # 时间衰减权重
        if self.cfg.weight_halflife is not None:
            sample_weight = np.power(2.0, -dist / self.cfg.weight_halflife)
        else:
            sample_weight = None

        # 合并参数
        params = self.cfg.lgbm_params.copy()
        params['scale_pos_weight'] = scale_pos_weight

        model = lgb.LGBMClassifier(**params)
        
        # 训练
        eval_set = [(X, y)] if self.cfg.early_stopping else None
        model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds)] if self.cfg.early_stopping else None
        )
        
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

        打分 = 模型预测的概率值（越高越好）。

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

        # 预测概率值（正例概率）
        scores = self._model.predict_proba(X)[:, 1]
        
        result = df_valid[[self.cfg.stock_col]].copy()
        result['score'] = scores
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # simple_backtest（与 LGBMStrategy 接口一致）
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

            # top-N 实际收益
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

        result = {'ic_df': ic_df, 'ls_df': ls_df, 'topn_df': topn_df,
                  'weights_df': weights_df}
        self.report_dump(result, label_period=n)
        return result
