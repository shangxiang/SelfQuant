import numpy as np
import pandas as pd
import warnings
from typing import Optional

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class ICWeightedConfig:
    """IC 加权合成因子截面选股策略配置。"""

    def __init__(self):
        # ---- 数据路径 ----
        self.data_dir        = 'data/section/'
        self.calendar_file   = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'

        # ---- 列名 ----
        self.stock_col       = 'ts_code'
        self.label_col       = 'label'
        self.label_period    : int = 5
        self.label_lookahead : int = 6

        # ---- 因子列（使用原始值，策略内部做截面 z-score）----
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
            # 二值信号
            'macd_divergence',
            'macd_air_refuel',
            # 基本面类
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
            # 中短期动量
            # 'ret_10d',
            # 'ret_20d',
            # 'ret_60d',
            # 'dist_52w_high',
            # 'close_ma20_ratio',
            # 'up_day_ratio_20',
            # 'vol_price_corr_20d',
            # 'adx',
            # # 高频痕迹因子
            # 'turnover_amplitude_ratio',
            # 'long_shadow_freq',
            # 'doji_freq',
            # 'intraday_drawdown',
            # 'gap_vs_range_ratio',
        ]

        # ---- 过滤 ----
        self.min_mv    : Optional[int] = None   # 万元；None = 不过滤
        self.filter_st : bool = True

        # ---- IC 滚动窗口 ----
        self.window: int = 40          # 计算 IC 的回望交易日数

        # ---- IC 截断（防止单期极端 IC 主导权重）----
        self.ic_clip: float = 0.15     # 将每期 IC 裁剪到 [-ic_clip, ic_clip]

        # ---- 最小有效截面股票数 ----
        self.min_stocks: int = 30      # 低于此数量的截面跳过 IC 计算

        # ---- 是否归一化权重（Σ|IC_i| = 1）----
        self.normalize_weights: bool = True


class ICWeightedStrategy(BaseStrategy):
    """
    IC 加权合成因子截面选股策略。

    核心逻辑：
      1. fit()：对滚动窗口内每个历史截面，逐因子计算 Rank IC（Spearman），
         取时间均值作为该因子的权重。IC 为负表示该因子反向有效，权重为负。
      2. generate_signals()：对当日截面逐因子做 z-score 标准化，
         然后 composite = Σ IC_i × z_score_i，按 composite 降序打分。

    与 ML 模型的本质区别：
      - 无参数拟合，无过拟合风险；权重直接来自因子自身的近期预测力。
      - 纯线性，不捕捉因子交互，适合作为基准对比 ML 模型的增量价值。
    """

    def __init__(self, config: ICWeightedConfig, data_loader):
        self.cfg    = config
        self.loader = data_loader
        # 每个因子的 IC 权重：dict {factor_name: float}
        self._ic_weights: Optional[dict] = None

    def reset(self) -> None:
        self._ic_weights = None

    # ------------------------------------------------------------------
    # fit：计算滚动 IC 权重
    # ------------------------------------------------------------------
    def fit(self, date_str: str) -> bool:
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            return False

        today_idx = all_dates.index(date_str)

        # 累计每个因子在各截面的 IC
        ic_accum = {f: [] for f in self.cfg.factor_cols}

        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            if j + self.cfg.label_lookahead > today_idx:
                continue

            t_date = all_dates[j]
            df = self.loader.get_data(t_date)
            if df is None:
                continue
            if self.cfg.stock_col not in df.columns or self.cfg.label_col not in df.columns:
                continue

            label = df[self.cfg.label_col].astype(float)
            valid_mask = label.notna()
            if valid_mask.sum() < self.cfg.min_stocks:
                continue

            for f in self.cfg.factor_cols:
                if f not in df.columns:
                    continue
                fval = df[f].astype(float)
                # 只用两者都有效的行
                both_valid = valid_mask & fval.notna()
                if both_valid.sum() < self.cfg.min_stocks:
                    continue
                ic = fval[both_valid].corr(label[both_valid], method='spearman')
                if np.isfinite(ic):
                    ic_accum[f].append(np.clip(ic, -self.cfg.ic_clip, self.cfg.ic_clip))

        # 取各因子 IC 均值
        ic_weights = {}
        for f, vals in ic_accum.items():
            if vals:
                ic_weights[f] = float(np.mean(vals))

        if not ic_weights:
            return False

        # 可选：按 Σ|IC| 归一化，使权重总绝对值 = 1
        if self.cfg.normalize_weights:
            total_abs = sum(abs(v) for v in ic_weights.values())
            if total_abs > 0:
                ic_weights = {f: v / total_abs for f, v in ic_weights.items()}

        self._ic_weights = ic_weights
        return True

    # ------------------------------------------------------------------
    # generate_signals：截面 z-score + IC 加权合成打分
    # ------------------------------------------------------------------
    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        if self._ic_weights is None:
            return None

        df = self.loader.get_data(date_str)
        if df is None or self.cfg.stock_col not in df.columns:
            return None

        result = df[[self.cfg.stock_col]].copy().reset_index(drop=True)
        composite = np.zeros(len(result), dtype=np.float64)

        for f, w in self._ic_weights.items():
            if f not in df.columns or w == 0.0:
                continue
            fval = df[f].astype(float).values
            # 截面 z-score（3σ winsorize 后标准化）
            valid = np.isfinite(fval)
            if valid.sum() < self.cfg.min_stocks:
                continue
            mu  = np.nanmean(fval)
            std = np.nanstd(fval)
            if std < 1e-8:
                continue
            z = (fval - mu) / std
            z = np.clip(z, -3.0, 3.0)
            z = np.where(valid, z, 0.0)   # NaN 填截面均值（z=0）
            composite += w * z

        result['score'] = composite
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # simple_backtest
    # ------------------------------------------------------------------
    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """Rank IC + 五分组多空收益评估，接口与其他策略一致。"""
        self.reset()
        all_dates = self.loader.get_trading_dates()
        dates     = [d for d in all_dates if start_date <= d <= end_date]

        ic_series          = []
        long_short_returns = []
        weight_history     = {}

        for today in dates:
            ok = self.fit(today)
            if not ok:
                continue
            weight_history[today] = dict(self._ic_weights)

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

        ic_df = pd.DataFrame(ic_series, columns=['date', 'ic']).set_index('date')
        ls_df = pd.DataFrame(
            long_short_returns, columns=['date', 'long_ret', 'short_ret', 'spread']
        ).set_index('date')
        weights_df = pd.DataFrame(weight_history).T.sort_index()

        print("===== Rank IC 统计 =====")
        print(f"IC 均值:  {ic_df['ic'].mean():.4f}")
        print(f"IC 标准差:{ic_df['ic'].std():.4f}")
        ic_std = ic_df['ic'].std()
        print(f"IR:       {ic_df['ic'].mean() / ic_std:.4f}" if ic_std > 0 else "IR: N/A")
        print(f"IC>0 比例:{(ic_df['ic'] > 0).mean():.2%}")

        n = self.cfg.label_period
        ls_nonoverlap = ls_df.iloc[::n]
        ann_obs = 252 / n
        sp = ls_nonoverlap['spread']
        cum_spread = (1 + sp).prod() - 1
        sharpe = (sp.mean() / sp.std()) * np.sqrt(ann_obs) if sp.std() != 0 else 0.0

        print(f"\n===== 多空收益统计（非重叠，持有期={n}d）=====")
        print(ls_nonoverlap.describe())
        print(f"多空累计收益: {cum_spread:.4%}")
        print(f"年化夏普:     {sharpe:.4f}")

        print("\n===== 因子 IC 权重均值（按绝对值排序 Top 15）=====")
        mean_w = weights_df.mean().reindex(weights_df.columns)
        print(mean_w.abs().sort_values(ascending=False).head(15)
              .apply(lambda x: f'{mean_w[x.name]:+.4f}').to_string())

        result = {'ic_df': ic_df, 'ls_df': ls_df, 'weights_df': weights_df}
        self.report_dump(result, label_period=n)
        return result
