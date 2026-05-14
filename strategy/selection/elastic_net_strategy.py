import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import ElasticNetCV
from typing import Optional

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


class ElasticNetConfig:
    def __init__(self):
        # 数据路径（相对于项目根目录）
        self.data_dir = 'data/section/'
        self.calendar_file = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'
        self.stock_col = 'ts_code'
        self.label_col = 'label'
        # 因子列
        self.factor_cols = [
            'pe_ttm_standard', 'pb_standard', 'dv_ttm_standard',
            'macd_standard', 'rsi_standard',
            'turnover_rate_x_standard', 'volume_ratio_standard',
            'positive_flow_standard', 'total_mv_standard',
            'macd_divergence',
            'macd_air_refuel',
            'volatility_20d_standard',
            'reversal_5d_standard'
        ]
        # 模型超参数
        self.window = 40
        self.l1_ratio_grid = [0.1, 0.5, 0.7, 0.9, 0.95, 1]
        self.cv = 5
        self.smooth_weights = True
        self.smooth_alpha = 0.2


class ElasticNetStrategy(BaseStrategy):
    """
    弹性网络因子选股策略。
    fit(date_str) 训练模型，generate_signals(date_str) 返回当日股票打分。
    """

    def __init__(self, config: ElasticNetConfig, data_loader):
        self.cfg = config
        self.loader = data_loader
        self._weights = None

    def reset(self) -> None:
        self._weights = None

    def _build_train_data(self, today_idx: int):
        """构建滚动窗口训练集，跳过标签未实现的最后5天"""
        all_dates = self.loader.get_trading_dates()
        X_list, y_list = [], []
        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            if j + 5 > today_idx:
                continue
            t_date = all_dates[j]
            df = self.loader.get_data(t_date)
            if df is None:
                continue
            needed = [self.cfg.stock_col] + self.cfg.factor_cols + [self.cfg.label_col]
            if not all(c in df.columns for c in needed):
                continue
            sub = df[needed].copy()
            sub = sub.dropna(subset=[self.cfg.label_col])
            if sub.empty:
                continue
            sub[self.cfg.factor_cols] = sub[self.cfg.factor_cols].astype(float).fillna(0.0)
            X_list.append(sub[self.cfg.factor_cols].values)
            y_list.append(sub[self.cfg.label_col].astype(float).values)
        if not X_list:
            return None, None
        return np.vstack(X_list), np.concatenate(y_list)

    def fit(self, date_str: str) -> bool:
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            print("今日并非交易日")
            return False
        today_idx = all_dates.index(date_str)
        X, y = self._build_train_data(today_idx)
        if X is None or len(y) < 30:
            return False
        model = ElasticNetCV(
            l1_ratio=self.cfg.l1_ratio_grid,
            cv=self.cfg.cv,
            max_iter=5000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        raw_weights = model.coef_
        if self.cfg.smooth_weights and self._weights is not None:
            self._weights = (self.cfg.smooth_alpha * raw_weights +
                             (1 - self.cfg.smooth_alpha) * self._weights)
        else:
            self._weights = raw_weights
        return True

    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """返回当日所有股票的打分，不依赖 label（可用于实盘）"""
        if self._weights is None:
            return None
        df = self.loader.get_data(date_str)
        if df is None:
            return None
        needed_factors = [self.cfg.stock_col] + self.cfg.factor_cols
        if not all(c in df.columns for c in needed_factors):
            return None
        df_valid = df[needed_factors].copy()
        df_valid[self.cfg.factor_cols] = df_valid[self.cfg.factor_cols].astype(float).fillna(0.0)
        scores = df_valid[self.cfg.factor_cols].values.dot(self._weights)
        result = df_valid[[self.cfg.stock_col]].copy()
        result['score'] = scores
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        纯信号质量评估：Rank IC + 五分组多空收益。
        每日调用 fit + generate_signals，与 label 拼接计算 IC。
        """
        self.reset()
        all_dates = self.loader.get_trading_dates()
        dates = [d for d in all_dates if start_date <= d <= end_date]

        ic_series = []
        long_short_returns = []
        weights_history = {}

        for today in dates:
            ok = self.fit(today)
            if not ok:
                continue
            weights_history[today] = dict(zip(self.cfg.factor_cols, self._weights))

            signals = self.generate_signals(today)
            if signals is None or signals.empty:
                continue

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

            ic = signals['score'].corr(signals[self.cfg.label_col], method='spearman')
            ic_series.append((today, ic))

            signals['rank'] = signals['score'].rank(pct=True)
            signals['group'] = pd.cut(
                signals['rank'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            )
            group_ret = signals.groupby('group')[self.cfg.label_col].mean()
            long_ret = group_ret.get('Q5', np.nan)
            short_ret = group_ret.get('Q1', np.nan)
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
        print("\n===== 多空收益统计 =====")
        print(ls_df.describe())
        cum_spread = (1 + ls_df['spread']).prod() - 1
        print(f"多空累计收益: {cum_spread:.4%}")
        sharpe = (ls_df['spread'].mean() / ls_df['spread'].std()) * np.sqrt(252)
        print(f"年化夏普: {sharpe:.4f}")
        print("\n===== 因子权重均值 =====")
        print(weights_df.mean().sort_values(ascending=False))

        return {'ic_df': ic_df, 'ls_df': ls_df, 'weights_df': weights_df}
