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
        self.label_col = 'label_standard'     # 预测目标列（未来 N 日收益率）
        # label 所代表的持有期天数，用于多空收益的非重叠采样和夏普年化
        # 与 label_col 对应：'label'=5, 'label_10'=10, 'label_25'=25
        self.label_period: int = 5

        # label 在截面日 d 之后多少个交易日才能确认（即 label 最后用到的未来价格偏移量）
        # 旧 label（T→T+5）：5 日后可知，lookahead=5
        # 新 label（T+1→T+6）：6 日后可知，lookahead=6
        # _build_train_data 用此值截断训练集末端，防止未来函数
        self.label_lookahead: int = 6

        # ---- 参与建模的因子列 ----
        self.factor_cols = [
            # 估值类
            'pe_ttm_standard',          # 市盈率（TTM）
            'pb_standard',              # 市净率
            'dv_ttm_standard',          # 股息率（TTM）
            # 技术类
            'macd_standard',            # MACD
            'rsi_standard',             # RSI
            'turnover_rate_x_standard', # 换手率
            'volume_ratio_standard',    # 量比
            'positive_flow_standard',   # 主力净流入
            'total_mv_standard',        # 总市值
            'volatility_20d_standard',  # 20 日波动率
            'reversal_5d_standard',     # 5 日反转因子
            # 二值信号（不做标准化）
            'macd_divergence',          # MACD 底背离信号
            'macd_air_refuel',          # MACD 零轴附近缩量信号
            # 基本面类（行业+市值中性化后标准化）
            'gross_margin_standard',        # 毛利率
            'debt_ratio_standard',          # 资产负债率
            'roe_ttm_standard',             # ROE（TTM）
            'revenue_growth_yoy_standard',  # 营收同比增速
            'profit_growth_yoy_standard',   # 净利润同比增速
            'accruals_standard',            # 应计项目（盈利质量）
            # Fama-French 风格因子
            'size_factor_standard',         # 规模因子（-ln 流通市值），直接 z-score
            'smb_squared_standard',         # 规模²，直接 z-score
            'value_factor_standard',        # 价值因子（1/PB），中性化后标准化
            'cma_factor_standard',          # 投资因子（资产增速取反），中性化后标准化
            'asset_growth_yoy_standard',    # 资产增速（CMA 代理），中性化后标准化
            'momentum_12_1_standard',       # 12-1 月动量，中性化后标准化
            # 高阶交叉因子
            'smb_mom_standard',             # 规模×动量
            'smb_squared_mom_standard',     # 规模²×动量
            'hml_rmw_standard',             # 价值×盈利质量
            'smb_hml_standard',             # 规模×价值
            'vol_mom_standard',             # 波动率×动量（动量崩溃信号）
        ]

        # ---- 模型超参数 ----
        # 滚动训练窗口，单位：交易日
        # 越大则训练数据越多但对近期市场反应越慢
        self.window = 80

        # ElasticNetCV 的 L1 比例搜索网格
        # 0 = 纯 Ridge（L2），1 = 纯 Lasso（L1），中间值为混合
        self.l1_ratio_grid = [0.1, 0.5, 0.7, 0.9, 0.95, 1]

        # 交叉验证折数，用于选择最优超参数
        self.cv = 5

        # 是否对每日更新的因子权重做指数移动平滑（EMA）
        # 可减少模型在相邻交易日之间的权重跳变，使仓位更稳定
        self.smooth_weights = False

        # EMA 平滑系数：新权重 = alpha * 新值 + (1-alpha) * 旧值
        # alpha 越小，历史权重影响越大，变化越平滑
        self.smooth_alpha = 0.2


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

        model = ElasticNetCV(
            l1_ratio=self.cfg.l1_ratio_grid,
            cv=self.cfg.cv,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,   # 并行交叉验证，充分利用多核
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
