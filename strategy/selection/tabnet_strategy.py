"""
基于 TabNet 的滚动窗口截面选股策略。

TabNet（Arik & Pfister, 2019）是 Google 提出的表格数据深度网络，核心是
**序贯注意力**：每一步用 sparsemax 选出一小批特征，多步累积后再输出结果。
相比树模型的两点差异：

1. 它是神经网络，对特征量纲敏感 → 默认优先使用项目已产出的 `<因子>_standard` 列
2. 训练成本远高于 LightGBM → 用 `refit_every` 控制多久重训一次，其余交易日复用模型

关于 "TabNet-boost"：这不是一个标准模型名。常见做法是把 TabNet 与 GBDT 组合，
本文件提供 `stack_gbdt` 选项——先训 TabNet，再把「原始特征 + TabNet 预测值」
喂给 LightGBM 做第二阶段拟合，等价于一种两阶段 boosting 集成。

依赖（可选，未安装时只在 fit() 时报错）：
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install pytorch-tabnet
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from strategy.selection.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


def _import_tabnet():
    """惰性导入：模块本身可以在没装 torch 的环境里被 import。"""
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TabNet 策略需要 torch 与 pytorch-tabnet，请先安装：\n"
            "    .venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "    .venv\\Scripts\\pip install pytorch-tabnet\n"
            f"（原始错误：{e}）"
        )
    return torch, TabNetRegressor


class TabNetConfig:
    """TabNet 截面选股策略超参数配置。"""

    def __init__(self):
        # ---- 数据路径 ----
        self.data_dir        = 'data/section/'
        self.calendar_file   = 'data/raw/trade_cal.csv'
        self.stock_list_file = 'data/raw/stock_list/stock_list.csv'

        # ---- 列名 ----
        self.stock_col       = 'ts_code'
        self.label_col       = 'label'
        self.label_period    = 5
        self.label_lookahead = 6

        # ---- 是否优先使用 _standard 列 ----
        # 神经网络对量纲敏感，而项目已经在 standardize 阶段产出了 z-score 列。
        # True 时若 <因子>_standard 存在就用它，否则退回原始因子列。
        self.use_standard: bool = True

        # ---- 因子列（与 LGBMConfig 保持一致，便于横向对比）----
        self.factor_cols = [
            'pe_ttm', 'pb', 'dv_ttm',
            'macd', 'cci', 'force_index_smoothed', 'net_mf_amount',
            'K', 'D', 'J', 'rsi',
            'turnover_rate_x', 'volume_ratio',
            'positive_flow', 'negative_flow',
            'total_mv', 'volatility_20d', 'reversal_5d',
            'macd_divergence', 'macd_air_refuel',
            'gross_margin', 'debt_ratio', 'roe_ttm',
            'revenue_growth_yoy', 'profit_growth_yoy', 'accruals',
            'mfi', 'vwap', 'close_to_vwap_ratio',
            'mtm_margin_balance_change', 'big_order_ratio', 'rzye',
            'size_factor', 'smb_squared', 'value_factor',
            'cma_factor', 'asset_growth_yoy', 'momentum_12_1',
            'ret_10d', 'dist_52w_high', 'close_ma20_ratio',
            'up_day_ratio_20', 'vol_price_corr_20d', 'adx',
            'turnover_amplitude_ratio', 'gap_vs_range_ratio',
            'K_chg_5d', 'K_chg_10d', 'D_chg_5d', 'D_chg_10d',
            'J_chg_5d', 'J_chg_10d', 'rsi_chg_10d',
            'macd_chg_5d', 'macd_chg_10d', 'adx_chg_5d',
            'volatility_20d_chg_5d', 'volatility_20d_chg_10d',
            'turnover_rate_x_chg_5d', 'turnover_rate_x_chg_10d',
            'reversal_5d_chg_5d', 'reversal_5d_chg_10d',
            'momentum_12_1_chg_5d', 'rzye_chg_5d', 'rzye_chg_10d',
        ]

        # ---- 过滤 ----
        self.min_mv: Optional[int] = 0
        self.filter_st: bool = True

        # ---- 滚动训练窗口（交易日数）----
        self.window: int = 60

        # ---- 每日截面抽样 ----
        # 全量截面约 3000 只 × 60 天 = 18 万样本，神经网络每轮 epoch 都要过一遍，
        # 训练成本会压到半小时一次。按天随机抽样可以在几乎不掉精度的前提下线性降本。
        # None = 不抽样（用全部股票）。
        self.subsample_per_day: Optional[int] = 1000
        self.random_state: int = 42

        # ---- 时间衰减权重（半衰期，交易日）----
        self.weight_halflife: Optional[int] = 40

        # ---- 重训频率 ----
        # 每 refit_every 个交易日重训一次，其余直接复用模型打分。
        # 设为 1 就是每日重训（2120 天回测下会非常慢）。
        self.refit_every: int = 10

        # ---- TabNet 超参数 ----
        self.n_d: int = 16            # 决策层宽度（预测分支）
        self.n_a: int = 16            # 注意力层宽度（特征选择分支）
        self.n_steps: int = 3         # 序贯注意力步数
        self.gamma: float = 1.3       # 特征复用松弛系数（1=每步不重复选特征）
        self.lambda_sparse: float = 1e-3   # 稀疏正则，鼓励少选特征
        self.momentum: float = 0.02
        self.lr: float = 2e-2

        self.max_epochs: int = 15
        self.patience: int = 5        # 早停轮数（需要验证集）

        # TabNet 的 ghost batch norm 会把每个 batch 再切成 virtual_batch_size 的微批
        # 串行跑，微批数是主要开销来源。实测同一份数据：
        #   batch=4096  virtual=256  → 2.87 s/轮
        #   batch=16384 virtual=4096 → 1.06 s/轮
        # 保持 4 个微批（保留 ghost BN 的正则效果）的同时把速度提上去。
        self.batch_size: int = 8192
        self.virtual_batch_size: int = 2048

        # 训练设备：'auto' = 有 CUDA 就用 GPU，否则 CPU；也可强制 'cpu' / 'cuda'
        self.device: str = 'auto'
        self.eval_ratio: float = 0.2  # 按时间切分的验证集比例

        # ---- 第二阶段：TabNet 预测值 + 原始特征 → LightGBM ----
        self.stack_gbdt: bool = False
        self.gbdt_params = {
            'objective': 'regression',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 15,
            'min_child_samples': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_lambda': 5.0,
            'n_jobs': -1,
            'verbose': -1,
            'random_state': 42,
        }

        # ---- Top-k 样本加权（与 LGBMStrategy 同款）----
        self.top_weight_pct: float = 0.1
        self.top_weight_factor: float = 3.0


class TabNetStrategy(BaseStrategy):
    """
    基于 TabNet 的滚动窗口截面选股策略。

    与 LGBMStrategy 的关系：
      - 训练样本构造、时间衰减权重、top-k 加权完全沿用同一套逻辑，
        保证换模型时改动只发生在 fit() 内部，便于横向对比。
      - 标签同样做截面 rank 归一化到 [-0.5, 0.5]。
      - 打分为模型预测值，越高越好；引擎取 top_n 建仓。

    成本控制：
      每个交易日都重训神经网络在 2120 天上不现实，因此用 refit_every 控制重训频率，
      其余交易日直接复用上一次的模型打分。
    """

    def __init__(self, config: TabNetConfig, data_loader):
        self.cfg    = config
        self.loader = data_loader

        self._model = None          # TabNetRegressor
        self._stack_model = None    # 第二阶段的 LightGBM
        self._last_fit_idx: Optional[int] = None
        self._feature_names: Optional[list] = None
        self._importances: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._model       = None
        self._stack_model = None
        self._last_fit_idx = None
        self._importances  = None

    # ------------------------------------------------------------------
    # 内部：特征列解析
    # ------------------------------------------------------------------

    def _resolve_cols(self, df: pd.DataFrame) -> list:
        """
        确定实际参与训练/推理的特征列名。

        use_standard=True 时优先取 `<因子>_standard`（z-score 后的列），
        缺失时退回原始因子列。截面数据列结构一致，每次重新解析即可，
        不做缓存以免首日数据为空时把空列集合固化下来。
        """
        if df is None or not len(df):
            return []
        cols = []
        for fc in self.cfg.factor_cols:
            if self.cfg.use_standard and f'{fc}_standard' in df.columns:
                cols.append(f'{fc}_standard')
            elif fc in df.columns:
                cols.append(fc)
        return cols

    @staticmethod
    def _matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
        X = np.empty((len(df), len(cols)), dtype=np.float64)
        for i, c in enumerate(cols):
            X[:, i] = pd.to_numeric(df[c], errors='coerce').to_numpy()
        # TabNet 不接受 NaN / inf
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 内部：构建训练集
    # ------------------------------------------------------------------

    def _build_train_data(self, today_idx: int):
        """构造截面 rank 归一化训练集，返回 (X, y, dist, rank_w) 或 (None,)*4。"""
        all_dates = self.loader.get_trading_dates()
        X_list, y_list, dist_list, w_list = [], [], [], []

        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            if j + self.cfg.label_lookahead > today_idx:
                continue

            df = self.loader.get_data(all_dates[j])
            if df is None or self.cfg.label_col not in df.columns:
                continue
            cols = self._resolve_cols(df)
            if not cols:
                continue
            self._feature_names = cols

            # 在整个截面上做 rank 归一化，再一次性过滤缺失行。
            # 注意不能用 dropna().reset_index() 后回 df.loc 取行 —— 那会取到错位的数据。
            X_full = self._matrix(df, cols)
            raw = pd.to_numeric(df[self.cfg.label_col], errors='coerce').to_numpy()
            valid = ~np.isnan(raw)
            if valid.sum() < 10:
                continue

            y_full = np.full(len(df), np.nan, dtype=np.float64)
            y_full[valid] = pd.Series(raw[valid]).rank(pct=True).to_numpy() - 0.5
            mask = ~np.isnan(y_full)

            idx = np.where(mask)[0]
            if self.cfg.subsample_per_day and len(idx) > self.cfg.subsample_per_day:
                rng = np.random.default_rng(self.cfg.random_state + j)
                idx = np.sort(rng.choice(idx, self.cfg.subsample_per_day, replace=False))

            yy = y_full[idx]
            X_list.append(X_full[idx])
            y_list.append(yy)
            dist_list.append(np.full(len(idx), today_idx - j, dtype=np.float64))

            if self.cfg.top_weight_factor > 1.0:
                thr = np.quantile(yy, 1.0 - self.cfg.top_weight_pct)
                w_list.append(np.where(yy >= thr, self.cfg.top_weight_factor, 1.0))
            else:
                w_list.append(np.ones(len(idx), dtype=np.float64))

        if not X_list:
            return None, None, None, None
        return (np.vstack(X_list), np.concatenate(y_list),
                np.concatenate(dist_list), np.concatenate(w_list))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, date_str: str) -> bool:
        """
        以 date_str 为基准日训练 TabNet。

        距上次训练不足 refit_every 个交易日时直接复用旧模型并返回 True。

        Returns
        -------
        bool  True = 已有可用模型，False = 非交易日或样本不足
        """
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            return False
        today_idx = all_dates.index(date_str)

        # 复用旧模型（神经网络重训很贵，这是主要的性能开关）
        if (self._model is not None and self._last_fit_idx is not None
                and today_idx - self._last_fit_idx < self.cfg.refit_every):
            return True

        X, y, dist, rank_w = self._build_train_data(today_idx)
        if X is None or len(y) < 200:
            return False

        if self.cfg.weight_halflife is not None:
            sample_weight = np.power(2.0, -dist / self.cfg.weight_halflife) * rank_w
        else:
            sample_weight = rank_w

        torch, TabNetRegressor = _import_tabnet()
        device = self.cfg.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            print('  [TabNet] 未检测到可用 CUDA（装的是 CPU 版 torch？），回退到 CPU')
            device = 'cpu'

        # 按时间切分验证集：行是按日期升序堆叠的，取最后一段做验证
        n = len(y)
        n_val = max(1, int(n * self.cfg.eval_ratio))
        X_tr, y_tr, w_tr = X[:-n_val], y[:-n_val], sample_weight[:-n_val]
        X_va, y_va = X[-n_val:], y[-n_val:]

        model = TabNetRegressor(
            n_d=self.cfg.n_d,
            n_a=self.cfg.n_a,
            n_steps=self.cfg.n_steps,
            gamma=self.cfg.gamma,
            lambda_sparse=self.cfg.lambda_sparse,
            momentum=self.cfg.momentum,
            optimizer_params={'lr': self.cfg.lr},
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            device_name=device,
            verbose=0,
        )

        model.fit(
            X_train=X_tr.astype(np.float32),
            y_train=y_tr.reshape(-1, 1).astype(np.float32),
            eval_set=[(X_va.astype(np.float32), y_va.reshape(-1, 1).astype(np.float32))],
            eval_name=['valid'],
            eval_metric=['rmse'],
            max_epochs=self.cfg.max_epochs,
            patience=self.cfg.patience,
            batch_size=self.cfg.batch_size,
            virtual_batch_size=self.cfg.virtual_batch_size,
            weights=w_tr.astype(np.float32),
            drop_last=False,
            compute_importance=True,   # 不开启的话 feature_importances_ 取不到
        )

        self._model = model
        self._last_fit_idx = today_idx

        # 第二阶段：TabNet 预测值作为额外特征喂给 LightGBM
        self._stack_model = None
        if self.cfg.stack_gbdt:
            import lightgbm as lgb
            emb_tr = model.predict(X_tr.astype(np.float32)).reshape(-1, 1)
            stack_X_tr = np.hstack([X_tr, emb_tr])
            self._stack_model = lgb.LGBMRegressor(**self.cfg.gbdt_params)
            self._stack_model.fit(stack_X_tr, y_tr, sample_weight=w_tr)

        # 特征重要度（TabNet 由注意力掩码聚合而来），取不到就跳过
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
            self._feature_names = self._feature_names or []
            if len(imp) == len(self._feature_names):
                self._importances = imp if self._importances is None else 0.3 * imp + 0.7 * self._importances
        except Exception:
            pass

        return True

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------

    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        用当前模型对 date_str 截面内所有股票打分，按 score 降序返回。
        """
        if self._model is None:
            return None

        df = self.loader.get_data(date_str)
        if df is None or self.cfg.stock_col not in df.columns:
            return None

        cols = self._resolve_cols(df)
        if not cols:
            return None

        X = self._matrix(df, cols).astype(np.float32)
        scores = self._model.predict(X).reshape(-1)

        if self._stack_model is not None:
            scores = self._stack_model.predict(np.hstack([X, scores.reshape(-1, 1)])).reshape(-1)

        result = pd.DataFrame({
            self.cfg.stock_col: df[self.cfg.stock_col].to_numpy(),
            'score': scores,
        })
        return result.sort_values('score', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 纯信号评估
    # ------------------------------------------------------------------

    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        纯信号质量评估：逐日打分，统计 Rank IC 与 Top-N 组合收益。

        返回 dict 含 ic_df / ls_df / topn_df / weights_df，
        可直接交给 BaseStrategy.report_dump() 落盘。
        """
        all_dates = self.loader.get_trading_dates()
        if start_date not in all_dates or end_date not in all_dates:
            return {}
        i0 = all_dates.index(start_date)
        i1 = all_dates.index(end_date)

        ic_rows, topn_rows = [], []
        for i in range(i0, i1 + 1):
            today = all_dates[i]
            if not self.fit(today):
                continue
            sig = self.generate_signals(today)
            if sig is None or sig.empty:
                continue

            df = self.loader.get_data(today)
            if df is None or self.cfg.label_col not in df.columns:
                continue

            merged = sig.merge(df[[self.cfg.stock_col, self.cfg.label_col]],
                               on=self.cfg.stock_col, how='inner').dropna()
            if len(merged) < 20:
                continue

            ic = merged['score'].corr(merged[self.cfg.label_col], method='spearman')
            ic_rows.append((today, ic))

            ranked = merged.sort_values('score', ascending=False)
            row = {'date': today}
            for k in (10, 50, 100):
                top = ranked.head(min(k, len(ranked)))
                row[f'top{k}'] = float(top[self.cfg.label_col].mean())
            row['mkt'] = float(merged[self.cfg.label_col].mean())
            topn_rows.append(row)

        if not ic_rows:
            return {}

        ic_df = pd.DataFrame(ic_rows, columns=['date', 'ic']).set_index('date')
        topn_df = pd.DataFrame(topn_rows).set_index('date')
        ls_df = pd.DataFrame({
            'spread': topn_df['top10'] - topn_df['mkt']
        }) if 'top10' in topn_df.columns else pd.DataFrame()

        weights_df = pd.DataFrame()
        if self._importances is not None and self._feature_names:
            weights_df = pd.DataFrame([self._importances], columns=self._feature_names)

        return {'ic_df': ic_df, 'ls_df': ls_df, 'topn_df': topn_df, 'weights_df': weights_df}
