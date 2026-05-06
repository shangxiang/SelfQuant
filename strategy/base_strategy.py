import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ===========================  配置类  ===========================
class BacktestConfig:
    def __init__(self):
        # 数据
        # self.factor_cols = ['pe_standard', 'mfi_standard', 'macd_standard',
        #                     'pe_ttm_standard', 'cci_standard',
        #                     'J_standard', 'turnover_rate_x_standard']  # 你的因子列名
        self.factor_cols = [
            'pe_ttm_standard', 'pb_standard', 'dv_ttm_standard',
            'macd_standard', 'rsi_standard',
            'turnover_rate_x_standard', 'volume_ratio_standard',
            'positive_flow_standard', 'total_mv_standard',
            'macd_divergence',  # 底背离，0/1，IC正向贡献
            'macd_air_refuel',  # 空中加油，0/1
            'volatility_20d_standard',  # 波动率，可能与换手率互补
            'reversal_5d_standard'  # 短期反转，动量增强
        ]
        self.label_col = 'label'
        self.stock_col = 'ts_code'
        # 模型
        self.window = 40                     # 滚动窗口长度（交易日）
        self.l1_ratio_grid = [0.1, 0.5, 0.7, 0.9, 0.95, 1]
        self.cv = 5
        self.smooth_weights = True           # 是否EMA平滑权重
        self.smooth_alpha = 0.2
        # 选股
        self.long_quantile = 0.1             # 做多前10%
        self.short_quantile = 0.1            # 做空后10%（空头组，可关闭）
        # 输出
        self.output_path = './backtest_results/'

cfg = BacktestConfig()

# ===========================  核心回测类  ===========================
class FactorRollingBacktest:
    def __init__(self, config, load_data_func, all_dates):
        self.cfg = config
        self.load_data = load_data_func      # 传入你的 load_data(date_str) 函数
        self.all_dates = all_dates
        self.factor_cols = config.factor_cols
        self.label_col = config.label_col
        self.stock_col = config.stock_col

        # 存储结果
        self.ic_series = []                  # (date, ic)
        self.long_short_returns = []         # (date, long_ret, short_ret, spread)
        self.weights_history = {}            # date -> dict(factor: weight)
        self.scores_history = {}             # date -> DataFrame with stock_code and score
        self.daily_picks = {}                # date -> list of stock_code (long)
        self.data_cache = {}
        self.max_cache_size = (self.cfg.window + 5) * 2

    def _get_data(self, date_str):
        if date_str in self.data_cache:
            return self.data_cache[date_str]
        df = self.load_data(date_str)
        stock_df = pd.read_csv("../data/raw/stock_list/stock_list.csv")
        df = df.merge(stock_df[["ts_code", "name"]], on="ts_code", how='left')
        df = df[~df['name_y'].str.contains('ST', na=False)]
        self.data_cache[date_str] = df
        if len(self.data_cache) > self.max_cache_size:
            oldest = min(self.data_cache.keys())
            del self.data_cache[oldest]
        return df

    def _build_window_data(self, today_idx):
        """构建训练集 X, y，返回当天有效的训练样本和对应的 scaler（此处无scaler）"""
        print("构建训练集")
        X_list, y_list = [], []
        today = self.all_dates[today_idx]
        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            t_date = self.all_dates[j]
            # 标签要求 t+5 ≤ today
            if j + 5 > today_idx:
                continue
            df = self._get_data(t_date)
            # 去除任一因子或标签为 NaN 的行（标签NaN必须丢弃）
            sub = df[[self.stock_col] + self.factor_cols + [self.label_col]].copy()
            # 标签缺失必须丢弃
            sub = sub.dropna(subset=[self.label_col])
            # 因子缺失填 0
            sub[self.factor_cols] = sub[self.factor_cols].fillna(0.0)
            if sub.empty: continue
            X_list.append(sub[self.factor_cols].values)
            y_list.append(sub[self.label_col].values)

        if not X_list:
            return None, None
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    def train_model(self, X, y):
        """训练弹性网络，返回权重向量"""
        model = ElasticNetCV(
            l1_ratio=self.cfg.l1_ratio_grid,
            cv=self.cfg.cv,
            max_iter=5000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        return model.coef_

    def run(self):
        n_dates = len(self.all_dates)
        prev_weights = None
        for i in range(self.cfg.window + 5, n_dates):  # 前窗口+5天无足够标签
            today = self.all_dates[i]
            # 1. 训练
            print("开始训练")
            X, y = self._build_window_data(i)
            if X is None or len(y) < 10:
                continue
            raw_weights = self.train_model(X, y)

            # 2. 权重平滑
            print("权重平滑")
            if self.cfg.smooth_weights and prev_weights is not None:
                weights = (self.cfg.smooth_alpha * raw_weights +
                          (1 - self.cfg.smooth_alpha) * prev_weights)
            else:
                weights = raw_weights
            prev_weights = weights
            self.weights_history[today] = dict(zip(self.factor_cols, weights))

            # 3. 对当天股票打分
            print("当天股票打分")
            df_today = self._get_data(today)
            # 同样要去掉有NaN的行（因子或label）
            df_valid = df_today[[self.stock_col] + self.factor_cols + [self.label_col]].copy()
            df_valid = df_valid.dropna(subset=[self.label_col])  # 标签必须存在
            df_valid[self.factor_cols] = df_valid[self.factor_cols].fillna(0.0)  # 因子缺失填0

            if df_valid.empty:
                continue
            X_today = df_valid[self.factor_cols].values
            scores = X_today.dot(weights)
            df_valid = df_valid.copy()
            df_valid['score'] = scores
            self.scores_history[today] = df_valid[['ts_code', 'score']]

            # 4. 评估：Rank IC
            print("评估rank IC")
            ic = df_valid['score'].corr(df_valid[self.label_col], method='spearman')
            self.ic_series.append((today, ic))

            # 5. 评估：分5组多空收益
            df_valid['rank'] = df_valid['score'].rank(pct=True)  # 0~1 分位数
            df_valid['group'] = pd.cut(df_valid['rank'],
                                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                       labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            group_ret = df_valid.groupby('group')[self.label_col].mean()
            long_ret = group_ret.get('Q5', np.nan)
            short_ret = group_ret.get('Q1', np.nan)
            spread = long_ret - short_ret
            self.long_short_returns.append((today, long_ret, short_ret, spread))

            # 6. 记录每日选股（做多 Top N）
            n_long = max(1, int(len(df_valid) * self.cfg.long_quantile))
            top_stocks = df_valid.nlargest(n_long, 'score')['ts_code'].tolist()
            self.daily_picks[today] = top_stocks

        # 整理结果
        self.ic_df = pd.DataFrame(self.ic_series, columns=['date', 'ic']).set_index('date')
        self.ls_df = pd.DataFrame(self.long_short_returns,
                                  columns=['date', 'long_ret', 'short_ret', 'spread']).set_index('date')
        self.weights_df = pd.DataFrame(self.weights_history).T.sort_index()
        self.picks_dict = self.daily_picks

    def report(self):
        """打印回测摘要"""
        print("===== Rank IC 统计 =====")
        if len(self.ic_df) > 0:
            ic_mean = self.ic_df['ic'].mean()
            ic_std = self.ic_df['ic'].std()
            ir = ic_mean / ic_std if ic_std > 0 else 0
            win_rate = (self.ic_df['ic'] > 0).mean()
            print(f"IC 均值: {ic_mean:.4f}")
            print(f"IC 标准差: {ic_std:.4f}")
            print(f"IR (IC/STD): {ir:.4f}")
            print(f"IC>0 比例: {win_rate:.2%}")
        else:
            print("无IC 数据")

        print("===== 夏普比率 统计 =====")
        if len(self.ls_df) > 0:
            daily_ret = self.ls_df['spread']
            mean_ret = daily_ret.mean()
            std_ret = daily_ret.std()
            sharpe_daily = mean_ret / std_ret if std_ret > 0 else 0
            sharpe_annual = sharpe_daily * np.sqrt(252)
            print(f"\n多空组合年化夏普比率: {sharpe_annual:.4f}")

            # 如果还想看纯多头夏普
            long_ret = self.ls_df['long_ret']
            long_sharpe = (long_ret.mean() / long_ret.std()) * np.sqrt(252) if long_ret.std() > 0 else 0
            print(f"纯多头（Q5）年化夏普: {long_sharpe:.4f}")

        print("\n===== 多空收益统计 =====")
        if len(self.ls_df) > 0:
            print(self.ls_df.describe())
            cum_spread = (1 + self.ls_df['spread']).prod() - 1
            print(f"\n多空累计收益 (spread): {cum_spread:.4%}")
        else:
            print("无多空数据")

        print("\n===== 因子权重均值 =====")
        print(self.weights_df.mean().sort_values(ascending=False))

    def save_results(self):
        import os
        os.makedirs(self.cfg.output_path, exist_ok=True)
        self.ic_df.to_csv(os.path.join(self.cfg.output_path, 'ic_series.csv'))
        self.ls_df.to_csv(os.path.join(self.cfg.output_path, 'long_short_returns.csv'))
        self.weights_df.to_csv(os.path.join(self.cfg.output_path, 'factor_weights.csv'))
        # 每日选股保存为单一文件
        picks_list = []
        for date, stocks in self.daily_picks.items():
            picks_list.append(pd.DataFrame({'date': date, 'stock_code': stocks}))
        if picks_list:
            pd.concat(picks_list).to_csv(os.path.join(self.cfg.output_path, 'daily_picks.csv'), index=False)
        print(f"结果已保存至: {self.cfg.output_path}")


# ===========================  主程序  ===========================
if __name__ == "__main__":
    # ---------------- 你已有的数据和函数 ----------------
    # 交易日列表（从 trade_cal.csv 读取）
    cal_df = pd.read_csv('../data/raw/trade_cal.csv', dtype={'cal_date': str})
    all_dates = sorted(cal_df['cal_date'].tolist())
    all_dates = all_dates[:-5]

    # 你的数据加载函数（返回包含 factor_cols 和 label 的 DataFrame）
    def load_data(date_str):
        path = f"../data/section/{date_str}.csv"   # 你的文件路径
        print("读取文件：", path)
        df = pd.read_csv(path)
        # 如果需要，转换类型
        return df

    # ---------------- 创建回测实例 ----------------
    bt = FactorRollingBacktest(cfg, load_data, all_dates)
    bt.run()
    bt.report()
    bt.save_results()

    # ---------------- 可选：快速调参循环 ----------------
    # for window in [30, 60, 90]:
    #     cfg.window = window
    #     bt = FactorRollingBacktest(cfg, load_data, all_dates)
    #     bt.run()
    #     print(f"\n窗口={window}, IC均值={bt.ic_df['ic'].mean():.4f}")