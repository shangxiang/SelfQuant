import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt


# ==================== 配置容器 ====================
class Config:
    def __init__(self):
        # 数据路径
        self.data_dir = '../data/section/'
        self.calendar_file = '../data/raw/trade_cal.csv'
        self.stock_col = 'ts_code'
        self.label_col = 'label'
        # 因子集合（可随时修改）
        # self.factor_cols = [
        #     'pe_ttm_standard', 'pb_standard', 'dv_ttm_standard',
        #     'macd_standard', 'rsi_standard',
        #     'turnover_rate_x_standard', 'volume_ratio_standard',
        #     'positive_flow_standard', 'total_mv_standard', 'macd_air_refuel',
        #     'macd_divergence', 'big_order_ratio_standard', 'lhb_strength_5d_standard',
        #     'vol_breakout', 'volatility_20d_standard', 'reversal_5d_standard',
        #     'high_low_spread_standard', 'gross_margin_standard', 'debt_ratio_standard',
        #     'roe_ttm_standard', 'revenue_growth_yoy_standard', 'profit_growth_yoy_standard',
        #     'accruals_standard'
        # ]

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
        # 模型参数
        self.window = 40               # 滚动窗口（交易日）
        self.l1_ratio_grid = [0.1, 0.5, 0.7, 0.9, 0.95, 1]
        self.cv = 5
        self.smooth_weights = True
        self.smooth_alpha = 0.2
        # 选股比例（用于每日信号）
        self.long_quantile = 0.1        # 前10%
        self.top_n = 100
        # 交易模拟参数
        self.commission = 0.0001        # 万分之一（单边）
        self.holding_period = 5         # 持有5个交易日
        self.initial_capital = 1_000_000

        # 风控参数
        # 状态参数（各状态下可以独立配置）
        self.state_params = {
            'bull': {
                'use_index_filter': False,  # 已经在状态中过滤，无需再过滤
                'use_vol_control': False,  # 保留波动率降仓，控制回撤
                'target_vol': 0.15,
                'vol_window': 20,
                'top_n': 100
            },
            'bear': {
                # 'use_index_filter': True,
                # 'force_empty': True,  # 直接空仓
                # 'top_n': 0
                'use_index_filter': False,  # 已经在状态中过滤，无需再过滤
                'use_vol_control': False,  # 保留波动率降仓，控制回撤
                'target_vol': 0.15,
                'vol_window': 20,
                'top_n': 100
            }
            # 无震荡状态
        }
        # 状态判断用固定参数（不随状态变）
        self.index_file = '../data/raw/index_daily/000905.SH.csv'  # 中证500指数日线
        self.state_ma_period = 60  # 固定用60日均线区分牛熊
        self.state_vol_window = 20  # 近期波动率窗口
        self.state_vol_long_window = 252  # 长期波动率参考窗





# ==================== 数据加载与缓存 ====================
class DataLoader:
    def __init__(self, config):
        self.cfg = config
        self.cache = {}
        # 交易日历
        cal = pd.read_csv(self.cfg.calendar_file, dtype={'cal_date': str})
        self.all_dates = sorted(cal['cal_date'].dropna().str.strip().tolist())
        self.all_dates = [d for d in self.all_dates if d >= '20200101']  # 按需过滤

    def get_data(self, date_str):
        """缓存读取"""
        if date_str in self.cache:
            return self.cache[date_str]
        path = os.path.join(self.cfg.data_dir, f"{date_str}.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        stock_df = pd.read_csv("../data/raw/stock_list/stock_list.csv")
        df = df.merge(stock_df[["ts_code", "name"]], on="ts_code", how='left')
        # columns = df.columns.tolist()
        # print(columns)
        df = df[~df['name_y'].str.contains('ST', na=False)]
        # 确保列存在
        needed = [self.cfg.stock_col] + self.cfg.factor_cols + [self.cfg.label_col]
        for c in needed:
            if c not in df.columns:
                return None
        # 统一类型
        df[self.cfg.factor_cols] = df[self.cfg.factor_cols].astype(float)
        df[self.cfg.label_col] = df[self.cfg.label_col].astype(float)
        self.cache[date_str] = df
        return df

    def get_trading_dates(self):
        return self.all_dates


# ==================== 因子模型训练器（核心） ====================
class FactorModel:
    def __init__(self, config, data_loader):
        self.cfg = config
        self.loader = data_loader
        self.prev_weights = None

    def _build_train_data(self, today_idx):
        """利用过去窗口构建训练集（标签需在今日前确认）"""
        all_dates = self.loader.get_trading_dates()
        X_list, y_list = [], []
        for j in range(max(0, today_idx - self.cfg.window + 1), today_idx + 1):
            t_date = all_dates[j]
            if j + 5 > today_idx:        # 标签未确认，跳过
                continue
            df = self.loader.get_data(t_date)
            if df is None:
                continue
            # 处理缺失：标签缺失丢弃，因子缺失填0
            df = df[[self.cfg.stock_col] + self.cfg.factor_cols + [self.cfg.label_col]].copy()
            df = df.dropna(subset=[self.cfg.label_col])
            if df.empty:
                continue
            df[self.cfg.factor_cols] = df[self.cfg.factor_cols].fillna(0.0)
            X_list.append(df[self.cfg.factor_cols].values)
            y_list.append(df[self.cfg.label_col].values)
        if not X_list:
            return None, None
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    def train(self, today_idx):
        """训练并返回权重，支持平滑"""
        X, y = self._build_train_data(today_idx)
        if X is None or len(y) < 30:
            return None
        model = ElasticNetCV(
            l1_ratio=self.cfg.l1_ratio_grid,
            cv=self.cfg.cv,
            max_iter=5000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        raw_weights = model.coef_
        if self.cfg.smooth_weights and self.prev_weights is not None:
            weights = (self.cfg.smooth_alpha * raw_weights +
                       (1 - self.cfg.smooth_alpha) * self.prev_weights)
        else:
            weights = raw_weights
        self.prev_weights = weights
        return weights

    def predict_one_day(self, date_str):
        """单日预测：返回当天买入股票列表及其真实收益"""
        all_dates = self.loader.get_trading_dates()
        if date_str not in all_dates:
            raise ValueError(f"日期 {date_str} 不在交易日历中")
        today_idx = all_dates.index(date_str)
        weights = self.train(today_idx)
        if weights is None:
            return None, None, None

        df_today = self.loader.get_data(date_str)
        if df_today is None:
            return None, None, None
        df_valid = df_today[[self.cfg.stock_col] + self.cfg.factor_cols + [self.cfg.label_col]].copy()
        df_valid = df_valid.dropna(subset=[self.cfg.label_col])
        if df_valid.empty:
            return None, None, None
        df_valid[self.cfg.factor_cols] = df_valid[self.cfg.factor_cols].fillna(0.0)
        X_today = df_valid[self.cfg.factor_cols].values
        scores = X_today.dot(weights)
        df_valid['score'] = scores
        # 选top_n
        #n_long = max(1, int(len(df_valid) * self.cfg.long_quantile))
        n_long = min(self.cfg.top_n, len(df_valid))
        top_df = df_valid.nlargest(n_long, 'score')
        picks = top_df[self.cfg.stock_col].tolist()
        avg_return = top_df[self.cfg.label_col].mean()
        return picks, avg_return, weights

    def get_weights(self, date_str):
        """仅获取权重，用于外部展示"""
        all_dates = self.loader.get_trading_dates()
        today_idx = all_dates.index(date_str)
        return self.train(today_idx)


# ==================== 原有因子有效性回测（保留） ====================
class FactorBacktest:
    """与之前完全相同的功能，仅重构命名，输出 IC/多空/夏普"""
    def __init__(self, config, data_loader):
        self.cfg = config
        self.loader = data_loader
        self.model = FactorModel(config, data_loader)
        self.ic_series = []
        self.long_short_returns = []
        self.weights_history = {}
        self.scores_history = {}
        self.daily_picks = {}

    def run(self, start_date=None, end_date=None):
        all_dates = self.loader.get_trading_dates()
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        # 需要保证最前有足够窗口且尾部可评估（至少5天）
        start_idx = self.cfg.window + 5
        end_idx = len(all_dates) - 5
        for i in range(start_idx, end_idx):
            today = all_dates[i]
            weights = self.model.train(i)
            if weights is None:
                continue
            self.weights_history[today] = dict(zip(self.cfg.factor_cols, weights))
            df_today = self.loader.get_data(today)
            if df_today is None:
                continue
            df_valid = df_today[[self.cfg.stock_col] + self.cfg.factor_cols + [self.cfg.label_col]].copy()
            df_valid = df_valid.dropna(subset=[self.cfg.label_col])
            if df_valid.empty:
                continue
            df_valid[self.cfg.factor_cols] = df_valid[self.cfg.factor_cols].fillna(0.0)
            X_today = df_valid[self.cfg.factor_cols].values
            scores = X_today.dot(weights)
            df_valid['score'] = scores
            self.scores_history[today] = df_valid[['ts_code', 'score']]
            # IC
            ic = df_valid['score'].corr(df_valid[self.cfg.label_col], method='spearman')
            self.ic_series.append((today, ic))
            # 分层
            df_valid['rank'] = df_valid['score'].rank(pct=True)
            df_valid['group'] = pd.cut(df_valid['rank'],
                                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                       labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            group_ret = df_valid.groupby('group')[self.cfg.label_col].mean()
            long_ret = group_ret.get('Q5', np.nan)
            short_ret = group_ret.get('Q1', np.nan)
            spread = long_ret - short_ret
            self.long_short_returns.append((today, long_ret, short_ret, spread))
            # 选股记录
            #n_long = max(1, int(len(df_valid) * self.cfg.long_quantile))
            n_long = min(self.cfg.top_n, len(df_valid))
            self.daily_picks[today] = df_valid.nlargest(n_long, 'score')['ts_code'].tolist()
        # 整理为DataFrame
        self.ic_df = pd.DataFrame(self.ic_series, columns=['date', 'ic']).set_index('date')
        self.ls_df = pd.DataFrame(self.long_short_returns,
                                  columns=['date', 'long_ret', 'short_ret', 'spread']).set_index('date')
        self.weights_df = pd.DataFrame(self.weights_history).T.sort_index()

    def report(self):
        print("===== Rank IC 统计 =====")
        print(f"IC 均值: {self.ic_df['ic'].mean():.4f}")
        print(f"IC 标准差: {self.ic_df['ic'].std():.4f}")
        print(f"IR: {self.ic_df['ic'].mean()/self.ic_df['ic'].std():.4f}")
        print(f"IC>0 比例: {(self.ic_df['ic']>0).mean():.2%}")
        print("\n===== 多空收益统计 =====")
        print(self.ls_df.describe())
        cum_spread = (1+self.ls_df['spread']).prod()-1
        print(f"多空累计收益: {cum_spread:.4%}")
        sharpe = (self.ls_df['spread'].mean() / self.ls_df['spread'].std()) * np.sqrt(252)
        print(f"年化夏普: {sharpe:.4f}")
        print("\n===== 因子权重均值 =====")
        print(self.weights_df.mean().sort_values(ascending=False))


# ==================== 真实交易模拟器（新增） ====================
class TradingSimulator:
    def __init__(self, config, data_loader):
        self.cfg = config
        self.loader = data_loader
        self.model = FactorModel(config, data_loader)
        self.price_col = 'close_x'   # 使用你的日收盘价列

    def _compute_state_map(self):
        index_df = pd.read_csv(self.cfg.index_file, parse_dates=['trade_date'])
        index_df = index_df.sort_values('trade_date')
        close = index_df['close']
        ma60 = close.rolling(60).mean()
        # 昨日收盘 > 60日均线 → 牛市；否则熊市
        bull = close.shift(1) > ma60.shift(1)
        state = pd.Series('bear', index=index_df.index)
        state[bull] = 'bull'
        # 转换为日期字典
        state_map = dict(zip(index_df['trade_date'].dt.strftime('%Y%m%d'), state))
        bull_map = {k: v == 'bull' for k, v in state_map.items()}
        return state_map, bull_map

    def run(self, start_date, end_date, capital=None):
        if capital is None:
            capital = self.cfg.initial_capital
        all_dates = self.loader.get_trading_dates()
        start_idx = all_dates.index(start_date)
        end_idx = all_dates.index(end_date)
        trade_dates = all_dates[start_idx:end_idx + 1]

        # ---------- 统一的状态和牛熊映射 ----------
        state_map, bull_map = self._compute_state_map()

        cash = capital
        positions = {}
        daily_nav = []
        trade_log = []
        cycle = self.cfg.holding_period

        for i, today in enumerate(trade_dates):
            # 获取当日市场状态及对应参数
            market_state = state_map.get(today, 'volatile')
            params = self.cfg.state_params[market_state]

            use_index_filter = params.get('use_index_filter', True)
            use_vol_control = params.get('use_vol_control', True)
            target_vol = params.get('target_vol', 0.15)
            vol_window = params.get('vol_window', 20)
            force_empty = params.get('force_empty', False)
            top_n = params.get('top_n', self.cfg.top_n)

            df_today = self.loader.get_data(today)
            if df_today is None:
                if daily_nav:
                    daily_nav.append((today, daily_nav[-1][1]))
                continue

            # 1. 牛熊判断（统一来源，无重复读取）
            if force_empty:
                in_bull = False
            elif use_index_filter:
                in_bull = bull_map.get(today, True)
            else:
                in_bull = True

            # 2. 处理卖出（到期 + 风控清仓）
            to_sell = []
            for stock, pos in positions.items():
                buy_idx = all_dates.index(pos['buy_date'])
                if buy_idx + cycle <= all_dates.index(today) or not in_bull:
                    to_sell.append(stock)

            for stock in to_sell:
                pos = positions.pop(stock)
                row = df_today[df_today[self.cfg.stock_col] == stock]
                if row.empty:
                    # 无今日价格，延后处理
                    positions[stock] = pos
                    continue
                buy_idx = all_dates.index(pos['buy_date'])  # 重新计算，避免变量错乱
                # 风控清仓（未到期但 in_bull 为 False）用开盘价，正常到期用收盘价
                if not in_bull and (buy_idx + cycle > all_dates.index(today)):
                    sell_price = row['open'].iloc[0]
                else:
                    sell_price = row[self.price_col].iloc[0]
                proceeds = pos['shares'] * sell_price * (1 - self.cfg.commission)
                cash += proceeds
                trade_log.append((today, 'SELL', stock, pos['shares'], sell_price, proceeds))

            # 3. 估值
            holdings_value = 0.0
            for stock, pos in positions.items():
                row = df_today[df_today[self.cfg.stock_col] == stock]
                if not row.empty:
                    price = row[self.price_col].iloc[0]
                    holdings_value += pos['shares'] * price
                else:
                    holdings_value += pos['shares'] * pos['buy_price']
            total_value = cash + holdings_value
            daily_nav.append((today, total_value))

            # 4. 波动率降仓比例
            vol_ratio = 1.0
            if use_vol_control and len(daily_nav) >= vol_window:
                nav_series = pd.Series([v[1] for v in daily_nav])
                daily_ret = nav_series.pct_change().dropna().iloc[-vol_window:]
                realized_vol = daily_ret.std() * np.sqrt(252)
                if realized_vol > 0:
                    vol_ratio = min(1.0, target_vol / realized_vol)
            invest_cash = cash * vol_ratio

            # 5. 信号日买入（仅当牛市且未强制空仓）
            if (i + cycle) < len(trade_dates) and ((i + start_idx) % cycle == 0) and in_bull:
                weights = self.model.train(all_dates.index(today))
                if weights is None:
                    continue
                # 准备有效股票池
                df_valid = df_today[[self.cfg.stock_col] + self.cfg.factor_cols +
                                    [self.cfg.label_col, self.price_col]].copy()
                df_valid = df_valid.dropna(subset=[self.cfg.label_col, self.price_col])
                if df_valid.empty:
                    continue
                df_valid[self.cfg.factor_cols] = df_valid[self.cfg.factor_cols].fillna(0.0)
                scores = df_valid[self.cfg.factor_cols].values.dot(weights)
                df_valid['score'] = scores
                n_long = min(top_n, len(df_valid))
                top_df = df_valid.nlargest(n_long, 'score')
                buy_cash_per_stock = invest_cash / n_long
                for _, row_stock in top_df.iterrows():
                    stock = row_stock[self.cfg.stock_col]
                    buy_price = row_stock[self.price_col]
                    shares = int(buy_cash_per_stock / buy_price)
                    if shares <= 0:
                        continue
                    cost = shares * buy_price * (1 + self.cfg.commission)
                    cash -= cost
                    positions[stock] = {'buy_date': today, 'buy_price': buy_price, 'shares': shares}
                    trade_log.append((today, 'BUY', stock, shares, buy_price, cost))

                # 买入后更新当日估值
                holdings_value = 0.0
                for stock, pos in positions.items():
                    row = df_today[df_today[self.cfg.stock_col] == stock]
                    if not row.empty:
                        price = row[self.price_col].iloc[0]
                        holdings_value += pos['shares'] * price
                    else:
                        holdings_value += pos['shares'] * pos['buy_price']
                total_value = cash + holdings_value
                daily_nav[-1] = (today, total_value)

        # ---------- 回测结束：强制卖出所有余仓 ----------
        if positions:
            last_date = trade_dates[-1]
            df_last = self.loader.get_data(last_date)
            if df_last is not None:
                for stock, pos in positions.items():
                    row = df_last[df_last[self.cfg.stock_col] == stock]
                    if not row.empty:
                        sell_price = row[self.price_col].iloc[0]
                        cash += pos['shares'] * sell_price * (1 - self.cfg.commission)
                positions.clear()
                daily_nav[-1] = (last_date, cash)

        # ---------- 4. 绩效统计 ----------
        nav_df = pd.DataFrame(daily_nav, columns=['date', 'nav'])
        nav_df['daily_return'] = nav_df['nav'].pct_change()
        mean_ret = nav_df['daily_return'].mean()
        std_ret = nav_df['daily_return'].std()
        sharpe_annual = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
        win_rate = (nav_df['daily_return'] > 0).mean()
        total_return = (nav_df['nav'].iloc[-1] / capital) - 1
        nav_df['cummax'] = nav_df['nav'].cummax()
        nav_df['drawdown'] = (nav_df['nav'] - nav_df['cummax']) / nav_df['cummax']
        max_drawdown = nav_df['drawdown'].min()
        trading_days = len(nav_df)
        ann_return = (nav_df['nav'].iloc[-1] / capital) ** (252 / trading_days) - 1

        print("========== 真实交易回测报告 ==========")
        print(f"初始资金: {capital:,.0f}")
        print(f"最终资金: {nav_df['nav'].iloc[-1]:,.0f}")
        print(f"总收益率: {total_return:.4%}")
        print(f"年化收益率: {ann_return:.4%}")
        print(f"年化夏普比率: {sharpe_annual:.4f}")
        print(f"最大回撤: {max_drawdown:.4%}")
        print(f"日胜率: {win_rate:.2%}")
        print(f"交易记录数: {len(trade_log)}")
        state_counts = pd.Series([state_map.get(d, 'volatile') for d in trade_dates]).value_counts()
        print(f"状态统计：", state_counts)

        return nav_df, trade_log


if __name__ == '__main__':

    cfg = Config()
    # 创建数据加载器与模拟器
    loader = DataLoader(cfg)


    # 单日预测
    # model = FactorModel(cfg, loader)

    # picks, avg_ret, weights = model.predict_one_day('20240930')
    # print("买入股票:", picks)
    # print("预计平均5日收益:", avg_ret)

    # 全量回归（因子评估）
    bt = FactorBacktest(cfg, loader)
    bt.run(start_date='20230101', end_date='20260331')
    bt.report()

    # 实战回测
    sim = TradingSimulator(cfg, loader)
    nav_df, trades = sim.run('20230103', '20260331', capital=1_000_000)

    # 查看净值曲线
    # nav_df.set_index('date')['nav'].plot(title='策略净值')