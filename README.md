# SelfQuant

A 股量化研究框架，覆盖数据拉取、因子计算、截面标准化到模拟交易回测的完整流程。

---

## 目录

- [项目结构](#项目结构)
- [整体流程](#整体流程)
- [模块说明](#模块说明)
  - [数据层](#数据层)
  - [因子层](#因子层)
  - [选股策略](#选股策略)
  - [卖出策略](#卖出策略)
  - [择时策略](#择时策略)
  - [回测引擎](#回测引擎)
- [快速开始](#快速开始)
- [开发新内容](#开发新内容)
  - [新增选股策略](#新增选股策略)
  - [新增卖出策略](#新增卖出策略)
  - [新增择时策略](#新增择时策略)

---

## 项目结构

```
SelfQuant/
├── run_backtest.py                        # 主入口，从根目录执行
│
├── data/                                  # 本地数据（不入 git）
│   ├── raw/
│   │   ├── trade_cal.csv                  # 交易日历
│   │   ├── stock_list/stock_list.csv      # 股票列表（含行业信息）
│   │   ├── stock_data/                    # 个股日线行情
│   │   ├── daily_basic_data/              # 每日基本面快照（PE/PB/市值等）
│   │   ├── moneyflow/                     # 大中小单资金流
│   │   ├── margin_detail/                 # 融资融券明细
│   │   ├── top_list/                      # 龙虎榜
│   │   ├── income/                        # 利润表
│   │   ├── balancesheet/                  # 资产负债表
│   │   ├── cashflow/                      # 现金流量表
│   │   └── index_daily/000905.SH.csv      # 中证500日线（择时用）
│   ├── series/                            # 按股票存储的时序数据 <ts_code>.csv
│   ├── section/                           # 每日截面数据（含标准化因子）YYYYMMDD.csv
│   └── financial.csv                      # 合并后的财务数据宽表
│
├── data_api/                              # 数据源接口（Tushare 封装）
├── data_local_dump.py                     # 从 Tushare 拉取原始数据到本地
├── merge_data.py                          # 合并行情/财务数据，生成 data/series/
├── factor/
│   └── new_factor_manager.py              # 因子计算：技术因子 + FF 风格因子 + 交叉因子
├── tools.py                               # 截面转换 + 去极值 + 中性化 + 标准化
│
├── strategy/                              # 策略层
│   ├── data_loader.py                     # 截面数据加载器（带缓存）
│   ├── selection/                         # 截面选股策略
│   │   ├── base_strategy.py               # 抽象基类 BaseStrategy
│   │   └── elastic_net_strategy.py        # 弹性网络策略（含因子评估）
│   └── sell/                              # 卖出策略
│       ├── base_sell_strategy.py          # 抽象基类 BaseSellStrategy
│       └── hold_n_days.py                 # 持有 N 日清仓策略
│
└── backtest/                              # 回测层
    ├── config.py                          # BacktestConfig 参数配置
    ├── timing.py                          # 择时策略（抽象基类 + MA 均线择时）
    └── engine.py                          # BacktestEngine 回测引擎
```

---

## 整体流程

```
① 数据拉取
data_local_dump.py  →  data/raw/

② 数据合并
merge_data.py  →  data/series/<ts_code>.csv  +  data/financial.csv

③ 因子计算
factor/new_factor_manager.py  →  data/series/<ts_code>.csv（写入因子列）

④ 截面化 + 标准化
tools.py: series_to_section()  →  data/section/YYYYMMDD.csv
tools.py: standardize()        →  data/section/YYYYMMDD.csv（写入 _standard 列）

⑤ 策略 + 回测
DataLoader → BaseStrategy.fit() + generate_signals() → BacktestEngine.run()
```

**关键设计原则：**
- 策略只负责"打分"，不感知资金和仓位
- 回测引擎只负责"交易模拟"，不内置任何策略逻辑
- 三类策略（选股 / 择时 / 卖出）均通过依赖注入，可自由组合替换
- 信号与成交错开一日（T 日收盘生成信号，T+1 日收盘成交），避免未来函数

---

## 模块说明

### 数据层

**`data_local_dump.py` — `DownloadData`**

从 Tushare 拉取所有原始数据到 `data/raw/`，支持断点续传（已存在的文件跳过）。

```bash
python data_local_dump.py
```

执行顺序：交易日历 → 股票列表 → 指数基础信息 → 按日期数据（行情快照/资金流/融资融券/龙虎榜）→ 按股票数据（行情/财务三表）→ 指数日线。

**`merge_data.py`**

- `financial_data_preprocess()`：合并财务三表，利润表/现金流量表做累计→单季度转换，输出 `data/financial.csv`
- `merge_basic_daily_data()`：将行情、基本面快照、融资融券、资金流、龙虎榜按日期合并为宽表
- `split_to_series_section()`：将宽表按股票代码拆分，输出 `data/series/<ts_code>.csv`

**`strategy/data_loader.py` — `DataLoader`**

统一的截面数据入口，负责读取每日截面 CSV、过滤 ST 股、内存缓存。

```python
loader = DataLoader(config)
df = loader.get_data('20240101')     # 返回当日截面 DataFrame（含因子列和 label）
dates = loader.get_trading_dates()   # 返回完整交易日历列表 ['20200104', ...]
```

---

### 因子层

**`factor/new_factor_manager.py` — `FactorManager`**

逐只股票计算因子，结果写回 `data/series/<ts_code>.csv`。支持增量重跑（幂等）。

因子分三大类（共约 50+ 列，下游使用 `_standard` 后缀版本）：

**技术/量价因子（基于日线行情）**

| 因子 | 说明 |
|---|---|
| `macd` / `rsi` | MACD、RSI |
| `macd_divergence` | MACD 底背离信号（0/1） |
| `macd_air_refuel` | MACD 零轴附近缩量信号（0/1） |
| `vol_breakout` | 成交量突破信号（0/1） |
| `volatility_20d` | 20 日历史波动率 |
| `reversal_5d` | 5 日反转因子 |
| `high_low_spread` | 5 日振幅 |
| `vwap` | 成交量加权均价 |
| `force_index` | 力量指数（EMA 平滑） |
| `big_order_ratio` | 大单占比 |
| `mfi` | 资金流量指标 |
| `mtm_margin_balance_change` | 融资余额变化×动量 |
| `lhb_strength_5d` | 5 日龙虎榜净买力度 |
| `label` / `label_10` / `label_25` | 未来 5 / 10 / 25 日收益率（训练标签） |

**Fama-French 风格因子（股票截面）**

| 因子 | 对应 FF 因子 | 说明 |
|---|---|---|
| `size_factor` | SMB | `-ln(流通市值)`，规模因子 |
| `value_factor` | HML | `1/PB`，价值因子 |
| `cma_factor` | CMA | 资产增速取反，保守投资因子 |
| `momentum_12_1` | Mom | 12-1 月动量 |
| `roe_ttm` | RMW | ROE（TTM），盈利因子代理 |

**基本面因子（来自财务三表）**

| 因子 | 说明 |
|---|---|
| `gross_margin` | 毛利率 |
| `debt_ratio` | 资产负债率 |
| `roe_ttm` | ROE（TTM） |
| `revenue_growth_yoy` | 营收同比增速 |
| `profit_growth_yoy` | 净利润同比增速 |
| `accruals` | 应计项目（盈利质量） |
| `asset_growth_yoy` | 总资产增速（CMA 代理） |

**高阶交叉因子**

| 因子 | 构造方式 | 经济含义 |
|---|---|---|
| `smb_squared` | `size_factor²` | 非线性规模效应 |
| `smb_mom` | `size_factor × momentum_12_1` | 小盘股动量溢价 |
| `smb_squared_mom` | `size_factor² × momentum_12_1` | 极小盘动量 |
| `hml_rmw` | `value_factor × roe_ttm` | 质量价值（优质低估） |
| `smb_hml` | `size_factor × value_factor` | 规模×价值双重溢价 |
| `vol_mom` | `volatility_20d × momentum_12_1` | 动量崩溃风险信号 |

**`tools.py`**

对 `data/section/` 下每张截面 CSV 做三步处理：

1. **缩尾去极值**：1%～99% 百分位截断
2. **行业 + 市值中性化**：OLS 回归残差（仅适用于受行业/市值影响的因子）
3. **Z-score 标准化**：输出列名为 `<factor>_standard`

`size_factor` / `smb_squared` 本身是市值的函数，中性化后会归零，跳过中性化直接标准化。二值信号（`macd_divergence` 等）不做任何处理。

---

### 选股策略

**`strategy/selection/base_strategy.py` — `BaseStrategy`（抽象基类）**

所有截面选股策略的父类，定义必须实现的方法：

| 方法 | 说明 |
|---|---|
| `fit(date_str) -> bool` | 用 `date_str` 及之前的数据训练/更新模型，成功返回 `True` |
| `generate_signals(date_str) -> DataFrame \| None` | 返回当日所有股票的打分，至少含 `ts_code`、`score` 列，按 `score` 降序 |
| `reset() -> None` | 可选，重置内部状态（每次独立回测前由引擎调用） |
| `simple_backtest(start, end) -> dict` | 可选，纯信号质量评估（不涉及交易） |

**`strategy/selection/elastic_net_strategy.py` — `ElasticNetStrategy`**

基于 `ElasticNetCV` 的滚动窗口因子选股策略。每个交易日滚动训练，用当日截面因子值线性加权打分。

`ElasticNetConfig` 关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `factor_cols` | 31 个因子 | 参与建模的因子列名列表 |
| `window` | `40` | 滚动训练窗口（交易日数） |
| `l1_ratio_grid` | `[0.1, 0.5, 0.7, 0.9, 0.95, 1]` | L1 比例超参搜索范围 |
| `cv` | `5` | 交叉验证折数 |
| `smooth_weights` | `True` | 是否对因子权重做指数移动平滑（EMA） |
| `smooth_alpha` | `0.2` | EMA 平滑系数（新权重占比，越小越平滑） |

当前使用的 31 个因子：

```
估值：     pe_ttm_standard、pb_standard、dv_ttm_standard
技术：     macd_standard、rsi_standard、volatility_20d_standard、reversal_5d_standard
量价：     turnover_rate_x_standard、volume_ratio_standard、positive_flow_standard、total_mv_standard
二值信号： macd_divergence、macd_air_refuel
基本面：   gross_margin_standard、debt_ratio_standard、roe_ttm_standard
           revenue_growth_yoy_standard、profit_growth_yoy_standard、accruals_standard
FF 因子：  size_factor_standard、smb_squared_standard、value_factor_standard
           cma_factor_standard、asset_growth_yoy_standard、momentum_12_1_standard
交叉因子： smb_mom_standard、smb_squared_mom_standard、hml_rmw_standard
           smb_hml_standard、vol_mom_standard
```

`simple_backtest()` 输出 Rank IC 统计、多空分层收益、因子权重均值，返回 `{'ic_df', 'ls_df', 'weights_df'}`。

---

### 卖出策略

**`strategy/sell/base_sell_strategy.py` — `BaseSellStrategy`（抽象基类）**

```python
def evaluate(
    positions: list[dict],  # 当前持仓，每项含 ts_code/buy_date/buy_price/shares/current_price
    today: str,             # 当日日期 YYYYMMDD
    all_dates: list[str],   # 完整交易日历（用于计算持仓天数）
) -> dict[str, float]:      # {ts_code: keep_ratio}
```

`keep_ratio` 含义：`1.0` = 全部保留，`0.0` = 全部卖出，`0.5` = 减仓 50%，以此类推。

**`strategy/sell/hold_n_days.py` — `HoldNDaysSellStrategy`**

最简卖出策略：持有满 `n` 个交易日后全部清仓。

```python
sell = HoldNDaysSellStrategy(n=5)
```

---

### 择时策略

**`backtest/timing.py` — `BaseTimingStrategy`（抽象基类）**

| 方法 | 说明 |
|---|---|
| `prepare(start_date, end_date)` | 可选，回测前预计算（如读取指数数据） |
| `get_position_ratio(date_str) -> float` | 返回当日建仓比例，`0.0` = 空仓，`1.0` = 满仓 |

**`backtest/timing.py` — `MATiming`**

基于指数均线的牛熊判断：前一日指数收盘价高于 N 日均线则满仓，否则空仓。

```python
timing = MATiming(
    index_file='data/raw/index_daily/000905.SH.csv',
    ma_period=60,   # 60 日均线
)
```

---

### 回测引擎

**`backtest/config.py` — `BacktestConfig`**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `top_n` | `100` | 每次建仓选股数量上限 |
| `commission` | `0.0001` | 单边佣金（万分之一） |
| `initial_capital` | `1_000_000` | 初始资金（元） |
| `use_vol_control` | `False` | 是否启用波动率目标控制 |
| `target_vol` | `0.15` | 目标年化波动率 |
| `vol_window` | `20` | 波动率计算滚动窗口（日） |
| `stop_loss` | `None` | 止损阈值（预留接口，如 `-0.08`） |
| `take_profit` | `None` | 止盈阈值（预留接口，如 `0.15`） |

**`backtest/engine.py` — `BacktestEngine`**

```python
engine = BacktestEngine(
    strategy,           # BaseStrategy 实例
    loader,             # DataLoader 实例
    config,             # BacktestConfig 实例
    timing=timing,      # BaseTimingStrategy 实例，可选，默认不择时
    sell_strategy=sell, # BaseSellStrategy 实例，可选，默认 HoldNDaysSellStrategy(5)
)

nav_df, trades = engine.run('20230103', '20260331')
engine.report(nav_df)
```

`nav_df`：`DataFrame[date, nav]`，每日净值。  
`trades`：`list of (date, 'BUY'|'SELL', ts_code, shares, price, amount)`。

每日执行流程：
1. 按当日收盘价更新持仓市值
2. 记录当日净值（交易前）
3. **有持仓** → 调用卖出策略，执行减仓/清仓
4. **无持仓** → 调用选股策略 + 择时策略，按等权建仓
5. 更新当日净值（交易后）

---

## 快速开始

**前提：** 已完成数据拉取和因子计算流程（`data/section/` 目录下有含 `_standard` 列的截面文件）。

```bash
# ① 拉取原始数据
python data_local_dump.py

# ② 合并行情 + 财务数据
python merge_data.py

# ③ 计算因子（写回 data/series/）
python factor/new_factor_manager.py

# ④ 截面化 + 标准化（生成 data/section/）
python tools.py

# ⑤ 运行回测
python run_backtest.py
```

`run_backtest.py` 将依次输出：
1. Rank IC 统计报告（因子质量评估）
2. 模拟交易绩效报告（总收益、年化、夏普、最大回撤、胜率）

---

## 开发新内容

### 新增选股策略

1. 在 `strategy/selection/` 下新建文件，如 `lgb_strategy.py`
2. 继承 `BaseStrategy`，实现 `fit()` 和 `generate_signals()`

```python
from strategy.selection.base_strategy import BaseStrategy
import pandas as pd

class LGBStrategy(BaseStrategy):
    def __init__(self, config, data_loader):
        self.cfg = config
        self.loader = data_loader
        self.model = None

    def reset(self) -> None:
        self.model = None

    def fit(self, date_str: str) -> bool:
        # 构建训练数据，训练 LightGBM 模型
        # 成功返回 True，数据不足返回 False
        ...
        return True

    def generate_signals(self, date_str: str) -> pd.DataFrame | None:
        # 用训练好的模型对当日截面打分
        # 返回含 ts_code、score 列的 DataFrame，按 score 降序
        # 不依赖 label，可用于实盘
        ...
        return result
```

3. 在 `run_backtest.py` 中替换：

```python
from strategy.selection.lgb_strategy import LGBStrategy
strategy = LGBStrategy(s_cfg, loader)
```

---

### 新增卖出策略

1. 在 `strategy/sell/` 下新建文件，如 `stop_loss.py`
2. 继承 `BaseSellStrategy`，实现 `evaluate()`

```python
from strategy.sell.base_sell_strategy import BaseSellStrategy

class StopLossSellStrategy(BaseSellStrategy):
    def __init__(self, stop_ratio: float = -0.08):
        self.stop_ratio = stop_ratio

    def evaluate(self, positions, today, all_dates):
        result = {}
        for pos in positions:
            pnl = (pos['current_price'] - pos['buy_price']) / pos['buy_price']
            result[pos['ts_code']] = 0.0 if pnl <= self.stop_ratio else 1.0
        return result
```

3. 在 `run_backtest.py` 中替换：

```python
from strategy.sell.stop_loss import StopLossSellStrategy
sell = StopLossSellStrategy(stop_ratio=-0.08)
```

多条件组合示例（止损 + 持有到期）：

```python
class ComboSellStrategy(BaseSellStrategy):
    def __init__(self, stop_ratio=-0.08, hold_days=10):
        self.stop = StopLossSellStrategy(stop_ratio)
        self.hold = HoldNDaysSellStrategy(hold_days)

    def evaluate(self, positions, today, all_dates):
        r1 = self.stop.evaluate(positions, today, all_dates)
        r2 = self.hold.evaluate(positions, today, all_dates)
        # 任一条件触发则卖出（取最小保留比例）
        return {k: min(r1[k], r2[k]) for k in r1}
```

---

### 新增择时策略

1. 在 `backtest/timing.py` 中追加，或新建文件
2. 继承 `BaseTimingStrategy`，实现 `get_position_ratio()`

```python
from backtest.timing import BaseTimingStrategy

class VIXTiming(BaseTimingStrategy):
    """根据波动率指数调整仓位，波动率越高仓位越低"""
    def __init__(self, vix_file: str, threshold: float = 20.0):
        self.vix_file = vix_file
        self.threshold = threshold
        self._map = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        # 预计算每日仓位比例，存入 self._map
        ...

    def get_position_ratio(self, date_str: str) -> float:
        return self._map.get(date_str, 1.0)
```

3. 在 `run_backtest.py` 中替换：

```python
from backtest.timing import VIXTiming
timing = VIXTiming(vix_file='data/raw/vix.csv', threshold=20.0)
```

---

## 依赖关系

```
run_backtest.py
    ├── strategy/selection/elastic_net_strategy.py
    │       └── strategy/selection/base_strategy.py
    ├── strategy/sell/hold_n_days.py
    │       └── strategy/sell/base_sell_strategy.py
    ├── strategy/data_loader.py
    ├── backtest/config.py
    ├── backtest/timing.py
    └── backtest/engine.py
            ├── strategy/selection/base_strategy.py  （类型引用）
            ├── strategy/sell/base_sell_strategy.py  （类型引用）
            ├── strategy/sell/hold_n_days.py         （默认卖出策略）
            └── backtest/timing.py                   （类型引用）
```

各模块间无循环依赖，策略层不依赖回测层。
