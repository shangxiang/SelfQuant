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
  - [微信推送](#微信推送)
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
├── run_signal.py                          # 信号生成入口（含数据更新全流程）
│
├── data/                                  # 本地数据（不入 git）
│   ├── raw/                               # 原始 CSV
│   │   ├── trade_cal.csv                  # 交易日历
│   │   ├── stock_list/stock_list.csv      # 股票列表（含行业信息）
│   │   ├── stock_data/                    # 个股日线行情（不复权）
│   │   ├── adj_factor/                    # 复权因子
│   │   ├── daily_basic_data/              # 每日基本面快照（PE/PB/市值等）
│   │   ├── moneyflow/                     # 大中小单资金流
│   │   ├── margin_detail/                 # 融资融券明细
│   │   ├── top_list/                      # 龙虎榜
│   │   ├── income/                        # 利润表
│   │   ├── balancesheet/                  # 资产负债表
│   │   ├── cashflow/                      # 现金流量表
│   │   └── index_daily/                       # 精选指数日线（16 个，择时/基准用）
│   ├── financial.parquet                  # 合并后的财务宽表
│   ├── st_flag.parquet                    # ST 标识（ts_code, trade_date, is_st）
│   ├── final_result.parquet               # 日线宽表（按 ts_code, trade_date 排序）
│   ├── series/<ts_code>.parquet           # 时序层：逐股票，含因子列
│   └── market/trade_date=YYYYMMDD/        # 截面层：按交易日分区的 Parquet
│           └── part-0.parquet             #   含标准化后的 _standard 列
│
├── data_store.py                          # DuckDB + Parquet 存储层（分区读写）
├── data_local_dump.py                     # 从 Tushare 拉取原始数据到本地
├── merge_data.py                          # 合并行情/财务数据，生成 data/series/
├── standardize.py                         # 截面转换 + 去极值 + 中性化 + 标准化
├── notify.py                              # PushPlus 微信推送
│
├── factor/
│   └── new_factor_manager.py              # 因子计算：技术因子 + FF 风格因子 + 交叉因子
│
├── strategy/                              # 策略层
│   ├── data_loader.py                     # 截面数据加载器（带缓存）
│   ├── selection/                         # 截面选股策略
│   │   ├── base_strategy.py               # 抽象基类 BaseStrategy
│   │   ├── elastic_net_strategy.py        # 弹性网络策略（含因子评估）
│   │   ├── lgbm_strategy.py               # LightGBM 回归策略（固定超参数）
│   │   ├── lgbm_dynamic_strategy.py       # LightGBM 动态超参数策略（自动搜索）
│   │   ├── lgbm_ranker_strategy.py        # LambdaRank 排序优化策略
│   │   └── ic_weighted_strategy.py        # IC 加权策略（无参数）
│   └── sell/                              # 卖出策略
│       ├── base_sell_strategy.py          # 抽象基类 BaseSellStrategy
│       ├── hold_n_days.py                 # 持有 N 日清仓策略
│       └── virtual_portfolio_stop.py      # 虚拟组合止损策略
│
└── backtest/                              # 回测层
    ├── config.py                          # BacktestConfig 参数配置
    ├── timing.py                          # 择时策略（5 种实现）
    └── engine.py                          # BacktestEngine 回测引擎
```

---

## 整体流程

```
① 数据拉取
data_local_dump.py  →  data/raw/

② 数据合并（DuckDB 执行宽表连接，避免 pandas 内存爆炸）
merge_data.py  →  data/final_result.parquet  →  data/series/<ts_code>.parquet

③ 因子计算（逐股票时序）
factor/new_factor_manager.py  →  写回 data/series/<ts_code>.parquet

④ 截面化 + 标准化（流式转置 + 逐日处理）
standardize.py: series_to_section()  →  data/market/trade_date=YYYYMMDD/part-*.parquet
standardize.py: standardize()        →  同一分区写回 _neutral / _standard 列

⑤ 策略 + 回测
DataLoader（读 data/market/ 分区）→ BaseStrategy.fit() + generate_signals() → BacktestEngine.run()
```

**关键设计原则：**
- 策略只负责"打分"，不感知资金和仓位
- 回测引擎只负责"交易模拟"，不内置任何策略逻辑
- 三类策略（选股 / 择时 / 卖出）均通过依赖注入，可自由组合替换
- 信号与成交错开一日（T 日收盘生成信号，T+1 日收盘成交），避免未来函数
- 财务数据用披露日（f_ann_date）对齐，而非报告期（end_date），保证 point-in-time 正确性

---

## 模块说明

### 数据层

数据分两层存储，互为转置，各自服务一种访问模式：

| 层 | 布局 | 服务谁 | 访问模式 |
|---|---|---|---|
| `series/` | 每只股票一个 Parquet | 因子计算 | 逐股票时序（rolling / ewm 沿时间轴） |
| `market/` | `trade_date=YYYYMMDD/` 分区 | 标准化、策略、回测 | 逐交易日截面 |

**`data_store.py` — `DataStore`（DuckDB + Parquet）**

统一负责 `market/` 层的读写。两个关键设计：

1. **分区键就是 `trade_date`**。DuckDB 只对 hive 分区列做文件裁剪，因此所有
   查询的过滤条件都写在分区列上。实测 300 天样本：命中分区列 0.23s，
   未命中 1.93s（扫描全部 300 个文件），差距随数据量线性放大。
2. **流式构建**。`build_sections_from_series()` 按股票分块读取 → 按交易日切分 →
   以 `part-<块号>.parquet` 追加写入对应分区，内存只保留「一块股票」的数据。
   一次性 concat 全量时序在 677 万行 × 数百列的规模下必然 OOM。

```python
from data_store import DataStore

store = DataStore()
df    = store.read_section('20240102')                  # 单个截面
win   = store.read_window('20240101', '20240301',        # 区间（命中分区裁剪）
                          columns=['ts_code', 'close_x', 'label'])
dates = store.get_all_dates()
```

**`data_local_dump.py` — `DownloadData`**

从 Tushare 拉取所有原始数据到 `data/raw/`，支持断点续传（已存在的文件跳过）。

```bash
python data_local_dump.py
```

执行顺序：交易日历 → 股票列表 → 指数基础信息 → 按日期数据（行情快照/资金流/融资融券/龙虎榜）→ 按股票数据（行情/财务三表）→ 指数日线。

#### 指数日线：精选 vs 全量

`index_daily()` 会遍历 `index_basic/SSE.csv` 的**全部指数**（数千个）逐个下载，绝大多数用不上。
日常使用请改用 `index_daily_selected()`，只下 `DownloadData.INDEX_SELECTED` 里的 16 个：

```python
d = DownloadData()
d.index_daily_selected()                       # 全部 16 个
d.index_daily_selected(categories=['scale'])   # 只要规模宽基
d.index_daily_selected(codes=['000852.SH'])    # 只要指定代码
```

| 分类 | 指数 |
|---|---|
| `scale` 规模宽基（10） | 上证指数、沪深300、中证500、中证1000、中证2000、中证全指、中证A500、上证50、创业板指、科创50 |
| `style` 风格（4） | 国证成长、国证价值、小盘成长、小盘价值 |
| `div` 红利（2） | 中证红利、红利指数 |

覆盖 2018-01-02 起共 2120 个交易日（科创50 自 2019-12-31 起）。增量逻辑复用
`_incremental_start()`（取已有最新日期的次日为起点）+ `_append_and_save()` 去重写回，
已下载过的重复调用不会产生新行也不会产生重复。

> ⚠️ 中证A500 的正确代码是 **`000510.SH`**。项目早期在 `backtest/timing.py`、
> `run_backtest.py` 等处误写成 `000510.CSI`（该代码不存在，择时会因文件缺失而静默失效），已统一修正。

**`merge_data.py`**

- `financial_data_preprocess()`：合并财务三表，利润表/现金流量表做累计→单季度转换，输出 `data/financial.parquet`
- `build_st_flag()`：由 namechange 构建 ST 标识，输出 `data/st_flag.parquet`（向量化，跳过 Tushare 写出的空文件）
- `merge_basic_daily_data()`：用 DuckDB 把行情、复权因子、基本面快照、融资融券、资金流、龙虎榜按 (ts_code, trade_date) 全外连接成宽表；重名列沿用 pandas 的 `_x` / `_y` 规则
- `split_to_series_section()`：用 pyarrow 按批流式拆分到 `data/series/<ts_code>.parquet`

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

**复权机制：** 所有技术因子基于**后复权价格**（`close_hfq` = 不复权价格 × 复权因子）计算，消除除权缺口对技术指标的干扰；回测引擎使用**不复权真实价格**进行交易模拟。

因子分九大类（共 150+ 列，下游使用 `_standard` 后缀版本）：

**Alpha101 因子（31 个）**

世坤经典因子，基于价格和成交量的复杂组合。

| 因子 | 说明 |
|---|---|
| `alpha101_1` ~ `alpha101_101` | 包含时序排名、相关性、条件判断等复杂构造 |

**技术/量价因子（基于日线行情）**

| 因子 | 说明 |
|---|---|
| `macd` / `rsi` / `kdj` / `cci` / `adx` | 经典技术指标 |
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

**Size 规模因子（2 个）**

| 因子 | 说明 |
|---|---|
| `size` | 总市值因子：`-log(TotalShares × ClosePrice / 1e6)` |
| `float_size` | 流通市值因子：`-log(FloatShares × ClosePrice / 1e6)` |

**Value 价值因子（5 个）**

| 因子 | 说明 |
|---|---|
| `earnings_to_price` | 市盈率倒数（E/P） |
| `book_to_market` | 账面市值比（B/M） |
| `ocf_to_market` | 经营现金流市值比 |
| `fcf_to_market` | 自由现金流市值比 |
| `sales_to_market` | 营业收入市值比 |

**Reversal 反转因子（2 个）**

| 因子 | 说明 |
|---|---|
| `small_cap_reversal_21d` | 小盘反转因子：市值最小的股票取过去 21 日累计收益的反转信号 |
| `price_dist` | 价格距离因子：股价与下一个整数关口的距离 |

**Momentum 动量因子（10 个）**

| 因子 | 说明 |
|---|---|
| `return_5d` / `return_21d` / `return_42d` / `return_63d` / `return_126d` / `return_252d` | 多周期累计收益率 |
| `ma_20d` | 20 日移动平均线 |
| `price_position_ir_60d` | 价格位置动量因子：过去 60 日（收盘-开盘）/（最高-最低）比率的信息比率 |
| `rsrs` | RSRS 指标：通过回归最高价和最低价得到斜率，再对斜率进行标准化 |
| `days_down_up` | 连续涨跌天数因子：连续上涨天数与连续下跌天数之差的绝对值减 1 |

**Risk 风险因子（15 个）**

| 因子 | 说明 |
|---|---|
| `return_std_21d` / `return_std_42d` / `return_std_63d` / `return_std_126d` / `return_std_252d` | 多周期收益率标准差 |
| `sharpe_60d` / `sharpe_750d` | 夏普比率 |
| `adjusted_sharpe_750d` | 调整夏普率：`Mean / Std^4`，对高波动性惩罚更重 |
| `high_low_21d` / `high_low_42d` / `high_low_63d` / `high_low_126d` / `high_low_252d` | 净值曲线最高点与最低点的比值 |
| `days_beyond_upper_lower_21d` | 21 日内价格超越均值±标准差的天数之差 |
| `log_price` | 收盘价的自然对数 |

**Liquidity 流动性因子（30 个）**

| 因子 | 说明 |
|---|---|
| `avg_turnover_5d` / `avg_turnover_10d` / `avg_turnover_20d` | 短期平均换手率 |
| `std_turnover_21d` / `std_turnover_42d` / `std_turnover_63d` / `std_turnover_126d` / `std_turnover_252d` | 多周期换手率标准差 |
| `avg_turnover_21d` / `avg_turnover_42d` / `avg_turnover_63d` / `avg_turnover_126d` / `avg_turnover_252d` | 多周期平均换手率 |
| `bias_turn_21d_252d` / `bias_turn_42d_252d` / `bias_turn_63d_252d` / `bias_turn_126d_252d` | 短期与长期换手率的乖离率 |
| `bias_std_turn_21d_252d` | 短期与长期换手率标准差的乖离率 |
| `amount_ma_20d` | 20 日成交额移动平均 |
| `turnover_ma_20d` | 20 日成交量与流通市值比率的移动平均 |
| `sum_abs_rtn_amount_20d` | 20 日累计绝对收益率与累计成交额的比值 |
| `turnover_ma_20d_120d` | 20 日成交量与流通市值比率与 120 日的比值 |

**Quality 质量因子（10 个）**

| 因子 | 说明 |
|---|---|
| `roe_ttm` | TTM 净资产收益率 |
| `roa_ttm` | TTM 总资产收益率 |
| `gross_margin` | 毛利率 |
| `net_margin` | 净利率 |
| `debt_to_assets` | 资产负债率 |
| `current_ratio` | 流动比率 |
| `quick_ratio` | 速动比率 |
| `cash_flow_to_debt` | 现金流负债比 |
| `accruals` | 应计项目 |
| `earnings_quality` | 盈余质量：经营现金流/净利润 |

**Growth 成长因子（10 个）**

| 因子 | 说明 |
|---|---|
| `revenue_growth_yoy` | 营收同比增长率 |
| `profit_growth_yoy` | 净利润同比增长率 |
| `asset_growth_yoy` | 资产同比增长率 |
| `roe_growth_yoy` | ROE 同比增长率 |
| `eps_growth_yoy` | EPS 同比增长率 |
| `revenue_growth_qoq` | 营收环比增长率 |
| `profit_growth_qoq` | 净利润环比增长率 |
| `gross_margin_growth` | 毛利率增长率 |
| `net_margin_growth` | 净利率增长率 |
| `ocf_growth_yoy` | 经营现金流同比增长率 |

**Fama-French 风格因子（股票截面）**

| 因子 | 对应 FF 因子 | 说明 |
|---|---|---|
| `size_factor` | SMB | `-ln(流通市值)`，规模因子 |
| `value_factor` | HML | `1/PB`，价值因子 |
| `cma_factor` | CMA | 资产增速取反，保守投资因子 |
| `momentum_12_1` | Mom | 12-1 月动量 |

**高阶交叉因子**

| 因子 | 构造方式 | 经济含义 |
|---|---|---|
| `smb_squared` | `size_factor²` | 非线性规模效应 |
| `smb_mom` | `size_factor × momentum_12_1` | 小盘股动量溢价 |
| `smb_squared_mom` | `size_factor² × momentum_12_1` | 极小盘动量 |
| `hml_rmw` | `value_factor × roe_ttm` | 质量价值（优质低估） |
| `smb_hml` | `size_factor × value_factor` | 规模×价值双重溢价 |
| `vol_mom` | `volatility_20d × momentum_12_1` | 动量崩溃风险信号 |

**`standardize.py`**

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

**策略矩阵**

| 策略 | 模型 | 特点 | 适用场景 |
|---|---|---|---|
| `ElasticNetStrategy` | 线性回归 | 滚动窗口 + EMA 权重平滑 | 因子线性关系强时 |
| `LGBMStrategy` | LightGBM 回归 | rank 归一化标签 + 时间衰减权重 | 非线性关系，默认选择 |
| `LGBMDynamicStrategy` | LightGBM 回归 | **自动搜索超参数**（window/halflife/模型参数） | 市场风格变化快时 |
| `LGBMRankerStrategy` | LambdaRank | 直接优化 NDCG@20 + 五档相关度 | 排序质量要求高时 |
| `ICWeightedStrategy` | IC 加权 | 无参数 + 滚动 RankIC 作为权重 | 快速 baseline |

**`ElasticNetStrategy`**

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

**`LGBMStrategy`**

基于 LightGBM 回归的选股策略。将 label 做 rank 百分位归一化后训练，模型学习"排序"而非"绝对收益"。

`LGBMConfig` 关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `factor_cols` | 57 个因子 | 参与建模的因子列名列表 |
| `window` | `40` | 滚动训练窗口（交易日数） |
| `weight_halflife` | `20` | 样本权重半衰期（越近样本权重越大） |
| `top_weight_factor` | `1.5` | 头部样本（label > 80%）额外加权倍数 |
| `label_period` | `5` | label 持有期（交易日） |
| `label_lookahead` | `6` | label 起始偏移（避免未来函数） |

**`LGBMDynamicStrategy`（动态超参数版）**

在 `LGBMStrategy` 基础上增加**自动超参数搜索**，定期用历史数据找最优参数。

两级搜索机制：

| 级别 | 搜索内容 | 频率 | 组合数 |
|---|---|---|---|
| 一级 | 模型超参数（max_depth、num_leaves 等） | 每月 | 162 |
| 二级 | window + weight_halflife | 每 3 个月 | 12 |

搜索目标：验证集 Rank IC 均值最高。

`LGBMDynamicConfig` 额外参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `tune_interval` | `20` | 一级搜索频率（交易日） |
| `tune_interval_v2` | `3` | 每 N 次一级搜索后触发二级搜索 |
| `tune_lookback` | `120` | 搜索用的历史窗口（交易日） |
| `tune_val_ratio` | `0.2` | 验证集比例（时间序列切分） |

使用方式：

```python
from strategy.selection.lgbm_dynamic_strategy import LGBMDynamicConfig, LGBMDynamicStrategy

s_cfg = LGBMDynamicConfig()
strategy = LGBMDynamicStrategy(s_cfg, loader)
```

**`LGBMRankerStrategy`**

使用 LightGBM 的 LambdaRank 模式，直接优化排序质量（NDCG）。

| 特点 | 说明 |
|---|---|
| 损失函数 | `lambdarank`，直接优化 NDCG |
| 标签 | 五档离散相关度（前 5%/20%/50%/80%/其余 → 4/3/2/1/0） |
| 评估指标 | NDCG@20 |

**`ICWeightedStrategy`**

无参数的因子加权策略。用过去 N 天各因子与 label 的 RankIC 作为权重，对当日因子标准化值加权求和。

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

| 策略 | 说明 |
|---|---|
| `HoldNDaysSellStrategy` | 持有满 N 个交易日后全部清仓 |
| `VirtualPortfolioStopStrategy` | 虚拟组合止损：用盲窗口内的"模拟考试"判断模型是否失效，触发降仓 |

---

### 择时策略

**`backtest/timing.py` — `BaseTimingStrategy`（抽象基类）**

| 方法 | 说明 |
|---|---|
| `prepare(start_date, end_date)` | 可选，回测前预计算（如读取指数数据） |
| `get_position_ratio(date_str) -> float` | 返回当日建仓比例，`0.0` = 空仓，`1.0` = 满仓 |

| 策略 | 逻辑 | 适用场景 |
|---|---|---|
| `MATiming` | 指数收盘价 vs N 日均线 | 趋势行情 |
| `StyleConvergenceTiming` | 大小盘高相关 + 双双下行 → 空仓 | 风格切换期 |
| `BlindWindowTiming` | 盲窗口相关性/波动率分位数 + 趋势滤波 | 震荡市 |
| `LastBatchTiming` | 上期亏损递减仓位，盈利恢复满仓 | 模型不稳定期 |
| `LGBMDriftTiming` | 因子漂移检测 → 降仓 | 因子失效期 |

---

### 回测引擎

**`backtest/config.py` — `BacktestConfig`**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `top_n` | `100` | 每次建仓选股数量上限 |
| `holding_period` | `5` | 固定持有期（交易日），同时作为默认卖出策略参数 |
| `commission` | `0.0001` | 单边佣金（万分之一），买卖各收一次 |
| `stamp_duty` | `0.0005` | 印花税，仅卖出收取（A 股 2023-08-28 起为 0.05%） |
| `slippage` | `0.001` | 双边滑点（买入价上浮、卖出价下浮，按成交价比例） |
| `enable_limit_check` | `True` | 是否启用涨跌停约束 |
| `limit_up_pct` | `9.8` | 涨幅 ≥ 此值视为涨停，不可买入 |
| `limit_down_pct` | `-9.8` | 跌幅 ≤ 此值视为跌停，不可卖出 |
| `initial_capital` | `1_000_000` | 初始资金（元） |
| `use_vol_control` | `False` | 是否启用波动率目标控制 |
| `target_vol` | `0.15` | 目标年化波动率 |
| `vol_window` | `20` | 波动率计算滚动窗口（日） |
| `stop_loss` | `None` | 止损阈值（预留接口，如 `-0.08`） |
| `take_profit` | `None` | 止盈阈值（预留接口，如 `0.15`） |

### 交易成本（务必先看）

5 日换手的单次往返成本 ≈ 滑点 0.1%×2 + 佣金 0.01%×2 + 印花税 0.05% = **0.27% 名义**，
一年约 50 个来回 → **年化约 12%**。策略毛 alpha 必须超过这个数才可能盈利。

实测（20260730 ~ 20260924，LGBMRanker 策略，top_n=10）：

| 配置 | 总收益率 | 年化 | 夏普 |
|---|---|---|---|
| 成本与涨跌停全关 | +1.77% | +12.03% | 0.2453 |
| 默认（印花税 + 滑点 + 涨跌停） | -0.43% | -2.68% | -0.0704 |

成本明细：佣金 1,447 + 印花税 3,375 + 滑点 14,470 = 19,292
（占初始资金 1.93%，约占交易额 0.133%）。

**滑点是大头**（占成本 75%）。想降成本，延长持有期比调参数更有效。

### 因子集（factor_set）

数据里实际有 **190 个原始因子**（对应 226 个 `_standard` 列），但各策略 Config 里的
`factor_cols` 是手工挑选的，长期落后于因子库扩充——Alpha101 31 个、多周期动量族、
风险族、流动性族、质量成长族、价值族一度完全没被启用。

各策略 Config 里的 `factor_cols` 已**按因子族分组写全**（LGBM / Ranker 211 个，ElasticNet 237 个），
想增减直接注释掉对应行即可。分组示例：

```python
self.factor_cols = [
    # ---- 技术 / 量价 / 资金（32）----
    'pe_ttm', 'pb', 'macd', 'K', 'rsi', ...
    # ---- Alpha101（31）----
    'alpha101_1', ...
    # ---- 动量（17）----
    'return_5d', ... 'momentum_12_1', ...
    # ---- 风险/波动 ----   ---- 流动性 ----   ---- 质量/成长 ----
    # ---- 价值/风格 ----   ---- 反转 ----     ---- 微观结构/形态 ----
    # ---- 时序差分 ----
]
# 末尾再追加 13 个行业/市场 beta 特征
```

`factor_set` 是兜底开关（默认 `curated` = 就用手写的这份）：

| `factor_set` | 含义 |
|---|---|
| `curated` | 只用 `factor_cols` 里手写的列表（默认） |
| `all` | 手写列表 ∪ `strategy/factor_columns.py` 从数据中自动发现的新因子 |

`all` 只在"以后新增了因子、还没来得及写进列表"时有用——`discover_factor_cols()` 会读截面
schema 自动识别因子（排除原始行情列 / `label*` / `*_hfq` / `*_neutral` / `*_standard` / 价格成交标识列）。

> 注意：ElasticNet 用 237 个特征时单次 fit 会超过 5 分钟（ElasticNetCV 的 6×100×5 折坐标下降）。
> 手写裁剪到 80~120 个，或调小 `l1_ratio_grid` / `cv` 可显著提速。树模型无此问题。

命令行覆盖：

```bash
python run_backtest.py --factor-set all        # 手写列表 + 自动发现
python run_backtest.py --factor-set curated    # 只用列表里的（默认）
```

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

### 微信推送

**`notify.py` — PushPlus 推送**

回测/信号生成完成后，自动推送结果到微信。

```python
from notify import send_wechat

send_wechat(
    title="今日信号",
    content="## 买入\n- 贵州茅台\n- 宁德时代\n\n## 卖出\n- 比亚迪"
)
```

**配置方式：**

1. 注册 [PushPlus](https://www.pushplus.plus)，获取 token
2. 设置环境变量：
   ```powershell
   $env:PUSHPLUS_TOKEN = "你的token"
   ```
3. 或在代码中直接传入：
   ```python
   send_wechat("标题", "内容", token="你的token")
   ```

支持 Markdown 格式，适合展示交易信号、回测结果等。

---

## 快速开始

**前提：** 已完成数据拉取和因子计算流程（`data/market/` 下有含 `_standard` 列的截面分区）。

```bash
# ① 拉取原始数据
python data_local_dump.py

# ② 合并行情 + 财务数据（宽表 → 时序层）
python merge_data.py

# ③ 计算因子（写回 data/series/）
python factor/new_factor_manager.py

# ④ 截面化 + 标准化（生成 data/market/trade_date=YYYYMMDD/）
python standardize.py

# ⑤ 运行回测
python run_backtest.py
```

②③④⑤ 也可以一步跑完（含下载，日常更新用这个）：

```bash
python run_signal.py
```

回测区间可用参数覆盖，默认取截面数据中最近约 250 个交易日：

```bash
python run_backtest.py --start 20240102 --end 20241231
python run_backtest.py 20260522        # 只查询该日的选股与择时信号
python run_backtest.py --ic            # 回测前额外跑一次 RankIC / 分层收益评估
```

`run_backtest.py` 将依次输出：
1. Rank IC 统计报告（因子质量评估）
2. 模拟交易绩效报告（总收益、年化、夏普、最大回撤、胜率）

---

## 每日增量更新

收盘后跑一次即可把数据推到最新交易日，**已下载 / 已计算的数据只补充、不清理**：

```bash
python run_signal.py                  # 增量：下载 → 合并 → 因子 → 标准化
python run_signal.py --end 20260925   # 指定更新到哪一天
python run_signal.py --signal         # 更新完顺带生成 T+1 的选股信号
python run_signal.py --since 20260901 # 重做 20260901 之后的历史（该日之后的行用新值替换）
python run_signal.py --full           # 全量重建（会覆盖 final_result.parquet，慎用）
```

各环节怎么做到"只追加"：

| 环节 | 增量行为 | 关键文件 |
| --- | --- | --- |
| 下载 | 各数据源自己算缺口，已存在的文件跳过 | `data_local_dump.py::_incremental_start` |
| ST 标识 | 用 `stock_list` 的当前名称与本地记录的上一次名称比对，**只追加新交易日**，历史行不动 | `merge_data.build_st_flag` + `data/name_state.csv` |
| 日线宽表 | 只合并 `trade_date > 已有最新日` 的新日期，追加进 `final_result.parquet` | `merge_data.merge_basic_daily_data(incremental=True)` |
| 时序层 | 只拿新增段去 upsert 各股票的 `series` 文件，历史行连同因子列一起保留 | `merge_data.split_to_series_section` |
| 因子 | 只算 `_date_range.csv` 记录之后的新行（携带 260 日历史上下文） | `factor/new_factor_manager.py` |
| 截面层 | 只构建缺失的 / 未标准化的交易日分区 | `standardize.series_to_section` / `standardize` |

**ST 判定的改动**：不再调用 `get_namechange` 接口。每次更新时用
`data/raw/stock_list/stock_list.csv` 里的当前名称，与 `data/name_state.csv`
中记录的上一次名称比对 —— 名称不一致即视为发生了变更（戴帽 / 摘帽），
并把新的 ST 状态写入 `data/st_flag.parquet` 的新交易日行。
因此**必须每日收盘后连续更新**：漏几天就会漏掉这几天的改名事件。
首次运行时会用本地已下载的 `data/raw/namechange/` 给状态打底（只读本地，不调接口）。

维护用的三个产物：
- `data/name_state.csv` —— 每只股票上一次记录的名称 / ST 状态 / 最后更新日
- `data/st_flag.parquet` —— `(ts_code, trade_date, is_st)`，历史区间的 ST 标识
- `data/_final_result_new.parquet` —— 本次新增段的宽表，供拆分步骤使用

重做历史（`--since`）时会自动把 `data/series/_date_range.csv` 里的因子进度回退到该日，
被覆盖的行才会重新计算因子，其余行不受影响。

---

## 开发新内容

### 新增选股策略

1. 在 `strategy/selection/` 下新建文件，如 `xgb_strategy.py`
2. 继承 `BaseStrategy`，实现 `fit()` 和 `generate_signals()`

```python
from strategy.selection.base_strategy import BaseStrategy
import pandas as pd

class XGBStrategy(BaseStrategy):
    def __init__(self, config, data_loader):
        self.cfg = config
        self.loader = data_loader
        self.model = None

    def reset(self) -> None:
        self.model = None

    def fit(self, date_str: str) -> bool:
        # 构建训练数据，训练 XGBoost 模型
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
from strategy.selection.xgb_strategy import XGBStrategy
strategy = XGBStrategy(s_cfg, loader)
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
