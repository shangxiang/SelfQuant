import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

# ---- 中文字体 ----
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 配置区：修改这里
# ================================================================
path  = "../backtest/results/20180103_20230103_20260519_203554/"
path2 = "../backtest/results/20180104_20260515_20260522_233941/"

# 归一化起点日期（YYYYMMDD 字符串），None = 使用数据第一行
# 例：'20240101' 表示从 2024 年起点对齐，方便对比某段时间内的相对表现
# NORM_START = None
# NORM_END = None
NORM_START = "20220104"
NORM_END = "20221231"
# ================================================================

nav = pd.read_csv(path + 'nav.csv')
nav.rename(columns={'date': 'trade_date'}, inplace=True)
nav_2 = pd.read_csv(path2 + 'nav.csv')
nav_2.rename(columns={'date': 'trade_date', 'nav': 'nav_2'}, inplace=True)

zza500 = pd.read_csv('../data/raw/index_daily/000510.CSI.csv')
zz2000 = pd.read_csv('../data/raw/index_daily/932000.CSI.csv')

nav = nav.merge(zza500[['trade_date', 'close']], on='trade_date', how='left')
nav.rename(columns={'close': 'zza500'}, inplace=True)
nav = nav.merge(zz2000[['trade_date', 'close']], on='trade_date', how='left')
nav.rename(columns={'close': 'zza2000'}, inplace=True)
nav = nav.merge(nav_2, on='trade_date', how='left')

nav['trade_date'] = pd.to_datetime(nav['trade_date'], format='%Y%m%d')
nav.sort_values('trade_date', inplace=True)
nav.reset_index(drop=True, inplace=True)

# 确定归一化起点行
if NORM_START is not None:
    norm_date = pd.to_datetime(NORM_START, format='%Y%m%d')
    # 找到 >= norm_date 的第一行
    candidates = nav[nav['trade_date'] >= norm_date]
    if candidates.empty:
        raise ValueError(f'NORM_START={NORM_START} 超出数据范围')
    norm_idx = candidates.index[0]
else:
    norm_idx = nav.index[0]

# 确定图标展示终点
if NORM_END is not None:
    norm_date = pd.to_datetime(NORM_END, format='%Y%m%d')
    # 找到 >= norm_date 的第一行
    candidates = nav[nav['trade_date'] <= norm_date]
    if candidates.empty:
        raise ValueError(f'NORM_END={NORM_END} 超出数据范围')
    end_idx = candidates.index[-1]
else:
    end_idx = nav.index[-1]

norm_label = nav.loc[norm_idx, 'trade_date'].strftime('%Y-%m-%d')

# 归一化：以 norm_idx 行的值为基准，令该行 = 100
cols = ['nav', 'nav_2', 'zza500', 'zza2000']
for col in cols:
    base = nav.loc[norm_idx, col]
    nav[col + '_norm'] = nav[col] / base * 100

# 只展示 norm_idx 之后的数据（之前的数据归一化后无意义）
nav_plot = nav.loc[norm_idx:end_idx].copy()

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(nav_plot['trade_date'], nav_plot['nav_norm'],    label='nav',    linewidth=2)
plt.plot(nav_plot['trade_date'], nav_plot['nav_2_norm'],  label='nav_2',  linewidth=2)
plt.plot(nav_plot['trade_date'], nav_plot['zza500_norm'], label='中证A500', linewidth=2)
plt.plot(nav_plot['trade_date'], nav_plot['zza2000_norm'],label='中证2000', linewidth=2)

plt.axhline(100, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
plt.title(f'净值归一化对比（起点={norm_label}，基准=100）', fontsize=14)
plt.xlabel('日期')
plt.ylabel('相对值（起始点=100）')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()