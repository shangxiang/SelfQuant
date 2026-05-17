import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- 中文字体 ----
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

path = "../backtest/results/20230103_20260331_20260517_150854/"

nav = pd.read_csv(path + 'nav.csv')
nav.rename(columns={'date':'trade_date'}, inplace=True)
zza500 = pd.read_csv('../data/raw/index_daily/000510.CSI.csv')
zz2000 = pd.read_csv('../data/raw/index_daily/932000.CSI.csv')
nav = nav.merge(zza500[['trade_date', 'close']], on='trade_date', how='left')
nav.rename(columns={'close':'zza500'}, inplace=True)
nav = nav.merge(zz2000[['trade_date', 'close']], on='trade_date', how='left')
nav.rename(columns={'close':'zza2000'}, inplace=True)

nav['trade_date'] = pd.to_datetime(nav['trade_date'], format='%Y%m%d')
# 按日期排序，防止顺序错乱
nav.sort_values('trade_date', inplace=True)

# 归一化：让每个序列的第一行数据变为 100
for col in ['nav', 'zza500', 'zza2000']:
    first_val = nav[col].iloc[0]          # 取第一个有效值
    nav[col + '_norm'] = nav[col] / first_val * 100

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(nav['trade_date'], nav['nav_norm'], label='nav', linewidth=2)
plt.plot(nav['trade_date'], nav['zza500_norm'], label='中证A500', linewidth=2)
plt.plot(nav['trade_date'], nav['zza2000_norm'], label='中证2000', linewidth=2)

# 装饰
plt.title('净值归一化对比 (起点=100)', fontsize=14)
plt.xlabel('日期')
plt.ylabel('相对值 (起始点=100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()