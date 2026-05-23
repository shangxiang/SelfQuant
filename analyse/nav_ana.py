import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from typing import Optional

# ---- 中文字体 ----
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# 配置区：修改这里
# ================================================================
path  = "../backtest/results/20230104_20260331_20260522_134709/"
path2 = "../backtest/results/20230104_20260331_20260522_142217/"

# 归一化起点日期（YYYYMMDD 字符串），None = 使用数据第一行
# 例：'20240101' 表示从 2024 年起点对齐，方便对比某段时间内的相对表现
NORM_START = None
NORM_END = None
# NORM_START = "20220104"
# NORM_END = "20221231"

# 热力图采样粒度：每隔多少个交易日取一个节点（越小越精细，越慢）
HEATMAP_STEP = 5
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

# ================================================================
# 图1：净值归一化折线图
# ================================================================
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

# ================================================================
# 图2-4：超额收益热力图
# 颜色含义：绿色 = 策略跑赢基准，红色 = 策略跑输基准
# 数值 = (策略区间收益) - (基准区间收益)，单位：百分点
# ================================================================

def build_excess_matrix(s1: pd.Series, s2: pd.Series, step: int) -> tuple:
    """
    计算 s1 相对 s2 的超额收益矩阵。
    matrix[i, j] = (s1[j]/s1[i] - 1) - (s2[j]/s2[i] - 1)，j > i
    行=起始日，列=结束日；配合 origin='lower' 使两轴均从左下角小日期开始。
    """
    combined = pd.DataFrame({'a': s1, 'b': s2}).dropna()
    sampled = combined.iloc[::step]
    a = sampled['a'].values
    b = sampled['b'].values
    dates = sampled.index

    n = len(a)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            ret_a = a[j] / a[i] - 1
            ret_b = b[j] / b[i] - 1
            matrix[i, j] = (ret_a - ret_b) * 100  # 行=起始日，列=结束日

    labels = [d.strftime('%y-%m') for d in dates]
    return matrix, labels


def plot_heatmap(ax, matrix: np.ndarray, labels: list, title: str, vmax: float) -> None:
    """在给定 ax 上绘制超额收益热力图。"""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['#d73027', '#f7f7f7', '#1a9850']
    )
    cmap.set_bad(color='#e0e0e0')  # NaN（下三角无意义区域）显示为灰色
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # origin='lower'：y 轴从下到上递增，与 x 轴方向一致，原点在左下角
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='超额收益（百分点）', fraction=0.046, pad=0.04)

    n = len(labels)
    tick_gap = max(1, n // 12)
    ticks = list(range(0, n, tick_gap))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=8)

    ax.set_xlabel('结束日期')
    ax.set_ylabel('起始日期')
    ax.set_title(title, fontsize=12)


# 使用全量数据（不受 NORM_START/END 限制）构建热力图，覆盖完整历史
hm_nav  = nav.set_index('trade_date')['nav'].dropna()
hm_nav2 = nav.set_index('trade_date')['nav_2'].dropna()

fig, ax = plt.subplots(figsize=(9, 8))
fig.suptitle('区间超额收益热力图（绿=跑赢，红=跑输）', fontsize=14)

matrix, labels = build_excess_matrix(hm_nav, hm_nav2, HEATMAP_STEP)
vmax = np.nanpercentile(np.abs(matrix), 95)
plot_heatmap(ax, matrix, labels, 'nav vs nav_2', vmax)

plt.tight_layout()
plt.show()
