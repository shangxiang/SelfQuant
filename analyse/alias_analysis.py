"""
analyse/alias_analysis.py

分析 batch_alias_profit.csv 中 alias（打分偏离度）与每期收益的关系。

alias 定义：alias = (第 top_n 名打分 - 全市场均分) / 全市场打分标准差
  alias 越高 → 入选股票的打分越极端，模型对本次选股越"自信"
  alias 越低 → 打分分布均匀，入选边际不清晰

用法：
    python analyse/alias_analysis.py <task目录路径>

示例：
    python analyse/alias_analysis.py backtest/results/20230103_20260331_20260517_164109
"""

import sys
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, linregress

plt.rcParams['font.family'] = ['STHeiti', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------ #
#  常量                                                                #
# ------------------------------------------------------------------ #

ALIAS_WARN  = 2.5   # 超过此值视为高风险区
ALIAS_CRIT  = 3.0   # 超过此值视为极端区
ROLL_WINDOW = 10    # 滚动相关窗口（期数）

# score 信号质量阈值：top-N 持仓的 score_max 低于此值视为"无信号"
SCORE_LOW_THRESH  = 0.09   # 极低：模型几乎无区分力
SCORE_HIGH_THRESH = 0.15   # 较高：模型有一定置信度

PERIOD_BINS   = ['2025-01-01', '2025-06-30', '2025-12-31', '2026-12-31']
PERIOD_LABELS = ['2025H1', '2025H2', '2026']
PERIOD_COLORS = {'2025H1': '#4575b4', '2025H2': '#91bfdb', '2026': '#d73027'}

# ------------------------------------------------------------------ #
#  数据加载                                                             #
# ------------------------------------------------------------------ #

def load_data(task_dir: str) -> pd.DataFrame:
    path = os.path.join(task_dir, 'batch_alias_profit.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到 {path}，请确认目录路径正确。')
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df.dropna(subset=['alias']).reset_index(drop=True)
    df['cum_profit'] = (1 + df['trade_profit']).cumprod() - 1
    df['win'] = (df['trade_profit'] > 0).astype(int)
    df['period'] = pd.cut(
        df['trade_date'],
        bins=pd.to_datetime(PERIOD_BINS),
        labels=PERIOD_LABELS,
    )
    df['alias_q'] = pd.qcut(df['alias'], q=4, labels=['Q1\n低alias', 'Q2', 'Q3', 'Q4\n高alias'])
    df['roll_corr'] = (
        df['alias'].rolling(ROLL_WINDOW)
                   .corr(df['trade_profit'])
    )
    return df


# ------------------------------------------------------------------ #
#  统计摘要（打印到终端）                                                #
# ------------------------------------------------------------------ #

def print_summary(df: pd.DataFrame) -> None:
    print('=' * 55)
    print('  alias 与 trade_profit 分析报告')
    print('=' * 55)

    r_p, _ = pearsonr(df['alias'], df['trade_profit'])
    r_s, _ = spearmanr(df['alias'], df['trade_profit'])
    print(f'\n【全区间】n={len(df)}')
    print(f'  alias    均值={df["alias"].mean():.3f}  std={df["alias"].std():.3f}')
    print(f'  收益率   均值={df["trade_profit"].mean():.4f}  std={df["trade_profit"].std():.4f}')
    print(f'  Pearson  r = {r_p:.4f}')
    print(f'  Spearman r = {r_s:.4f}')

    print('\n【分时段】')
    for p, g in df.groupby('period', observed=True):
        if len(g) < 2:
            continue
        rs, _ = spearmanr(g['alias'], g['trade_profit'])
        print(f'  {p:<8}  n={len(g):>3}  '
              f'alias={g["alias"].mean():.2f}  '
              f'收益={g["trade_profit"].mean():+.4f}  '
              f'胜率={g["win"].mean():.1%}  '
              f'Spearman={rs:.3f}')

    print('\n【alias 四分位】')
    grp = df.groupby('alias_q', observed=True)
    for q, g in grp:
        print(f'  {str(q):<12}  alias∈[{g["alias"].min():.2f},{g["alias"].max():.2f}]  '
              f'n={len(g):>3}  '
              f'收益={g["trade_profit"].mean():+.4f}  '
              f'胜率={g["win"].mean():.1%}')

    r_at, _ = pearsonr(range(len(df)), df['alias'])
    print(f'\n【alias 时间趋势】Pearson(alias, 时序) = {r_at:.4f}  '
          f'(>0 表示随时间上升)')
    print()


# ------------------------------------------------------------------ #
#  图表 1 — 时间序列总览（3行）                                          #
# ------------------------------------------------------------------ #

def plot_timeseries(df: pd.DataFrame, save_path: str) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#f8f8f8')
    gs = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    x      = np.arange(len(df))
    x_lbl  = [d.strftime('%Y-%m-%d') for d in df['trade_date']]
    tick_step = max(1, len(x) // 15)   # 横轴最多显示 ~15 个刻度

    # ── 子图1: alias 时间序列 ─────────────────────────────────────── #
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, df['alias'], color='#4575b4', linewidth=1.8,
             marker='o', markersize=4, zorder=3, label='alias')
    ax1.axhline(ALIAS_WARN, color='#fc8d59', linewidth=1.2,
                linestyle='--', label=f'警戒线 {ALIAS_WARN}')
    ax1.axhline(ALIAS_CRIT, color='#d73027', linewidth=1.2,
                linestyle='--', label=f'极端线 {ALIAS_CRIT}')
    # 背景着色：超过 ALIAS_WARN 的区域
    ax1.fill_between(x, ALIAS_WARN, df['alias'],
                     where=(df['alias'] > ALIAS_WARN),
                     alpha=0.18, color='#d73027', label='高风险区')
    ax1.set_title('alias 随时间变化\n（alias = 第top_n名打分偏离全市场均值的标准差倍数，越高代表模型越"自信"但也越极端）',
                  fontsize=11, fontweight='bold', pad=8)
    ax1.set_ylabel('alias', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.85, ncol=4)
    ax1.set_facecolor('#f0f0f0')
    ax1.grid(axis='y', color='white', linewidth=0.5)
    ax1.set_xticks(x[::tick_step])
    ax1.set_xticklabels(x_lbl[::tick_step], rotation=45, ha='right', fontsize=7.5)

    # ── 子图2: 每期收益（柱状）+ 累计收益（折线）────────────────────── #
    ax2 = fig.add_subplot(gs[1])
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in df['trade_profit']]
    ax2.bar(x, df['trade_profit'] * 100, color=colors, width=0.7,
            linewidth=0, alpha=0.85, label='单期收益')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2r = ax2.twinx()
    ax2r.plot(x, df['cum_profit'] * 100, color='#1f77b4',
              linewidth=1.8, label='累计收益')
    ax2r.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2r.set_ylabel('累计收益 (%)', fontsize=9, color='#1f77b4')
    ax2.set_title('每批次收益率（绿=正 / 红=负）及累计净值',
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_ylabel('单期收益 (%)', fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax2.set_facecolor('#f0f0f0')
    ax2.grid(axis='y', color='white', linewidth=0.5)
    ax2.set_xticks(x[::tick_step])
    ax2.set_xticklabels(x_lbl[::tick_step], rotation=45, ha='right', fontsize=7.5)
    lines1 = [Line2D([0],[0], color='#2ca02c', lw=8, alpha=0.7),
              Line2D([0],[0], color='#d62728', lw=8, alpha=0.7),
              Line2D([0],[0], color='#1f77b4', lw=1.8)]
    ax2.legend(lines1, ['正收益', '负收益', '累计收益'],
               loc='upper left', fontsize=8.5, framealpha=0.85)

    # ── 子图3: 滚动相关 ──────────────────────────────────────────── #
    ax3 = fig.add_subplot(gs[2])
    roll = df['roll_corr'].values
    pos_mask = roll >= 0
    ax3.fill_between(x, 0, roll, where=pos_mask,
                     alpha=0.45, color='#91bfdb', label='正相关（alias高→收益高）')
    ax3.fill_between(x, 0, roll, where=~pos_mask,
                     alpha=0.45, color='#fc8d59', label='负相关（alias高→收益低）')
    ax3.plot(x, roll, color='#333333', linewidth=1.2, zorder=3)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.axhline(0.3,  color='#4575b4', linewidth=0.7, linestyle=':')
    ax3.axhline(-0.3, color='#d73027', linewidth=0.7, linestyle=':')
    ax3.set_ylim(-1, 1)
    ax3.set_title(f'滚动 {ROLL_WINDOW} 期相关系数（alias vs trade_profit）\n'
                  '橙色区域 = 高alias对应差收益；蓝色区域 = 高alias对应好收益',
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_ylabel('Pearson r', fontsize=10)
    ax3.legend(loc='lower left', fontsize=8.5, framealpha=0.85, ncol=2)
    ax3.set_facecolor('#f0f0f0')
    ax3.grid(axis='y', color='white', linewidth=0.5)
    ax3.set_xticks(x[::tick_step])
    ax3.set_xticklabels(x_lbl[::tick_step], rotation=45, ha='right', fontsize=7.5)

    fig.suptitle('alias 时间序列分析', fontsize=14, fontweight='bold', y=1.005)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'图1已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  图表 2 — 分布与相关性分析（2×2）                                      #
# ------------------------------------------------------------------ #

def plot_distribution(df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('#f8f8f8')
    fig.suptitle('alias 分布与收益相关性分析', fontsize=14,
                 fontweight='bold', y=1.01)

    # ── 左上: alias 四分位 vs 平均收益 + 胜率 ────────────────────── #
    ax = axes[0, 0]
    grp = df.groupby('alias_q', observed=True)
    q_labels  = [str(k) for k in grp.groups]
    mean_ret  = [g['trade_profit'].mean() * 100 for _, g in grp]
    win_rates = [g['win'].mean() * 100 for _, g in grp]
    xq = np.arange(len(q_labels))

    bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in mean_ret]
    bars = ax.bar(xq - 0.18, mean_ret, width=0.35, color=bar_colors,
                  alpha=0.85, label='平均收益 (%)')
    for bar, val in zip(bars, mean_ret):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8.5)
    ax.axhline(0, color='black', linewidth=0.8)

    ax2r = ax.twinx()
    ax2r.bar(xq + 0.18, win_rates, width=0.35, color='#4575b4',
             alpha=0.65, label='胜率 (%)')
    for xi, wr in zip(xq + 0.18, win_rates):
        ax2r.text(xi, wr + 0.5, f'{wr:.0f}%',
                  ha='center', va='bottom', fontsize=8.5, color='#4575b4')
    ax2r.set_ylabel('胜率 (%)', fontsize=9, color='#4575b4')
    ax2r.set_ylim(0, 110)
    ax2r.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    ax.set_xticks(xq)
    ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel('平均收益 (%)', fontsize=10)
    ax.set_title('alias 四分位 vs 平均收益 & 胜率\n（Q1=低alias，Q4=高alias）',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('#f0f0f0')
    ax.grid(axis='y', color='white', linewidth=0.5)
    handles = [
        Line2D([0],[0], color='#2ca02c', lw=8, alpha=0.85),
        Line2D([0],[0], color='#d62728', lw=8, alpha=0.85),
        Line2D([0],[0], color='#4575b4', lw=8, alpha=0.65),
    ]
    ax.legend(handles, ['正收益', '负收益', '胜率'],
              loc='lower left', fontsize=8.5, framealpha=0.85)

    # ── 右上: 散点图 alias vs trade_profit（按时段着色）─────────────── #
    ax = axes[0, 1]
    for period, g in df.groupby('period', observed=True):
        c = PERIOD_COLORS.get(str(period), '#888888')
        ax.scatter(g['alias'], g['trade_profit'] * 100,
                   color=c, alpha=0.75, s=50, label=str(period), zorder=3)

    # 全局回归线
    slope, intercept, _, _, _ = linregress(df['alias'], df['trade_profit'] * 100)
    x_fit = np.linspace(df['alias'].min(), df['alias'].max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept,
            color='#333333', linewidth=1.5, linestyle='--', label='线性回归')

    r_p, _ = pearsonr(df['alias'], df['trade_profit'])
    ax.text(0.97, 0.97, f'Pearson r = {r_p:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(ALIAS_WARN, color='#fc8d59', linewidth=1, linestyle='--',
               label=f'警戒线 {ALIAS_WARN}')
    ax.axvline(ALIAS_CRIT, color='#d73027', linewidth=1, linestyle='--',
               label=f'极端线 {ALIAS_CRIT}')
    ax.set_xlabel('alias', fontsize=10)
    ax.set_ylabel('单期收益 (%)', fontsize=10)
    ax.set_title('alias vs 单期收益散点（按时段着色）\n虚线为全区间线性回归趋势',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('#f0f0f0')
    ax.grid(color='white', linewidth=0.5)
    ax.legend(fontsize=8, framealpha=0.85, loc='upper right')

    # ── 左下: 月度 alias 均值 vs 胜率热力条形 ─────────────────────── #
    ax = axes[1, 0]
    df['ym'] = df['trade_date'].dt.to_period('M')
    monthly = df.groupby('ym').agg(
        alias_mean=('alias', 'mean'),
        win_rate=('win', 'mean'),
        profit_mean=('trade_profit', 'mean'),
        n=('trade_profit', 'count'),
    ).reset_index()
    monthly['ym_str'] = monthly['ym'].astype(str)

    xm = np.arange(len(monthly))
    bar_c = ['#2ca02c' if v >= 0 else '#d62728' for v in monthly['profit_mean']]
    ax.bar(xm, monthly['alias_mean'], color='#91bfdb', alpha=0.8,
           label='alias 月均值', zorder=2)
    ax.axhline(ALIAS_WARN, color='#fc8d59', linewidth=1.2,
               linestyle='--', label=f'警戒线 {ALIAS_WARN}')
    ax.axhline(ALIAS_CRIT, color='#d73027', linewidth=1.2,
               linestyle='--', label=f'极端线 {ALIAS_CRIT}')

    axr = ax.twinx()
    axr.plot(xm, monthly['win_rate'] * 100, color='#d73027',
             marker='o', markersize=5, linewidth=1.6, label='月胜率 (%)', zorder=3)
    axr.set_ylabel('月胜率 (%)', fontsize=9, color='#d73027')
    axr.set_ylim(-5, 120)
    axr.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    ax.set_xticks(xm)
    ax.set_xticklabels(monthly['ym_str'], rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel('alias 月均值', fontsize=10)
    ax.set_title('月度 alias 均值 vs 胜率\n蓝柱=alias；红线=当月胜率',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('#f0f0f0')
    ax.grid(axis='y', color='white', linewidth=0.5)
    handles = [
        Line2D([0],[0], color='#91bfdb', lw=8, alpha=0.8),
        Line2D([0],[0], color='#fc8d59', lw=1.2, linestyle='--'),
        Line2D([0],[0], color='#d73027', lw=1.2, linestyle='--'),
        Line2D([0],[0], color='#d73027', lw=1.6, marker='o'),
    ]
    ax.legend(handles, ['alias月均', f'警戒线{ALIAS_WARN}', f'极端线{ALIAS_CRIT}', '月胜率'],
              fontsize=8, framealpha=0.85, loc='upper left')

    # ── 右下: alias 分段平均收益 vs 阈值（violin / box 替代方案: 分组箱线图）─ #
    ax = axes[1, 1]
    bins   = [0, 1.5, 2.0, 2.5, 3.0, 99]
    labels = ['<1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '>3.0']
    df['alias_bin'] = pd.cut(df['alias'], bins=bins, labels=labels)

    box_data  = [df[df['alias_bin'] == lbl]['trade_profit'].values * 100
                 for lbl in labels]
    box_data  = [d for d in box_data if len(d) > 0]
    used_lbls = [lbl for lbl, d in zip(labels, box_data) if len(d) > 0]

    bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))
    bin_colors = ['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027']
    for patch, c in zip(bp['boxes'], bin_colors[:len(used_lbls)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # 标注各组 n 和均值
    for i, (lbl, d) in enumerate(zip(used_lbls, box_data), 1):
        ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > -20 else -12,
                f'n={len(d)}\n均值{np.mean(d):.2f}%',
                ha='center', va='bottom', fontsize=7.5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(1, len(used_lbls) + 1))
    ax.set_xticklabels(used_lbls, fontsize=9)
    ax.set_xlabel('alias 区间', fontsize=10)
    ax.set_ylabel('单期收益 (%)', fontsize=10)
    ax.set_title('alias 细分区间收益分布（箱线图）\n颜色由蓝→红对应低alias→高alias',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('#f0f0f0')
    ax.grid(axis='y', color='white', linewidth=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'图2已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  图表 3 — score 信号质量分析                                           #
# ------------------------------------------------------------------ #



    """
    读取 daily_picks.csv，聚合为每期 score 统计。
    返回列：trade_date, score_max, score_mean, score_spread, score_q10。
    如果文件不存在返回 None。
    """
    path = os.path.join(task_dir, 'daily_picks.csv')
    if not os.path.exists(path):
        return None
    picks = pd.read_csv(path)
    picks['date'] = pd.to_datetime(picks['date'].astype(str), format='%Y%m%d')
    agg = picks.groupby('date')['score'].agg(
        score_max='max',
        score_mean='mean',
        score_min='min',
        score_std='std',
    ).reset_index().rename(columns={'date': 'trade_date'})
    # top1 与 topN 末位的分差，反映内部区分度
    agg['score_spread'] = agg['score_max'] - agg['score_min']
    return agg


def plot_score_quality(df: pd.DataFrame, score_df: pd.DataFrame, save_path: str) -> None:
    """
    图3：用持仓 score 分布判断当期信号质量。

    layout 2×2：
      左上 — score_max 时序 + 阈值线 + batch_profit 叠加
      右上 — score_max 分档 vs 收益分布（箱线图）
      左下 — score_max vs batch_return 散点 + 回归
      右下 — 高/低 score 期条件统计对比
    """
    merged = pd.merge(df, score_df, on='trade_date', how='inner')
    if merged.empty:
        print('score 数据与 batch 数据无法对齐，跳过图3。')
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('#f8f8f8')
    fig.suptitle('Score 信号质量分析（持仓打分分布 → 当期可信度判断）',
                 fontsize=14, fontweight='bold', y=1.01)

    x      = np.arange(len(merged))
    x_lbl  = [d.strftime('%Y-%m-%d') for d in merged['trade_date']]
    tick_step = max(1, len(x) // 15)

    # ── 左上：score_max 时序 + 阈值 + batch_profit 柱 ─────────────── #
    ax = axes[0, 0]
    ax.plot(x, merged['score_max'], color='#4575b4', linewidth=1.8,
            marker='o', markersize=3.5, zorder=4, label='score_max')
    ax.plot(x, merged['score_mean'], color='#91bfdb', linewidth=1.2,
            linestyle='--', zorder=3, label='score_mean')
    ax.axhline(SCORE_LOW_THRESH,  color='#d73027', linewidth=1.3,
               linestyle='--', label=f'低阈值 {SCORE_LOW_THRESH}（无信号）')
    ax.axhline(SCORE_HIGH_THRESH, color='#4dac26', linewidth=1.3,
               linestyle='--', label=f'高阈值 {SCORE_HIGH_THRESH}（置信）')
    ax.fill_between(x, 0, merged['score_max'],
                    where=(merged['score_max'] < SCORE_LOW_THRESH),
                    alpha=0.20, color='#d73027', label='无信号区')
    ax.fill_between(x, SCORE_HIGH_THRESH, merged['score_max'],
                    where=(merged['score_max'] >= SCORE_HIGH_THRESH),
                    alpha=0.15, color='#4dac26', label='置信区')
    ax.set_ylabel('score', fontsize=10, color='#4575b4')
    ax.set_ylim(bottom=0)

    axr = ax.twinx()
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in merged['trade_profit']]
    axr.bar(x, merged['trade_profit'] * 100, color=colors, width=0.6,
            alpha=0.35, zorder=2)
    axr.set_ylabel('批次收益 (%)', fontsize=9, color='#555555')
    axr.axhline(0, color='black', linewidth=0.6)
    axr.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(x_lbl[::tick_step], rotation=45, ha='right', fontsize=7.5)
    ax.set_title('score_max / score_mean 时序\n红色背景=无信号区（<{:.2f}）；绿色背景=置信区（≥{:.2f}）'.format(
                 SCORE_LOW_THRESH, SCORE_HIGH_THRESH),
                 fontsize=10, fontweight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85, ncol=3)
    ax.set_facecolor('#f0f0f0')
    ax.grid(axis='y', color='white', linewidth=0.5)

    # ── 右上：score_max 分档箱线图 ──────────────────────────────────── #
    ax = axes[0, 1]
    q_lo = merged['score_max'].quantile(0.25)
    q_hi = merged['score_max'].quantile(0.75)
    bins_s   = [0, SCORE_LOW_THRESH, q_lo, q_hi, 1.0]
    labels_s = [f'<{SCORE_LOW_THRESH}\n（无信号）',
                f'{SCORE_LOW_THRESH}~{q_lo:.3f}',
                f'{q_lo:.3f}~{q_hi:.3f}',
                f'>{q_hi:.3f}\n（高置信）']
    merged['score_bin'] = pd.cut(merged['score_max'], bins=bins_s,
                                  labels=labels_s, include_lowest=True)
    box_data  = [merged[merged['score_bin'] == lbl]['trade_profit'].values * 100
                 for lbl in labels_s]
    valid     = [(lbl, d) for lbl, d in zip(labels_s, box_data) if len(d) > 0]
    v_lbls    = [v[0] for v in valid]
    v_data    = [v[1] for v in valid]

    bp = ax.boxplot(v_data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))
    seg_colors = ['#d73027', '#fc8d59', '#91bfdb', '#4dac26']
    for patch, c in zip(bp['boxes'], seg_colors[:len(v_lbls)]):
        patch.set_facecolor(c); patch.set_alpha(0.72)
    for i, (lbl, d) in enumerate(zip(v_lbls, v_data), 1):
        ax.text(i, ax.get_ylim()[0] + 0.3,
                f'n={len(d)}\n{np.mean(d):.2f}%',
                ha='center', va='bottom', fontsize=7.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(1, len(v_lbls) + 1))
    ax.set_xticklabels(v_lbls, fontsize=8.5)
    ax.set_ylabel('批次收益 (%)', fontsize=10)
    ax.set_title('score_max 分档 vs 批次收益分布\n红→绿：无信号→高置信',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('#f0f0f0')
    ax.grid(axis='y', color='white', linewidth=0.5)

    # ── 左下：score_max vs batch_return 散点 ────────────────────────── #
    ax = axes[1, 0]
    for period, g in merged.groupby('period', observed=True):
        c = PERIOD_COLORS.get(str(period), '#888888')
        ax.scatter(g['score_max'], g['trade_profit'] * 100,
                   color=c, alpha=0.78, s=55, label=str(period), zorder=3)
    # 回归线
    sl, ic, _, _, _ = linregress(merged['score_max'], merged['trade_profit'] * 100)
    xf = np.linspace(merged['score_max'].min(), merged['score_max'].max(), 100)
    ax.plot(xf, sl * xf + ic, color='#333333', linewidth=1.5,
            linestyle='--', label='回归线')
    rs, _ = spearmanr(merged['score_max'], merged['trade_profit'])
    ax.text(0.97, 0.97, f'Spearman r = {rs:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(SCORE_LOW_THRESH,  color='#d73027', linewidth=1.1,
               linestyle='--', label=f'低阈值 {SCORE_LOW_THRESH}')
    ax.axvline(SCORE_HIGH_THRESH, color='#4dac26', linewidth=1.1,
               linestyle='--', label=f'高阈值 {SCORE_HIGH_THRESH}')
    ax.set_xlabel('score_max（当期最高打分）', fontsize=10)
    ax.set_ylabel('批次收益 (%)', fontsize=10)
    ax.set_title('score_max vs 批次收益（按时段着色）',
                 fontsize=10, fontweight='bold', pad=8)
    ax.legend(fontsize=8, framealpha=0.85, loc='upper left')
    ax.set_facecolor('#f0f0f0')
    ax.grid(color='white', linewidth=0.5)

    # ── 右下：高/低 score 期的条件统计 ──────────────────────────────── #
    ax = axes[1, 1]
    thresholds = np.linspace(
        merged['score_max'].quantile(0.05),
        merged['score_max'].quantile(0.90),
        30,
    )
    win_above, win_below, ret_above, ret_below, n_above = [], [], [], [], []
    for thr in thresholds:
        above = merged[merged['score_max'] >= thr]
        below = merged[merged['score_max'] <  thr]
        win_above.append(above['win'].mean() if len(above) else np.nan)
        win_below.append(below['win'].mean() if len(below) else np.nan)
        ret_above.append(above['trade_profit'].mean() if len(above) else np.nan)
        ret_below.append(below['trade_profit'].mean() if len(below) else np.nan)
        n_above.append(len(above))

    ax.plot(thresholds, np.array(win_above) * 100, color='#4dac26',
            linewidth=1.8, label='score≥阈值 胜率')
    ax.plot(thresholds, np.array(win_below) * 100, color='#d73027',
            linewidth=1.8, linestyle='--', label='score<阈值 胜率')
    ax.axhline(50, color='black', linewidth=0.7, linestyle=':')
    ax.axvline(SCORE_LOW_THRESH,  color='#d73027', linewidth=1.1,
               linestyle='--', alpha=0.7)
    ax.axvline(SCORE_HIGH_THRESH, color='#4dac26', linewidth=1.1,
               linestyle='--', alpha=0.7)

    axr2 = ax.twinx()
    axr2.plot(thresholds, np.array(ret_above) * 100, color='#2ca02c',
              linewidth=1.4, linestyle=':', label='score≥阈值 均收益')
    axr2.plot(thresholds, np.array(ret_below) * 100, color='#d62728',
              linewidth=1.4, linestyle=':', label='score<阈值 均收益')
    axr2.axhline(0, color='gray', linewidth=0.5)
    axr2.set_ylabel('均收益 (%)', fontsize=9, color='#555555')
    axr2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))

    ax.set_xlabel('score_max 阈值', fontsize=10)
    ax.set_ylabel('胜率 (%)', fontsize=10)
    ax.set_title('阈值扫描：score_max ≥ 阈值 vs < 阈值\n实线=胜率；虚点=均收益',
                 fontsize=10, fontweight='bold', pad=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    lines1 = ax.get_lines() + axr2.get_lines()
    labels1 = [l.get_label() for l in lines1]
    ax.legend(lines1, labels1, fontsize=8, framealpha=0.85, loc='lower left', ncol=2)
    ax.set_facecolor('#f0f0f0')
    ax.grid(axis='y', color='white', linewidth=0.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'图3已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print('用法: python analyse/alias_analysis.py <task目录路径>')
        print('示例: python analyse/alias_analysis.py backtest/results/20230103_20260331_20260517_164109')
        sys.exit(1)

    task_dir = sys.argv[1]
    print(f'读取：{task_dir}')
    df = load_data(task_dir)
    print(f'共 {len(df)} 批次，时间范围 {df["trade_date"].min().date()} ~ {df["trade_date"].max().date()}')

    print_summary(df)

    out_dir = os.path.abspath(task_dir)
    plot_timeseries(df,   os.path.join(out_dir, 'alias_timeseries.png'))
    plot_distribution(df, os.path.join(out_dir, 'alias_distribution.png'))

    score_df = load_score_data(task_dir)
    if score_df is not None:
        plot_score_quality(df, score_df, os.path.join(out_dir, 'score_quality.png'))
    else:
        print('未找到 daily_picks.csv，跳过图3（score 信号质量分析）。')


if __name__ == '__main__':
    main()
