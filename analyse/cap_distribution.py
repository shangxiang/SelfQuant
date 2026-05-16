"""
analyse/cap_distribution.py

分析 daily_picks.csv 中每日选股的市值分布变化。

用法：
    python analyse/cap_distribution.py <task目录路径>

示例：
    python analyse/cap_distribution.py backtest/results/20230103_20260331_20260516_162203
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- 中文字体 ----
plt.rcParams['font.family'] = 'STHeiti'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------ #
#  配置                                                                #
# ------------------------------------------------------------------ #

TIER_DEFS = [
    ('<20亿',       0,    20,   '#d73027'),
    ('20-50亿',    20,    50,   '#fc8d59'),
    ('50-100亿',   50,   100,   '#fee090'),
    ('100-300亿', 100,   300,   '#91bfdb'),
    ('300-1000亿',300,  1000,   '#4575b4'),
    ('>1000亿',  1000, np.inf,  '#313695'),
]
TIER_LABELS  = [t[0] for t in TIER_DEFS]
TIER_COLORS  = [t[3] for t in TIER_DEFS]


def assign_tier(mv_yi: pd.Series) -> pd.Series:
    labels = pd.cut(
        mv_yi,
        bins=[t[1] for t in TIER_DEFS] + [np.inf],
        labels=TIER_LABELS,
        right=False,
    )
    return labels


def resolve_picks_path(task_dir: str) -> str:
    """将 task 目录路径解析为 daily_picks.csv 的完整路径。"""
    path = os.path.join(task_dir, 'daily_picks.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到 {path}，请确认目录路径正确。')
    return path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df['mv_yi'] = df['total_mv'] / 10000          # 万元 → 亿元
    df['tier']  = assign_tier(df['mv_yi'])
    return df


def make_figure(df: pd.DataFrame, save_path: str) -> None:
    dates = sorted(df['date'].unique())
    n = len(dates)

    # ---- 每日各档位占比 ----
    pct_matrix = pd.DataFrame(index=dates, columns=TIER_LABELS, dtype=float)
    for d in dates:
        sub = df[df['date'] == d]
        counts = sub['tier'].value_counts()
        total  = counts.sum()
        for lbl in TIER_LABELS:
            pct_matrix.loc[d, lbl] = counts.get(lbl, 0) / total * 100

    # ---- 每日市值统计（中位 + IQR）----
    stats = df.groupby('date')['mv_yi'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        q10=lambda x: x.quantile(0.10),
        q90=lambda x: x.quantile(0.90),
    )

    # ---------------------------------------------------------------- #
    #  绘图                                                              #
    # ---------------------------------------------------------------- #
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#f8f8f8')

    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])   # 堆叠面积图
    ax2 = fig.add_subplot(gs[1])   # 中位市值 + IQR

    x = np.arange(n)
    x_dates = [d.strftime('%Y-%m-%d') for d in dates]

    # ---- 上图：堆叠面积（各档位占比）----
    bottom = np.zeros(n)
    for lbl, color in zip(TIER_LABELS, TIER_COLORS):
        vals = pct_matrix[lbl].values.astype(float)
        ax1.bar(x, vals, bottom=bottom, color=color, label=lbl,
                width=0.75, linewidth=0)
        bottom += vals

    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('占比 (%)', fontsize=11)
    ax1.set_title('每日选股市值档位分布', fontsize=13, fontweight='bold', pad=10)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_dates, rotation=45, ha='right', fontsize=7.5)
    ax1.axhline(50, color='white', linewidth=0.6, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.85,
               ncol=len(TIER_LABELS), bbox_to_anchor=(0, 1.01), borderaxespad=0)
    ax1.grid(axis='y', color='white', linewidth=0.5, alpha=0.5)
    ax1.set_facecolor('#f0f0f0')

    # ---- 下图：中位市值折线 + IQR 阴影 ----
    med = stats['median'].values
    q25 = stats['q25'].values
    q75 = stats['q75'].values
    q10 = stats['q10'].values
    q90 = stats['q90'].values

    ax2.fill_between(x, q10, q90, alpha=0.15, color='#4575b4', label='P10-P90')
    ax2.fill_between(x, q25, q75, alpha=0.30, color='#4575b4', label='P25-P75')
    ax2.plot(x, med, color='#d73027', linewidth=2, marker='o',
             markersize=4, label='中位市值', zorder=3)

    # 标注中位值
    for xi, yi in zip(x, med):
        ax2.annotate(f'{yi:.0f}', xy=(xi, yi), xytext=(0, 7),
                     textcoords='offset points', ha='center',
                     fontsize=6.5, color='#d73027')

    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_dates, rotation=45, ha='right', fontsize=7.5)
    ax2.set_ylabel('市值（亿元）', fontsize=11)
    ax2.set_title('每日选股市值分布（中位 ± IQR）', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.85)
    ax2.grid(axis='y', color='#cccccc', linewidth=0.5, linestyle='--', alpha=0.7)
    ax2.set_facecolor('#f0f0f0')

    # ---- 整体注释 ----
    src_label = os.path.basename(os.path.dirname(os.path.abspath(
        df.attrs.get('source', 'daily_picks.csv'))))
    fig.text(0.99, 0.01, f'来源：{src_label}',
             ha='right', va='bottom', fontsize=8, color='#888888')

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'图表已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print('用法: python analyse/cap_distribution.py <task目录路径>')
        print('示例: python analyse/cap_distribution.py backtest/results/20230103_20260331_20260516_162203')
        sys.exit(1)

    task_dir   = sys.argv[1]
    picks_path = resolve_picks_path(task_dir)

    print(f'读取：{picks_path}')
    df = load_data(picks_path)
    df.attrs['source'] = picks_path

    print(f'共 {len(df)} 条记录，{df["date"].nunique()} 个交易日，'
          f'{df["ts_code"].nunique()} 只不同股票')
    print(f'整体中位市值：{df["mv_yi"].median():.1f} 亿')

    out_dir  = os.path.dirname(os.path.abspath(picks_path))
    out_path = os.path.join(out_dir, 'cap_distribution.png')
    make_figure(df, out_path)


if __name__ == '__main__':
    main()
