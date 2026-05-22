"""
analyse/holding_pattern.py

持仓期走势形态分析。

分析每批持仓在 5 个交易日（T+1 买入收盘 ~ T+6 卖出收盘）内的价格走势，
提取形态特征（A型/V型/单边上涨/单边下跌/混合），
并分析这些特征与当期及下期收益的关系，用于提前识别止损信号。

用法：
    python analyse/holding_pattern.py <task目录路径>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.stats import spearmanr

plt.rcParams['font.family'] = ['STHeiti', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SERIES_DIR = 'data/series'
CAL_FILE   = 'data/raw/trade_cal.csv'
FIG_BG     = '#f8f8f8'
AX_BG      = '#f0f0f0'

PATTERN_COLORS = {
    'monotone_up':   '#2ca02c',
    'monotone_down': '#d62728',
    'A_type':        '#ff7f0e',
    'V_type':        '#1f77b4',
    'mixed':         '#9467bd',
}
PATTERN_LABELS = {
    'monotone_up':   '单边上涨（≥4天涨）',
    'monotone_down': '单边下跌（≥4天跌）',
    'A_type':        'A型（先涨后跌）',
    'V_type':        'V型（先跌后涨）',
    'mixed':         '混合',
}


# ------------------------------------------------------------------ #
#  数据加载                                                             #
# ------------------------------------------------------------------ #

def load_trading_dates() -> list:
    cal = pd.read_csv(CAL_FILE)
    return sorted(cal[cal['is_open'] == 1]['cal_date'].astype(str).tolist())


def load_series(ts_code: str):
    path = os.path.join(SERIES_DIR, f'{ts_code}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=['trade_date', 'close_x'])
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.drop_duplicates(subset='trade_date', keep='first')
    return df.set_index('trade_date')['close_x']


def load_data(task_dir: str):
    picks = pd.read_csv(os.path.join(task_dir, 'daily_picks.csv'))
    picks['buy_date'] = picks['date'].astype(str)

    profit_path = os.path.join(task_dir, 'batch_alias_profit.csv')
    if os.path.exists(profit_path):
        profit = pd.read_csv(profit_path)
        profit['buy_date'] = profit['trade_date'].astype(str)
    else:
        tlog = pd.read_csv(os.path.join(task_dir, 'trade_log.csv'))
        buy_dates = sorted(picks['buy_date'].unique())
        rows = []
        for i, bd in enumerate(buy_dates):
            cost = tlog[(tlog['date'].astype(str) == bd) & (tlog['side'] == 'BUY')]['amount'].sum()
            next_bd = buy_dates[i + 1] if i + 1 < len(buy_dates) else '99999999'
            proceeds = tlog[
                (tlog['side'] == 'SELL') &
                (tlog['date'].astype(str) > bd) &
                (tlog['date'].astype(str) <= next_bd)
            ]['amount'].sum()
            tp = (proceeds - cost) / cost if cost > 0 else float('nan')
            rows.append({'buy_date': bd, 'trade_profit': tp})
        profit = pd.DataFrame(rows)

    return picks, profit


# ------------------------------------------------------------------ #
#  形态分类                                                             #
# ------------------------------------------------------------------ #

def classify_pattern(daily_rets: list) -> str:
    """
    5个持仓日的日收益率序列 → 走势形态。

    单边上涨：≥4天正收益
    单边下跌：≤1天正收益
    A型（先涨后跌）：前2天均涨 且 后2天均跌
    V型（先跌后涨）：前2天均跌 且 后2天均涨
    混合：其余
    """
    if len(daily_rets) < 5:
        return 'mixed'
    up = [r > 0 for r in daily_rets]
    up_count = sum(up)
    if up_count >= 4:
        return 'monotone_up'
    if up_count <= 1:
        return 'monotone_down'
    first_up = sum(up[:2])
    last_up  = sum(up[3:])
    if first_up == 2 and last_up == 0:
        return 'A_type'
    if first_up == 0 and last_up == 2:
        return 'V_type'
    return 'mixed'


# ------------------------------------------------------------------ #
#  持仓期日收益构建                                                     #
# ------------------------------------------------------------------ #

def build_holding_returns(picks: pd.DataFrame, profit: pd.DataFrame,
                          trade_dates: list) -> pd.DataFrame:
    """
    为每批每只股票构建持仓期日收益序列。

    持仓期：buy_date（T+1，买入收盘）→ buy_date 后第5个交易日（T+6，卖出收盘）
    r_d = (close[d] - close[d-1]) / close[d-1]，d=1..5
    c_d = (close[d] - close[buy]) / close[buy]，d=1..5
    """
    profit_map = dict(zip(profit['buy_date'], profit['trade_profit']))
    series_cache = {}
    rows = []

    for buy_date, group in picks.groupby('buy_date'):
        if buy_date not in trade_dates:
            continue
        idx = trade_dates.index(buy_date)
        hold_dates = trade_dates[idx: idx + 6]   # T+1 .. T+6（共6个日期，5段收益）
        if len(hold_dates) < 6:
            continue

        batch_return = profit_map.get(buy_date, float('nan'))

        for _, row in group.iterrows():
            ts = row['ts_code']
            if ts not in series_cache:
                series_cache[ts] = load_series(ts)
            ser = series_cache[ts]
            if ser is None:
                continue

            closes = []
            for d in hold_dates:
                val = ser.get(d, float('nan'))
                closes.append(float(val))
            if any(np.isnan(c) for c in closes):
                continue

            daily_rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, 6)]
            cum_rets   = [(closes[i] - closes[0])   / closes[0]   for i in range(1, 6)]

            rows.append({
                'buy_date':     buy_date,
                'ts_code':      ts,
                'score':        row.get('score', float('nan')),
                'buy_price':    row.get('buy_price', float('nan')),
                'r1': daily_rets[0], 'r2': daily_rets[1], 'r3': daily_rets[2],
                'r4': daily_rets[3], 'r5': daily_rets[4],
                'c1': cum_rets[0],   'c2': cum_rets[1],   'c3': cum_rets[2],
                'c4': cum_rets[3],   'c5': cum_rets[4],
                'up_days':      sum(1 for r in daily_rets if r > 0),
                'pattern':      classify_pattern(daily_rets),
                'batch_return': batch_return,
                'batch_win':    batch_return >= 0 if not pd.isna(batch_return) else False,
            })

    df = pd.DataFrame(rows)
    df['buy_date'] = pd.to_datetime(df['buy_date'], format='%Y%m%d')
    return df


# ------------------------------------------------------------------ #
#  批次特征聚合                                                         #
# ------------------------------------------------------------------ #

def build_batch_features(holding_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bd, g in holding_df.groupby('buy_date'):
        row = {
            'buy_date':     bd,
            'batch_return': g['batch_return'].iloc[0],
            'batch_win':    g['batch_win'].iloc[0],
            'n_stocks':     len(g),
        }
        for d in range(1, 6):
            row[f'day{d}_up_pct'] = (g[f'r{d}'] > 0).mean()
        for pat in PATTERN_LABELS:
            row[f'pct_{pat}'] = (g['pattern'] == pat).mean()
        row['avg_up_days'] = g['up_days'].mean()
        row['avg_c1']      = g['c1'].mean()
        row['avg_c2']      = g['c2'].mean()
        row['avg_c5']      = g['c5'].mean()
        rows.append(row)

    feat = pd.DataFrame(rows).sort_values('buy_date').reset_index(drop=True)
    feat['next_batch_return'] = feat['batch_return'].shift(-1)
    return feat


# ------------------------------------------------------------------ #
#  文本摘要                                                             #
# ------------------------------------------------------------------ #

def print_summary(holding_df: pd.DataFrame, feat: pd.DataFrame) -> None:
    print('=' * 60)
    print('  持仓期走势形态分析报告')
    print('=' * 60)
    print(f'\n共 {holding_df["buy_date"].nunique()} 批次，{len(holding_df)} 条持股记录')

    pat_counts = holding_df['pattern'].value_counts()
    print('\n【个股走势形态分布】')
    for pat, label in PATTERN_LABELS.items():
        n = pat_counts.get(pat, 0)
        print(f'  {label:<22}: {n:>4} 只  {n/len(holding_df):.1%}')

    print('\n【形态 vs 批次收益（均值）】')
    for pat, label in PATTERN_LABELS.items():
        sub = holding_df[holding_df['pattern'] == pat]
        if sub.empty:
            continue
        print(f'  {label:<22}: 批次收益 {sub["batch_return"].mean():+.4f}  '
              f'个股c5 {sub["c5"].mean():+.4f}  n={len(sub)}')

    print('\n【持仓期每日上涨股票占比（全批次均值）】')
    win  = feat[feat['batch_win']]
    loss = feat[~feat['batch_win']]
    for d in range(1, 6):
        col = f'day{d}_up_pct'
        print(f'  Day{d}: 全体 {feat[col].mean():.1%}  '
              f'上涨批 {win[col].mean():.1%}  下跌批 {loss[col].mean():.1%}')

    valid = feat.dropna(subset=['next_batch_return'])
    print('\n【早期预警 Spearman 相关性】')
    for col, desc in [('day1_up_pct', 'Day1上涨占比'), ('avg_c2', 'Day2累计收益均值')]:
        r_cur,  _ = spearmanr(feat[col],  feat['batch_return'])
        r_next, _ = spearmanr(valid[col], valid['next_batch_return'])
        print(f'  {desc:<16} vs 当期: {r_cur:+.4f}   vs 下期: {r_next:+.4f}')
    print()


# ------------------------------------------------------------------ #
#  图1：每日上涨占比热力图 + 时序折线                                    #
# ------------------------------------------------------------------ #

def plot_daily_updown(feat: pd.DataFrame, save_path: str) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(FIG_BG)
    gs = gridspec.GridSpec(2, 1, hspace=0.5, figure=fig, height_ratios=[1.2, 1])

    batches   = feat['buy_date'].tolist()
    n         = len(batches)
    day_cols  = [f'day{d}_up_pct' for d in range(1, 6)]
    matrix    = feat[day_cols].values.T   # shape (5, n_batches)

    # ── 热力图 ──────────────────────────────────────────────────────── #
    ax1 = fig.add_subplot(gs[0])
    im  = ax1.imshow(matrix, aspect='auto', cmap='RdYlGn',
                     vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01, label='上涨股票占比')
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([f'Day{d}' for d in range(1, 6)], fontsize=9)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([bd.strftime('%m-%d') for bd in batches],
                        rotation=45, ha='right', fontsize=6.5)
    for xi, row in feat.iterrows():
        marker = '▲' if row['batch_win'] else '▼'
        color  = '#1a6b1a' if row['batch_win'] else '#8b0000'
        ax1.text(xi, -0.75, marker, ha='center', va='top', fontsize=7, color=color)
    ax1.set_title('持仓期每日上涨股票占比热力图\n'
                  '绿=多数上涨，红=多数下跌；▲=批次盈利，▼=批次亏损',
                  fontsize=11, fontweight='bold', pad=8)

    # ── Day1/Day2 时序 + 批次收益柱 ─────────────────────────────────── #
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(AX_BG)
    x = range(n)
    ax2.plot(x, feat['day1_up_pct'] * 100, color='#1f77b4', lw=1.5,
             marker='o', ms=4, label='Day1 上涨占比')
    ax2.plot(x, feat['day2_up_pct'] * 100, color='#ff7f0e', lw=1.5,
             marker='s', ms=4, label='Day2 上涨占比', alpha=0.85)
    ax2.axhline(50, color='black', lw=0.8, linestyle='--')
    ax2.set_ylabel('上涨占比 (%)', fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2.set_xticks(x)
    ax2.set_xticklabels([bd.strftime('%m-%d') for bd in batches],
                        rotation=45, ha='right', fontsize=6.5)
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax2.grid(axis='y', color='white', lw=0.5)

    ax2r = ax2.twinx()
    colors = ['#2ca02c' if w else '#d62728' for w in feat['batch_win']]
    ax2r.bar(x, feat['batch_return'] * 100, color=colors, alpha=0.3, width=0.6)
    ax2r.set_ylabel('批次收益 (%)', fontsize=9)
    ax2r.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax2.set_title('Day1/Day2 上涨占比时序（早期止损信号）\n'
                  '柱状背景=批次收益（绿盈/红亏）',
                  fontsize=11, fontweight='bold', pad=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    print(f'图1已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  图2：形态分布 + 预测分析（2×2）                                      #
# ------------------------------------------------------------------ #

def plot_pattern_analysis(holding_df: pd.DataFrame, feat: pd.DataFrame,
                          save_path: str) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(FIG_BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32, figure=fig)

    # ── 左上：形态分布堆叠柱（按批次，上涨/下跌批次对比）──────────────── #
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(AX_BG)
    pat_keys = list(PATTERN_LABELS.keys())
    win_pcts  = [holding_df[holding_df['batch_win']][holding_df['pattern'] == p].shape[0] /
                 max(holding_df[holding_df['batch_win']].shape[0], 1) for p in pat_keys]
    loss_pcts = [holding_df[~holding_df['batch_win']][holding_df['pattern'] == p].shape[0] /
                 max(holding_df[~holding_df['batch_win']].shape[0], 1) for p in pat_keys]
    x = np.arange(len(pat_keys))
    w = 0.35
    ax1.bar(x - w/2, [v*100 for v in win_pcts],  w, color='#2ca02c', alpha=0.75, label='上涨批次')
    ax1.bar(x + w/2, [v*100 for v in loss_pcts], w, color='#d62728', alpha=0.75, label='下跌批次')
    ax1.set_xticks(x)
    ax1.set_xticklabels([PATTERN_LABELS[p].split('（')[0] for p in pat_keys],
                        fontsize=8, rotation=15, ha='right')
    ax1.set_ylabel('占比 (%)', fontsize=10)
    ax1.set_title('上涨/下跌批次中各形态占比\n形态分布差异反映批次盈亏的结构性特征',
                  fontsize=10, fontweight='bold', pad=8)
    ax1.legend(fontsize=9, framealpha=0.85)
    ax1.grid(axis='y', color='white', lw=0.5)

    # ── 右上：个股 up_days 分布（上涨批 vs 下跌批）────────────────────── #
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(AX_BG)
    win_up  = holding_df[holding_df['batch_win']]['up_days']
    loss_up = holding_df[~holding_df['batch_win']]['up_days']
    bins = np.arange(-0.5, 6.5, 1)
    ax2.hist(win_up,  bins=bins, alpha=0.6, color='#2ca02c', density=True,
             label=f'上涨批次 均值{win_up.mean():.1f}天')
    ax2.hist(loss_up, bins=bins, alpha=0.6, color='#d62728', density=True,
             label=f'下跌批次 均值{loss_up.mean():.1f}天')
    ax2.set_xlabel('持仓期内上涨天数', fontsize=10)
    ax2.set_ylabel('密度', fontsize=10)
    ax2.set_xticks(range(6))
    ax2.set_title('个股持仓期上涨天数分布\n上涨批次中个股普遍涨天更多',
                  fontsize=10, fontweight='bold', pad=8)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(axis='y', color='white', lw=0.5)

    # ── 左下：Day1上涨占比 / Day2累计收益均值 vs 当期/下期收益散点 ──────── #
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(AX_BG)
    valid = feat.dropna(subset=['next_batch_return'])

    # Day1 上涨占比（横轴）
    ax3.scatter(feat['day1_up_pct'] * 100, feat['batch_return'] * 100,
                color='#1f77b4', alpha=0.7, s=40, label='Day1占比 vs 当期')
    ax3.scatter(valid['day1_up_pct'] * 100, valid['next_batch_return'] * 100,
                color='#1f77b4', alpha=0.3, s=30, marker='^', label='Day1占比 vs 下期')

    ax3r = ax3.twiny()
    # Day2 累计收益均值（上横轴）
    ax3r.scatter(feat['avg_c2'] * 100, feat['batch_return'] * 100,
                 color='#ff7f0e', alpha=0.7, s=40, marker='s', label='Day2累计 vs 当期')
    ax3r.scatter(valid['avg_c2'] * 100, valid['next_batch_return'] * 100,
                 color='#ff7f0e', alpha=0.3, s=30, marker='D', label='Day2累计 vs 下期')
    ax3r.set_xlabel('Day2 累计收益均值 (%)', fontsize=9, color='#ff7f0e')
    ax3r.tick_params(axis='x', colors='#ff7f0e', labelsize=8)

    ax3.axhline(0, color='black', lw=0.7)
    ax3.axvline(50, color='#1f77b4', lw=0.7, linestyle='--', alpha=0.5)
    ax3r.axvline(0, color='#ff7f0e', lw=0.7, linestyle='--', alpha=0.5)

    r_d1_cur,  _ = spearmanr(feat['day1_up_pct'], feat['batch_return'])
    r_d1_next, _ = spearmanr(valid['day1_up_pct'], valid['next_batch_return'])
    r_c2_cur,  _ = spearmanr(feat['avg_c2'], feat['batch_return'])
    r_c2_next, _ = spearmanr(valid['avg_c2'], valid['next_batch_return'])
    ax3.text(0.03, 0.97,
             f'Day1占比  当期 r={r_d1_cur:+.3f}  下期 r={r_d1_next:+.3f}\n'
             f'Day2累计  当期 r={r_c2_cur:+.3f}  下期 r={r_c2_next:+.3f}',
             transform=ax3.transAxes, va='top', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, framealpha=0.85,
               loc='lower right')
    ax3.set_xlabel('Day1 上涨股票占比 (%)', fontsize=10)
    ax3.set_ylabel('收益率 (%)', fontsize=10)
    ax3.set_title('Day1占比 / Day2累计收益 vs 当期/下期收益\n蓝=Day1占比，橙=Day2累计；△▽=下期',
                  fontsize=10, fontweight='bold', pad=8)
    ax3.grid(color='white', lw=0.5)

    # ── 右下：累计收益路径（上涨批 vs 下跌批，均值±std）────────────────── #
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(AX_BG)
    days = range(1, 6)
    for win, color, label in [(True, '#2ca02c', '上涨批次'), (False, '#d62728', '下跌批次')]:
        sub = holding_df[holding_df['batch_win'] == win]
        means = [sub[f'c{d}'].mean() * 100 for d in days]
        stds  = [sub[f'c{d}'].std()  * 100 for d in days]
        ax4.plot(days, means, color=color, lw=2, marker='o', ms=5, label=label)
        ax4.fill_between(days,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=color, alpha=0.15)
    ax4.axhline(0, color='black', lw=0.8)
    ax4.set_xlabel('持仓日（Day1=T+2，Day5=T+6）', fontsize=10)
    ax4.set_ylabel('累计收益率 (%)', fontsize=10)
    ax4.set_xticks(list(days))
    ax4.set_title('持仓期累计收益路径（均值±1σ）\n上涨批次与下跌批次的分化时点',
                  fontsize=10, fontweight='bold', pad=8)
    ax4.legend(fontsize=9, framealpha=0.85)
    ax4.grid(color='white', lw=0.5)

    fig.suptitle('持仓期走势形态深度分析', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    print(f'图2已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  图3：模型失效检测（当期特征 vs 下期收益时序）                          #
# ------------------------------------------------------------------ #

def plot_staleness(feat: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.patch.set_facecolor(FIG_BG)
    fig.subplots_adjust(hspace=0.35)

    n = len(feat)
    x = range(n)
    xlabels = [bd.strftime('%m-%d') for bd in feat['buy_date']]

    # ── 子图1：批次收益时序（当期 + 下期）──────────────────────────────── #
    ax = axes[0]
    ax.set_facecolor(AX_BG)
    ax.bar(x, feat['batch_return'] * 100,
           color=['#2ca02c' if w else '#d62728' for w in feat['batch_win']],
           alpha=0.7, width=0.6, label='当期收益')
    ax.plot(x, feat['next_batch_return'] * 100, color='#ff7f0e',
            lw=1.5, marker='D', ms=4, label='下期收益', zorder=3)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('收益率 (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.legend(fontsize=9, framealpha=0.85, loc='upper left')
    ax.set_title('当期收益 vs 下期收益时序\n连续两期亏损区间=模型可能失效',
                 fontsize=10, fontweight='bold', pad=6)
    ax.grid(axis='y', color='white', lw=0.5)

    # ── 子图2：Day1上涨占比 + Day2累计收益均值 时序 ─────────────────────── #
    ax = axes[1]
    ax.set_facecolor(AX_BG)
    ax.bar(x, feat['day1_up_pct'] * 100,
           color=['#2ca02c' if v >= 0.5 else '#d62728' for v in feat['day1_up_pct']],
           alpha=0.5, width=0.6, label='Day1上涨占比')
    roll5 = feat['day1_up_pct'].rolling(5, min_periods=2).mean() * 100
    ax.plot(x, roll5, color='#1f77b4', lw=1.8, label='Day1占比 滚动5期均值')
    ax.axhline(50, color='black', lw=0.8, linestyle='--')
    ax.set_ylabel('Day1 上涨占比 (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    ax_r = ax.twinx()
    ax_r.plot(x, feat['avg_c2'] * 100, color='#ff7f0e', lw=1.8,
              marker='s', ms=4, label='Day2累计收益均值')
    roll_c2 = feat['avg_c2'].rolling(5, min_periods=2).mean() * 100
    ax_r.plot(x, roll_c2, color='#d6550d', lw=1.2, linestyle='--',
              label='Day2累计 滚动5期均值')
    ax_r.axhline(0, color='#ff7f0e', lw=0.6, linestyle=':')
    ax_r.set_ylabel('Day2 累计收益均值 (%)', fontsize=9, color='#ff7f0e')
    ax_r.tick_params(axis='y', colors='#ff7f0e', labelsize=8)
    ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.85,
              loc='upper left', ncol=2)
    ax.grid(axis='y', color='white', lw=0.5)
    ax.set_title('Day1上涨占比（蓝柱）+ Day2累计收益均值（橙线）时序\n'
                 '两者持续低迷=持仓期整体走弱，可考虑提前止损',
                 fontsize=10, fontweight='bold', pad=6)

    # ── 子图3：A型占比（先涨后跌，最危险形态）时序 ──────────────────────── #
    ax = axes[2]
    ax.set_facecolor(AX_BG)
    ax.bar(x, feat['pct_A_type'] * 100, color='#ff7f0e', alpha=0.75, width=0.6,
           label='A型占比')
    ax.bar(x, feat['pct_V_type'] * 100, color='#1f77b4', alpha=0.75, width=0.6,
           bottom=feat['pct_A_type'] * 100, label='V型占比')
    roll_a = feat['pct_A_type'].rolling(5, min_periods=2).mean() * 100
    ax.plot(x, roll_a, color='#8b4513', lw=1.8, label='A型滚动5期均值')
    ax.set_ylabel('形态占比 (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_title('A型（先涨后跌）/ V型（先跌后涨）占比时序\nA型占比持续偏高=买入时机偏早，需提前止损',
                 fontsize=10, fontweight='bold', pad=6)
    ax.grid(axis='y', color='white', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)

    fig.suptitle('模型失效早期预警分析', fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    print(f'图3已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  CSV 输出                                                            #
# ------------------------------------------------------------------ #

def save_csv(holding_df: pd.DataFrame, feat: pd.DataFrame, out_dir: str) -> None:
    h = holding_df.copy()
    h['buy_date'] = h['buy_date'].dt.strftime('%Y%m%d')
    h.to_csv(os.path.join(out_dir, 'holding_pattern_stock.csv'),
             index=False, float_format='%.6f')

    f = feat.copy()
    f['buy_date'] = f['buy_date'].dt.strftime('%Y%m%d')
    f.to_csv(os.path.join(out_dir, 'holding_pattern_batch.csv'),
             index=False, float_format='%.6f')
    print(f'CSV已保存 → {out_dir}')


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print('用法: python analyse/holding_pattern.py <task目录路径>')
        sys.exit(1)

    task_dir = sys.argv[1]
    print(f'读取：{task_dir}')

    trade_dates = load_trading_dates()
    picks, profit = load_data(task_dir)

    holding_df = build_holding_returns(picks, profit, trade_dates)
    print(f'构建完成：{holding_df["buy_date"].nunique()} 批次，{len(holding_df)} 条持股记录')

    feat = build_batch_features(holding_df)
    print_summary(holding_df, feat)

    out_dir = os.path.abspath(task_dir)
    save_csv(holding_df, feat, out_dir)
    plot_daily_updown(feat,                    os.path.join(out_dir, 'holding_daily_updown.png'))
    plot_pattern_analysis(holding_df, feat,    os.path.join(out_dir, 'holding_pattern_analysis.png'))
    plot_staleness(feat,                       os.path.join(out_dir, 'holding_staleness.png'))


if __name__ == '__main__':
    main()
