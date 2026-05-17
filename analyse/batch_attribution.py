"""
analyse/batch_attribution.py

逐批次股票收益贡献分析。

对每批 top_n 选股，计算各股的实际持仓收益率和贡献度，
并在"上涨批次"与"下跌批次"两组视角下对比贡献分布。

主要回答：
  1. 上涨批次靠几只股票拉动，还是整体普涨？
  2. 下跌批次是少数拖累还是整体下行？
  3. 买入时打分更高的股票，实际表现更好吗？

用法：
    python analyse/batch_attribution.py <task目录路径>

示例：
    python analyse/batch_attribution.py backtest/results/20230103_20260331_20260517_164109
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

plt.rcParams['font.family'] = ['STHeiti', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIG_BG = '#f8f8f8'
AX_BG  = '#f0f0f0'


# ------------------------------------------------------------------ #
#  数据加载与收益匹配                                                   #
# ------------------------------------------------------------------ #

def load_and_match(task_dir: str) -> pd.DataFrame:
    """
    合并 daily_picks 和 trade_log，为每笔买入匹配对应的卖出记录，
    计算每只股票的持仓收益率和在批次内的贡献度。

    支持止损中途卖出：同一批次内一只股票可能存在多条 SELL 记录
    （止损 + 到期），通过加权平均计算最终成交价。

    Returns
    -------
    DataFrame，每行代表一笔持股记录，包含：
        buy_date, ts_code, score, buy_price, sell_price,
        stock_return, weight, contribution,
        batch_return, batch_win
    """
    picks = pd.read_csv(os.path.join(task_dir, 'daily_picks.csv'))
    tlog  = pd.read_csv(os.path.join(task_dir, 'trade_log.csv'))

    picks['buy_date'] = picks['date'].astype(str)
    sells = tlog[tlog['side'] == 'SELL'][['date', 'ts_code', 'price', 'shares']].copy()
    sells.columns = ['sell_date', 'ts_code', 'sell_price', 'sell_shares']
    sells['sell_date'] = sells['sell_date'].astype(str)

    buy_dates = sorted(picks['buy_date'].unique())

    rows = []
    for i, bd in enumerate(buy_dates):
        batch = picks[picks['buy_date'] == bd].copy()
        # 卖出发生在本批买入之后、下一批买入之前（含）
        next_bd = buy_dates[i + 1] if i + 1 < len(buy_dates) else '99999999'
        batch_sells_raw = sells[
            (sells['sell_date'] > bd) & (sells['sell_date'] <= next_bd)
        ]

        if batch_sells_raw.empty:
            continue

        # 同一股票可能有多条 SELL（如止损后又到期卖出），取加权平均成交价
        agg = (
            batch_sells_raw
            .assign(sell_amount=lambda x: x['sell_price'] * x['sell_shares'])
            .groupby('ts_code', as_index=False)
            .agg(sell_amount=('sell_amount', 'sum'), sell_shares=('sell_shares', 'sum'))
        )
        agg['sell_price'] = agg['sell_amount'] / agg['sell_shares']
        batch_sells = agg[['ts_code', 'sell_price']]

        # 匹配每只股票的卖出价
        merged = batch.merge(batch_sells, on='ts_code', how='left')
        if merged['sell_price'].isna().all():
            continue

        # 过滤掉没有找到卖出价的股票（少数情况）
        merged = merged.dropna(subset=['sell_price'])
        if merged.empty:
            continue

        # 每只股票的持仓成本
        merged['cost'] = merged['shares'] * merged['buy_price']
        total_cost = merged['cost'].sum()
        if total_cost <= 0:
            continue

        # 等权重（按实际投入资金）
        merged['weight'] = merged['cost'] / total_cost

        # 持仓收益率（含佣金误差可忽略，已通过买入/卖出价近似）
        merged['stock_return'] = (merged['sell_price'] - merged['buy_price']) / merged['buy_price']

        # 对批次收益的贡献 = 收益率 × 资金权重
        merged['contribution'] = merged['stock_return'] * merged['weight']

        batch_return = merged['contribution'].sum()
        merged['buy_date']     = bd
        merged['batch_return'] = batch_return
        merged['batch_win']    = batch_return >= 0

        rows.append(merged[['buy_date', 'ts_code', 'score', 'buy_price',
                             'sell_price', 'stock_return', 'weight',
                             'contribution', 'batch_return', 'batch_win']])

    if not rows:
        raise ValueError('未找到可匹配的买卖记录，请确认 task 目录正确。')

    df = pd.concat(rows, ignore_index=True)
    df['buy_date'] = pd.to_datetime(df['buy_date'], format='%Y%m%d')
    return df


# ------------------------------------------------------------------ #
#  文本摘要                                                             #
# ------------------------------------------------------------------ #

def print_summary(df: pd.DataFrame) -> None:
    batches = df.groupby('buy_date')['batch_return'].first()
    n_win   = (batches >= 0).sum()
    n_loss  = (batches < 0).sum()

    stock_win = (df['stock_return'] >= 0).mean()

    print('=' * 55)
    print('  批次收益贡献分析报告')
    print('=' * 55)
    print(f'\n共 {len(batches)} 批次  |  上涨 {n_win} 批  |  下跌 {n_loss} 批')
    print(f'批次胜率: {n_win/len(batches):.1%}    '
          f'个股胜率: {stock_win:.1%}')

    for win, label in [(True, '上涨批次'), (False, '下跌批次')]:
        sub = df[df['batch_win'] == win]
        if sub.empty:
            continue
        # 贡献度最高/最低的股票
        top = sub.groupby('ts_code')['contribution'].mean().nlargest(3)
        bot = sub.groupby('ts_code')['contribution'].mean().nsmallest(3)
        # 每批次中上涨股票的比例
        win_ratio_per_batch = sub.groupby('buy_date').apply(
            lambda g: (g['stock_return'] >= 0).mean(), include_groups=False
        )
        print(f'\n【{label}】n={sub["buy_date"].nunique()} 批')
        print(f'  批内平均上涨股票占比: {win_ratio_per_batch.mean():.1%}'
              f'  (最低 {win_ratio_per_batch.min():.1%}'
              f'  最高 {win_ratio_per_batch.max():.1%})')
        print(f'  个股收益均值: {sub["stock_return"].mean():+.4f}'
              f'  std: {sub["stock_return"].std():.4f}')
        print(f'  贡献度最高股: {", ".join(top.index.tolist())}')
        print(f'  贡献度最低股: {", ".join(bot.index.tolist())}')

    r_s, _ = spearmanr(df['score'], df['stock_return'])
    r_p, _ = pearsonr(df['score'].fillna(0), df['stock_return'])
    print(f'\n【打分与收益相关性】Spearman={r_s:.4f}  Pearson={r_p:.4f}')
    print()


# ------------------------------------------------------------------ #
#  图1：逐批次贡献堆叠柱状图                                             #
# ------------------------------------------------------------------ #

def plot_batch_stacked(df: pd.DataFrame, save_path: str) -> None:
    batches = sorted(df['buy_date'].unique())
    n = len(batches)

    fig, ax = plt.subplots(figsize=(max(16, n * 0.55), 7))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    x = np.arange(n)

    for xi, bd in enumerate(batches):
        sub = df[df['buy_date'] == bd]

        # 负贡献：按绝对值升序排列（最小负贡献在零线处，最大负贡献在最底部）
        neg = sub[sub['contribution'] < 0].sort_values('contribution', ascending=False)
        # 正贡献：按绝对值升序排列（最小正贡献在零线处，最大正贡献在最顶部）
        pos = sub[sub['contribution'] >= 0].sort_values('contribution', ascending=True)

        n_neg, n_pos = len(neg), len(pos)

        bottom_neg = 0.0
        bottom_pos = 0.0

        # 绘制负贡献段：离零最近的最浅，离零最远的最深
        for j, (_, row) in enumerate(neg.iterrows()):
            c = row['contribution']
            intensity = 0.35 + 0.55 * j / max(n_neg - 1, 1)   # 0.35（浅）→ 0.90（深）
            ax.bar(xi, c * 100, bottom=bottom_neg * 100,
                   color=plt.cm.Reds(intensity), width=0.75,
                   linewidth=0.5, edgecolor='white')
            bottom_neg += c

        # 绘制正贡献段：离零最近的最浅，离零最远的最深
        for j, (_, row) in enumerate(pos.iterrows()):
            c = row['contribution']
            intensity = 0.35 + 0.55 * j / max(n_pos - 1, 1)   # 0.35（浅）→ 0.90（深）
            ax.bar(xi, c * 100, bottom=bottom_pos * 100,
                   color=plt.cm.Greens(intensity), width=0.75,
                   linewidth=0.5, edgecolor='white')
            bottom_pos += c

        # 批次总收益标注
        total = sub['contribution'].sum()
        ax.text(xi, (total * 100) + (0.12 if total >= 0 else -0.22),
                f'{total*100:.1f}', ha='center', fontsize=5.5,
                color='#333333')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [bd.strftime('%m-%d') for bd in batches],
        rotation=45, ha='right', fontsize=7
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.set_ylabel('贡献度（%）', fontsize=11)
    ax.set_title(
        '逐批次个股收益贡献堆叠图\n'
        '每格=一只股票；颜色越深=贡献/拖累越大（离零越远）；柱顶数字=批次总收益',
        fontsize=11, fontweight='bold', pad=10
    )
    ax.grid(axis='y', color='white', linewidth=0.5)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=plt.cm.Greens(0.7), label='正贡献（深=贡献大）'),
        Patch(facecolor=plt.cm.Reds(0.7),   label='负贡献（深=拖累大）'),
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.85, loc='upper left')

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    print(f'图1已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  图2：四格深度分析                                                    #
# ------------------------------------------------------------------ #

def plot_analysis(df: pd.DataFrame, save_path: str) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(FIG_BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32, figure=fig)

    # ── 左上: 上涨/下跌批次中个股收益分布（密度直方图对比）──────────── #
    ax1 = fig.add_subplot(gs[0, 0])
    win_ret  = df[df['batch_win']]['stock_return'] * 100
    loss_ret = df[~df['batch_win']]['stock_return'] * 100
    bins = np.linspace(
        min(win_ret.min(), loss_ret.min()),
        max(win_ret.max(), loss_ret.max()),
        40
    )
    ax1.hist(win_ret,  bins=bins, alpha=0.55, color='#2ca02c',
             density=True, label=f'上涨批次 (n={df[df["batch_win"]]["buy_date"].nunique()}批)')
    ax1.hist(loss_ret, bins=bins, alpha=0.55, color='#d62728',
             density=True, label=f'下跌批次 (n={df[~df["batch_win"]]["buy_date"].nunique()}批)')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.axvline(win_ret.mean(),  color='#2ca02c', linewidth=1.5,
                linestyle='--', label=f'上涨均值 {win_ret.mean():.2f}%')
    ax1.axvline(loss_ret.mean(), color='#d62728', linewidth=1.5,
                linestyle='--', label=f'下跌均值 {loss_ret.mean():.2f}%')
    ax1.set_xlabel('个股持仓收益率 (%)', fontsize=10)
    ax1.set_ylabel('密度', fontsize=10)
    ax1.set_title('上涨/下跌批次中个股收益分布\n两组分布的重叠程度反映了批次结果的可预测性',
                  fontsize=10, fontweight='bold', pad=8)
    ax1.legend(fontsize=8, framealpha=0.85)
    ax1.set_facecolor(AX_BG)
    ax1.grid(axis='y', color='white', linewidth=0.5)

    # ── 右上: 每批次中上涨股票占比（按批次盈亏排序）──────────────────── #
    ax2 = fig.add_subplot(gs[0, 1])
    batch_stats = df.groupby('buy_date').agg(
        batch_return=('batch_return', 'first'),
        win_stock_pct=('stock_return', lambda x: (x >= 0).mean()),
        batch_win=('batch_win', 'first'),
    ).sort_values('batch_return')

    colors = ['#2ca02c' if w else '#d62728' for w in batch_stats['batch_win']]
    ax2.bar(range(len(batch_stats)), batch_stats['win_stock_pct'] * 100,
            color=colors, alpha=0.75, linewidth=0)
    ax2.axhline(50, color='black', linewidth=0.8, linestyle='--', label='50% 线')

    mean_win  = batch_stats[batch_stats['batch_win']]['win_stock_pct'].mean() * 100
    mean_loss = batch_stats[~batch_stats['batch_win']]['win_stock_pct'].mean() * 100
    ax2.axhline(mean_win,  color='#2ca02c', linewidth=1.2, linestyle=':',
                label=f'上涨批均值 {mean_win:.1f}%')
    ax2.axhline(mean_loss, color='#d62728', linewidth=1.2, linestyle=':',
                label=f'下跌批均值 {mean_loss:.1f}%')

    ax2.set_xlabel('批次（按批次收益升序排列）', fontsize=10)
    ax2.set_ylabel('批内上涨股票占比 (%)', fontsize=10)
    ax2.set_title('批次内上涨股票占比\n按批次总收益从低到高排序，绿=上涨批，红=下跌批',
                  fontsize=10, fontweight='bold', pad=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2.legend(fontsize=8, framealpha=0.85, loc='upper left')
    ax2.set_facecolor(AX_BG)
    ax2.grid(axis='y', color='white', linewidth=0.5)

    # ── 左下: 打分 vs 实际收益散点 ──────────────────────────────────── #
    ax3 = fig.add_subplot(gs[1, 0])
    win_mask = df['batch_win']
    ax3.scatter(df[win_mask]['score'],  df[win_mask]['stock_return'] * 100,
                alpha=0.35, s=18, color='#2ca02c', label='上涨批次')
    ax3.scatter(df[~win_mask]['score'], df[~win_mask]['stock_return'] * 100,
                alpha=0.35, s=18, color='#d62728', label='下跌批次')

    # 整体回归线
    valid = df.dropna(subset=['score'])
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(valid['score'], valid['stock_return'] * 100)
    xs = np.linspace(valid['score'].min(), valid['score'].max(), 100)
    ax3.plot(xs, slope * xs + intercept, color='#333333',
             linewidth=1.5, linestyle='--', label='回归线')

    r_s, _ = spearmanr(valid['score'], valid['stock_return'])
    ax3.text(0.97, 0.97, f'Spearman r = {r_s:.3f}',
             transform=ax3.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax3.axhline(0, color='black', linewidth=0.6)
    ax3.set_xlabel('买入时打分 (score)', fontsize=10)
    ax3.set_ylabel('持仓收益率 (%)', fontsize=10)
    ax3.set_title('买入打分 vs 实际持仓收益\nSpearman 相关系数衡量选股打分的预测力',
                  fontsize=10, fontweight='bold', pad=8)
    ax3.legend(fontsize=8, framealpha=0.85)
    ax3.set_facecolor(AX_BG)
    ax3.grid(color='white', linewidth=0.5)

    # ── 右下: 贡献集中度——最大单笔贡献占批次总收益的比例 ─────────────── #
    ax4 = fig.add_subplot(gs[1, 1])

    def concentration(g):
        total = g['contribution'].sum()
        if abs(total) < 1e-9:
            return float('nan')
        # 最大正贡献 / 总收益（衡量"一人独撑"程度）
        top1 = g['contribution'].max()
        return top1 / total if total > 0 else g['contribution'].min() / total

    batch_conc = df.groupby('buy_date').apply(concentration, include_groups=False).dropna()
    batch_win_map = df.groupby('buy_date')['batch_win'].first()

    conc_win  = batch_conc[batch_win_map[batch_conc.index] == True]
    conc_loss = batch_conc[batch_win_map[batch_conc.index] == False]

    bins2 = np.linspace(0, 1, 21)
    ax4.hist(conc_win,  bins=bins2, alpha=0.55, color='#2ca02c',
             density=True, label=f'上涨批 均值{conc_win.mean():.2f}')
    ax4.hist(conc_loss, bins=bins2, alpha=0.55, color='#d62728',
             density=True, label=f'下跌批 均值{conc_loss.mean():.2f}')
    ax4.axvline(conc_win.mean(),  color='#2ca02c', linewidth=1.5, linestyle='--')
    ax4.axvline(conc_loss.mean(), color='#d62728', linewidth=1.5, linestyle='--')
    ax4.set_xlabel('最大单笔贡献 / 批次总收益', fontsize=10)
    ax4.set_ylabel('密度', fontsize=10)
    ax4.set_title('贡献集中度分布\n值越高说明批次收益越依赖单只股票',
                  fontsize=10, fontweight='bold', pad=8)
    ax4.legend(fontsize=8, framealpha=0.85)
    ax4.set_facecolor(AX_BG)
    ax4.grid(axis='y', color='white', linewidth=0.5)

    fig.suptitle('批次股票收益贡献深度分析', fontsize=14,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    print(f'图2已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  CSV 输出                                                            #
# ------------------------------------------------------------------ #

def save_csv(df: pd.DataFrame, save_path: str) -> None:
    out = df.copy()
    out['buy_date'] = out['buy_date'].dt.strftime('%Y%m%d')
    out[['buy_date', 'ts_code', 'score', 'stock_return',
         'weight', 'contribution', 'batch_return', 'batch_win']].to_csv(
        save_path, index=False, float_format='%.6f'
    )
    print(f'数据已保存 → {save_path}')


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print('用法: python analyse/batch_attribution.py <task目录路径>')
        sys.exit(1)

    task_dir = sys.argv[1]
    print(f'读取：{task_dir}')

    df = load_and_match(task_dir)
    print(f'匹配完成：{df["buy_date"].nunique()} 批次，{len(df)} 条持股记录')

    print_summary(df)

    out_dir = os.path.abspath(task_dir)
    save_csv(df, os.path.join(out_dir, 'batch_attribution.csv'))
    plot_batch_stacked(df, os.path.join(out_dir, 'attribution_stacked.png'))
    plot_analysis(df,      os.path.join(out_dir, 'attribution_analysis.png'))


if __name__ == '__main__':
    main()
