"""
analyse/timing_alignment.py

择时策略对齐度分析。

理论上，一个完美的择时策略，其"空仓时间段"应与回测 NAV 的下跌时间段完美重合。
本脚本对此进行可视化和定量评估：

  - 上图：策略 NAV 走势，叠加各择时策略的"空仓期"色块
  - 中图：每批次实际收益 + 择时信号，观察错误分类情况
  - 下图：混淆矩阵统计（各策略对"亏损批次"的召回率/精确率/F1）

用法：
    python analyse/timing_alignment.py [nav_dir]

    nav_dir 默认读取最新一次 LGBM 回测结果。
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = ['STHeiti', 'Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------ #
#  配置                                                                #
# ------------------------------------------------------------------ #

NAV_DIR  = 'backtest/results/LGBM_无择时'
BATCH_FILE  = os.path.join(NAV_DIR, 'batch_alias_profit.csv')
NAV_FILE    = os.path.join(NAV_DIR, 'nav.csv')

SMALL_FILE = 'data/raw/index_daily/932000.CSI.csv'
LARGE_FILE = 'data/raw/index_daily/000510.CSI.csv'

# 每种择时策略的配置：(label, callable -> {date_str: ratio})
# 在 build_timing_maps() 中统一实例化，避免重复读取文件

LOSS_THRESHOLD = 0.0   # 批次收益 < 此值 → 视为"亏损批次"


# ------------------------------------------------------------------ #
#  数据加载                                                            #
# ------------------------------------------------------------------ #

def load_nav() -> pd.DataFrame:
    nav = pd.read_csv(NAV_FILE)
    nav['date'] = pd.to_datetime(nav['date'].astype(str), format='%Y%m%d')
    nav = nav.sort_values('date').reset_index(drop=True)
    nav['drawdown'] = (nav['nav'] - nav['nav'].cummax()) / nav['nav'].cummax()
    return nav


def load_batches() -> pd.DataFrame:
    bap = pd.read_csv(BATCH_FILE)
    bap['trade_date'] = pd.to_datetime(bap['trade_date'].astype(str), format='%Y%m%d')
    bap['loss'] = bap['trade_profit'] < LOSS_THRESHOLD
    return bap


def build_timing_maps() -> dict:
    """实例化所有择时策略，返回 {label: {date_str: ratio}} 映射字典。"""
    sys.path.insert(0, '.')
    from backtest.timing import (
        BlindWindowTiming, MATiming, StyleConvergenceTiming, HybridTiming
    )

    start, end = '20200101', '20260522'

    strategies = {
        'BlindWindow\n(corr20+vol33+ma20)': BlindWindowTiming(
            small_file=SMALL_FILE, large_file=LARGE_FILE,
            corr_pct=20.0, vol_pct=33.0, avoid_ratio=0.0,
            rolling_window=80, trend_ma=20,
        ),
        'MATiming\n(MA60)': MATiming(
            index_file=SMALL_FILE, ma_period=60,
        ),
        'StyleConvergence\n(corr>0.75)': StyleConvergenceTiming(
            small_file=SMALL_FILE, large_file=LARGE_FILE,
            roll_window=5, corr_threshold=0.75, avoid_ratio=0.0,
        ),
        'Hybrid\n(BW+SC)': HybridTiming(
            small_file=SMALL_FILE, large_file=LARGE_FILE,
            corr_pct=20.0, vol_pct=33.0, rolling_window=80,
            roll_window=5, corr_threshold=0.75,
            partial_blind=0.3, partial_style=0.5,
        ),
    }

    # MATiming 当前 hardcode return 1.0，这里临时 patch 恢复原始逻辑
    _ma = strategies['MATiming\n(MA60)']
    _ma.prepare(start, end)
    def _ma_ratio(date_str, _map=_ma._map):
        return _map.get(date_str, 1.0)
    strategies['MATiming\n(MA60)'] = _ma_ratio   # type: ignore

    maps = {}
    for label, strat in strategies.items():
        if callable(strat) and not hasattr(strat, 'prepare'):
            # 已经是函数（patch 过的 MATiming）
            maps[label] = strat
        else:
            strat.prepare(start, end)
            maps[label] = strat.get_position_ratio

    return maps


# ------------------------------------------------------------------ #
#  指标计算                                                            #
# ------------------------------------------------------------------ #

def compute_metrics(batches: pd.DataFrame, get_ratio) -> dict:
    """
    对一批批次，计算择时策略的过滤质量。

    分类定义（以择时是否投资 vs 批次是否亏损）：
      TP = 亏损批次 & 择时空仓（正确拒绝）
      FN = 亏损批次 & 择时满仓（漏掉了亏损）
      FP = 盈利批次 & 择时空仓（错误拒绝）
      TN = 盈利批次 & 择时满仓（正确放行）

    召回率  = TP / (TP + FN)  → 在亏损批次里，成功拦截的比例（越高越好）
    精确率  = TP / (TP + FP)  → 在所有被拦截的批次里，真正亏损的比例（越高越好）
    F1      = 2 * P * R / (P + R)
    避免的亏损均值 = 被正确拦截的批次的平均收益（负数，绝对值越大说明拦截质量越高）
    """
    rows = []
    for _, row in batches.iterrows():
        date_str = row['trade_date'].strftime('%Y%m%d')
        ratio = get_ratio(date_str)
        invest = ratio > 0
        rows.append({'loss': row['loss'], 'invest': invest,
                     'profit': row['trade_profit']})
    df = pd.DataFrame(rows)

    TP = ((df['loss']) & (~df['invest'])).sum()
    FN = ((df['loss']) & ( df['invest'])).sum()
    FP = ((~df['loss']) & (~df['invest'])).sum()
    TN = ((~df['loss']) & ( df['invest'])).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avoided_mean = df.loc[(df['loss']) & (~df['invest']), 'profit'].mean()
    missed_mean  = df.loc[(df['loss']) & ( df['invest']), 'profit'].mean()
    avoid_rate   = (~df['invest']).mean()

    return dict(TP=TP, FN=FN, FP=FP, TN=TN,
                precision=precision, recall=recall, f1=f1,
                avoid_rate=avoid_rate,
                avoided_mean=avoided_mean, missed_mean=missed_mean)


# ------------------------------------------------------------------ #
#  可视化                                                              #
# ------------------------------------------------------------------ #

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']


def _shade_avoid(ax, nav: pd.DataFrame, get_ratio, color: str, alpha: float = 0.18):
    """在 ax 上叠加择时策略的"空仓期"色块。"""
    prev_avoid = False
    seg_start  = None
    for _, row in nav.iterrows():
        d = row['date']
        ds = d.strftime('%Y%m%d')
        avoid = get_ratio(ds) == 0.0
        if avoid and not prev_avoid:
            seg_start = d
        elif not avoid and prev_avoid and seg_start is not None:
            ax.axvspan(seg_start, d, alpha=alpha, color=color, linewidth=0)
        prev_avoid = avoid
    if prev_avoid and seg_start is not None:
        ax.axvspan(seg_start, nav['date'].iloc[-1], alpha=alpha, color=color, linewidth=0)


def _shade_drawdown(ax, nav: pd.DataFrame, alpha: float = 0.12):
    """在 ax 上叠加 NAV 实际回撤期（灰色）。"""
    in_dd   = False
    dd_start = None
    for _, row in nav.iterrows():
        d  = row['date']
        dd = row['drawdown'] < -0.005   # 忽略微小波动
        if dd and not in_dd:
            dd_start = d
        elif not dd and in_dd and dd_start is not None:
            ax.axvspan(dd_start, d, alpha=alpha, color='#aaaaaa', linewidth=0)
        in_dd = dd
    if in_dd and dd_start is not None:
        ax.axvspan(dd_start, nav['date'].iloc[-1], alpha=alpha, color='#aaaaaa', linewidth=0)


def plot(nav: pd.DataFrame, batches: pd.DataFrame, timing_maps: dict,
         metrics_df: pd.DataFrame, save_path: str) -> None:

    n_strats = len(timing_maps)
    fig = plt.figure(figsize=(18, 5 + 3 * n_strats))
    fig.patch.set_facecolor('#f8f8f8')

    # 行比例：NAV 图 2.5, 每个策略的批次图 1.5, 指标表 1.0
    ratios = [2.5] + [1.5] * n_strats + [1.2]
    gs = gridspec.GridSpec(1 + n_strats + 1, 1,
                           hspace=0.55, height_ratios=ratios, figure=fig)

    def fmt_ax(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
        ax.set_facecolor('#f0f0f0')
        ax.grid(axis='y', color='white', linewidth=0.5)
        ax.set_xlim(nav['date'].iloc[0], nav['date'].iloc[-1])

    # ── 子图0: NAV + 所有择时空仓区 + 实际回撤区 ──────────────────── #
    ax0 = fig.add_subplot(gs[0])
    initial = nav['nav'].iloc[0]
    ax0.plot(nav['date'], nav['nav'] / initial, color='#333333',
             linewidth=1.8, label='策略净值', zorder=5)

    _shade_drawdown(ax0, nav, alpha=0.20)

    patches = [mpatches.Patch(facecolor='#aaaaaa', alpha=0.35, label='实际回撤期')]
    for i, (label, get_ratio) in enumerate(timing_maps.items()):
        _shade_avoid(ax0, nav, get_ratio, COLORS[i], alpha=0.20)
        short = label.split('\n')[0]
        patches.append(mpatches.Patch(facecolor=COLORS[i], alpha=0.35, label=f'空仓期: {short}'))

    ax0.axhline(1.0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax0.set_ylabel('净值（归一）', fontsize=10)
    ax0.set_title('策略净值 vs 各择时策略空仓期（灰=实际回撤期）',
                  fontsize=11, fontweight='bold')
    handles_l, labels_l = ax0.get_legend_handles_labels()
    ax0.legend(handles_l + patches, labels_l + [p.get_label() for p in patches],
               fontsize=8, framealpha=0.85, loc='upper left', ncol=2)
    fmt_ax(ax0)

    # ── 子图1..n: 每个策略的批次收益 + 择时信号 ─────────────────────── #
    for i, (label, get_ratio) in enumerate(timing_maps.items()):
        ax = fig.add_subplot(gs[i + 1])
        m = metrics_df.loc[label]

        for _, row in batches.iterrows():
            d     = row['trade_date']
            ret   = row['trade_profit']
            ds    = d.strftime('%Y%m%d')
            ratio = get_ratio(ds)
            invest = ratio > 0

            if invest:
                bar_color = '#d73027' if ret < 0 else '#1a9850'
                ax.bar(d, ret * 100, width=3, color=bar_color, alpha=0.85, zorder=3)
            else:
                # 择时空仓：用淡色标注，正确/错误分开
                bar_color = '#4daf4a' if ret < 0 else '#e6ab02'
                ax.bar(d, ret * 100, width=3, color=bar_color, alpha=0.5,
                       hatch='//', zorder=3)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('批次收益 (%)', fontsize=9)
        title = (
            f'{label.replace(chr(10), "  ")}   '
            f'召回率={m["recall"]:.0%}  精确率={m["precision"]:.0%}  F1={m["f1"]:.0%}  '
            f'空仓率={m["avoid_rate"]:.0%}  '
            f'拦截亏损均值={m["avoided_mean"]*100:.2f}%  漏掉亏损均值={m["missed_mean"]*100:.2f}%'
        )
        ax.set_title(title, fontsize=9, fontweight='bold')

        legend_patches = [
            mpatches.Patch(color='#d73027', alpha=0.85, label='投资→亏损（FN）'),
            mpatches.Patch(color='#1a9850', alpha=0.85, label='投资→盈利（TN）'),
            mpatches.Patch(color='#4daf4a', alpha=0.5, label='空仓→正确拦截（TP）', hatch='//'),
            mpatches.Patch(color='#e6ab02', alpha=0.5, label='空仓→错误拦截（FP）', hatch='//'),
        ]
        ax.legend(handles=legend_patches, fontsize=7.5, framealpha=0.85,
                  loc='upper left', ncol=2)
        fmt_ax(ax)

    # ── 最后一行: 指标汇总表 ──────────────────────────────────────────── #
    ax_t = fig.add_subplot(gs[-1])
    ax_t.axis('off')

    cols = ['召回率\n(Recall)', '精确率\n(Precision)', 'F1',
            '空仓率', '拦截亏损均值', '漏掉亏损均值',
            'TP', 'FN', 'FP', 'TN']
    rows_data = []
    row_labels = []
    for label, _ in timing_maps.items():
        m = metrics_df.loc[label]
        rows_data.append([
            f'{m["recall"]:.1%}', f'{m["precision"]:.1%}', f'{m["f1"]:.1%}',
            f'{m["avoid_rate"]:.1%}',
            f'{m["avoided_mean"]*100:.2f}%' if not np.isnan(m["avoided_mean"]) else 'N/A',
            f'{m["missed_mean"]*100:.2f}%'  if not np.isnan(m["missed_mean"])  else 'N/A',
            int(m['TP']), int(m['FN']), int(m['FP']), int(m['TN']),
        ])
        row_labels.append(label.replace('\n', ' '))

    table = ax_t.table(
        cellText=rows_data,
        rowLabels=row_labels,
        colLabels=cols,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    ax_t.set_title('各择时策略对"亏损批次"的拦截质量汇总\n'
                   '（TP=正确拒绝亏损批次，FN=漏掉亏损批次，FP=错误拒绝盈利批次，TN=正确放行盈利批次）',
                   fontsize=10, fontweight='bold', pad=10)

    loss_n   = batches['loss'].sum()
    profit_n = (~batches['loss']).sum()
    fig.suptitle(
        f'择时策略对齐度分析   共 {len(batches)} 批次（亏损 {loss_n} / 盈利 {profit_n}）\n'
        f'完美策略 = 召回率100% + 精确率100%，实际需权衡二者',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'图表已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else 'analyse'
    os.makedirs(out_dir, exist_ok=True)

    print('加载数据...')
    nav     = load_nav()
    batches = load_batches()

    print('计算择时信号...')
    timing_maps = build_timing_maps()

    print('计算对齐指标...')
    metrics = {}
    for label, get_ratio in timing_maps.items():
        metrics[label] = compute_metrics(batches, get_ratio)
    metrics_df = pd.DataFrame(metrics).T

    print('\n===== 各择时策略对齐指标 =====')
    print(f'{"策略":<30} {"召回":>6} {"精确":>6} {"F1":>6} {"空仓率":>7}')
    print('-' * 60)
    for label, m in metrics.items():
        print(f'{label.replace(chr(10)," "):<30} '
              f'{m["recall"]:>6.1%} {m["precision"]:>6.1%} '
              f'{m["f1"]:>6.1%} {m["avoid_rate"]:>7.1%}')

    save_path = os.path.join(out_dir, 'timing_alignment.png')
    print('\n绘图...')
    plot(nav, batches, timing_maps, metrics_df, save_path)


if __name__ == '__main__':
    main()
