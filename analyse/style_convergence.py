"""
analyse/style_convergence.py

大小盘风格趋同评估。

通过滚动相关性检测小微盘（932000.CSI）与大盘（000510.CSI）日收益率
的同步程度，相关性越高说明两者走势越趋同、风格差异越小。

用法：
    python analyse/style_convergence.py [输出目录]

    若不指定输出目录，图表保存至 analyse/ 同级的 results/ 下。

数据依赖：
    data/raw/index_daily/932000.CSI.csv   中证2000（小微盘）
    data/raw/index_daily/000510.CSI.csv   中证500（大中盘）
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = ['STHeiti', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------ #
#  配置                                                                #
# ------------------------------------------------------------------ #

SMALL_FILE  = 'data/raw/index_daily/932000.CSI.csv'   # 中证2000（小微盘）
LARGE_FILE  = 'data/raw/index_daily/000510.CSI.csv'   # 中证500（大中盘）
SMALL_NAME  = '中证2000（小微盘）'
LARGE_NAME  = '中证500（大中盘）'
SMALL_COLOR = '#d73027'
LARGE_COLOR = '#4575b4'

ROLL_WINDOW = 5    # 滚动相关窗口（交易日）

# 相关性分区阈值：用于背景着色和统计
CORR_HIGH   = 0.6   # 高度趋同
CORR_LOW    = -0.2  # 明显背离


# ------------------------------------------------------------------ #
#  数据加载与预处理                                                     #
# ------------------------------------------------------------------ #

def load_index(path: str, name: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df.sort_values('trade_date').set_index('trade_date')
    ret = df['pct_chg'] / 100   # 转为小数
    ret.name = name
    return ret


def build_dataset() -> pd.DataFrame:
    small = load_index(SMALL_FILE, 'small')
    large = load_index(LARGE_FILE, 'large')
    df = pd.concat([small, large], axis=1).dropna()

    df['roll_corr'] = df['small'].rolling(ROLL_WINDOW).corr(df['large'])

    # 相关性分级（用于背景着色）
    df['regime'] = pd.cut(
        df['roll_corr'],
        bins=[-1.01, CORR_LOW, CORR_HIGH, 1.01],
        labels=['背离', '中性', '趋同'],
    )

    # 累计净值（从1出发，用于走势对比）
    df['small_nav'] = (1 + df['small']).cumprod()
    df['large_nav'] = (1 + df['large']).cumprod()

    # 小盘相对大盘的超额（spread）
    df['spread'] = df['small'] - df['large']
    df['spread_cum'] = df['spread'].cumsum()

    # 过去5天累计收益（判断是否整体处于下行通道）
    df['small_5d_cum'] = df['small'].rolling(5).sum()
    df['large_5d_cum'] = df['large'].rolling(5).sum()

    # 风险信号：0 = 回避（高相关 + 双双下行），1 = 正常
    # 含义：大小盘高度趋同且同步下跌，说明整体风险偏好收缩，策略表现往往较差
    df['signal'] = np.where(
        (df['roll_corr'] > 0.75)
        & (df['small_5d_cum'] < 0)
        & (df['large_5d_cum'] < 0),
        0, 1
    ).astype(int)

    return df


# ------------------------------------------------------------------ #
#  文本摘要                                                             #
# ------------------------------------------------------------------ #

def print_summary(df: pd.DataFrame) -> None:
    rc = df['roll_corr'].dropna()
    print('=' * 55)
    print('  大小盘风格趋同分析报告')
    print('=' * 55)
    print(f'\n数据区间：{df.index[0].date()} ~ {df.index[-1].date()}  共{len(df)}个交易日')
    print(f'\n【滚动{ROLL_WINDOW}日相关系数统计】')
    print(f'  均值:   {rc.mean():.4f}')
    print(f'  中位数: {rc.median():.4f}')
    print(f'  标准差: {rc.std():.4f}')
    print(f'  最大值: {rc.max():.4f}  ({rc.idxmax().date()})')
    print(f'  最小值: {rc.min():.4f}  ({rc.idxmin().date()})')

    total = len(rc)
    n_high = (rc >= CORR_HIGH).sum()
    n_low  = (rc <= CORR_LOW).sum()
    n_mid  = total - n_high - n_low
    print(f'\n【分区间天数占比（窗口内有值的{total}天）】')
    print(f'  趋同区（≥{CORR_HIGH}）: {n_high:>4}天  {n_high/total:.1%}')
    print(f'  中性区           : {n_mid:>4}天  {n_mid/total:.1%}')
    print(f'  背离区（≤{CORR_LOW}）: {n_low:>4}天  {n_low/total:.1%}')

    # 最近30天
    recent = rc.iloc[-30:]
    print(f'\n【最近30天】均值={recent.mean():.4f}  最新值={rc.iloc[-1]:.4f}  '
          f'（{df.index[-1].date()}）')

    sig = df['signal'].dropna()
    n_avoid = (sig == 0).sum()
    print(f'\n【风险信号统计（信号0=回避）】')
    print(f'  回避天数: {n_avoid}天  占比={n_avoid/len(sig):.1%}')
    print(f'  最新信号: {"⚠ 0（回避）" if sig.iloc[-1] == 0 else "✓ 1（正常）"}  ({df.index[-1].date()})')
    print()


# ------------------------------------------------------------------ #
#  绘图                                                                #
# ------------------------------------------------------------------ #

def plot(df: pd.DataFrame, save_path: str) -> None:
    fig = plt.figure(figsize=(16, 17))
    fig.patch.set_facecolor('#f8f8f8')
    gs = gridspec.GridSpec(5, 1, hspace=0.48, figure=fig,
                           height_ratios=[2.2, 1.3, 2.2, 1.3, 0.8])

    dates = df.index  # DatetimeIndex，直接用于横轴

    # 格式化横轴（季度刻度）
    def fmt_ax(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
        ax.set_facecolor('#f0f0f0')
        ax.grid(axis='y', color='white', linewidth=0.5)
        ax.set_xlim(dates[0], dates[-1])

    # ── 子图1: 累计净值走势 ─────────────────────────────────────── #
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df['small_nav'], color=SMALL_COLOR, linewidth=1.6,
             label=SMALL_NAME)
    ax1.plot(dates, df['large_nav'], color=LARGE_COLOR, linewidth=1.6,
             label=LARGE_NAME)
    ax1.set_ylabel('累计净值（基准=1）', fontsize=10)
    ax1.set_title('大小盘指数累计走势', fontsize=11, fontweight='bold', pad=8)
    ax1.legend(fontsize=9, framealpha=0.85, loc='upper left')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fmt_ax(ax1)

    # ── 子图2: 小盘相对大盘的日超额（spread）& 累计超额 ──────────── #
    ax2 = fig.add_subplot(gs[1])
    pos = df['spread'] >= 0
    ax2.bar(dates[pos],  df['spread'][pos] * 100,
            width=1.5, color=SMALL_COLOR, alpha=0.7, label='小盘占优')
    ax2.bar(dates[~pos], df['spread'][~pos] * 100,
            width=1.5, color=LARGE_COLOR, alpha=0.7, label='大盘占优')
    ax2.axhline(0, color='black', linewidth=0.7)
    ax2r = ax2.twinx()
    ax2r.plot(dates, df['spread_cum'] * 100, color='#e6550d',
              linewidth=1.4, label='累计超额')
    ax2r.set_ylabel('累计超额 (%)', fontsize=9, color='#e6550d')
    ax2r.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax2.set_ylabel('日超额 (%)', fontsize=10)
    ax2.set_title('小盘 − 大盘日超额收益（红正=小盘占优，蓝负=大盘占优）',
                  fontsize=11, fontweight='bold', pad=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    fmt_ax(ax2)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=SMALL_COLOR, lw=6, alpha=0.7),
               Line2D([0],[0], color=LARGE_COLOR, lw=6, alpha=0.7),
               Line2D([0],[0], color='#e6550d', lw=1.4)]
    ax2.legend(handles, ['小盘占优', '大盘占优', '累计超额'],
               fontsize=8.5, framealpha=0.85, loc='upper left')

    # ── 子图3: 滚动相关系数（主图）──────────────────────────────── #
    ax3 = fig.add_subplot(gs[2])
    rc = df['roll_corr']

    # 背景色：趋同区绿，背离区橙，中性区白
    ax3.axhspan(CORR_HIGH,  1.0,  alpha=0.10, color='#2ca02c', zorder=0)
    ax3.axhspan(CORR_LOW,   CORR_HIGH, alpha=0.05, color='#888888', zorder=0)
    ax3.axhspan(-1.0, CORR_LOW, alpha=0.10, color='#d73027', zorder=0)

    # 填充：相关系数本体
    ax3.fill_between(dates, 0, rc,
                     where=(rc >= 0), alpha=0.35, color='#2ca02c', interpolate=True)
    ax3.fill_between(dates, 0, rc,
                     where=(rc < 0),  alpha=0.35, color='#d73027', interpolate=True)
    ax3.plot(dates, rc, color='#333333', linewidth=1.4, zorder=3)

    ax3.axhline(0,          color='black',   linewidth=0.8)
    ax3.axhline(CORR_HIGH,  color='#2ca02c', linewidth=1.0, linestyle='--',
                label=f'趋同阈值 {CORR_HIGH}')
    ax3.axhline(CORR_LOW,   color='#d73027', linewidth=1.0, linestyle='--',
                label=f'背离阈值 {CORR_LOW}')

    # 标注最新值
    last_rc = rc.dropna().iloc[-1]
    last_dt = rc.dropna().index[-1]
    ax3.annotate(f'最新: {last_rc:.3f}',
                 xy=(last_dt, last_rc),
                 xytext=(-60, 15), textcoords='offset points',
                 fontsize=8.5, color='#333333',
                 arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))

    ax3.set_ylim(-1, 1)
    ax3.set_ylabel('Pearson r', fontsize=10)
    ax3.set_title(f'滚动 {ROLL_WINDOW} 日相关系数（绿=正相关/趋同，红=负相关/背离）\n'
                  '绿色背景区 = 趋同警示区（r ≥ 0.6），风格轮动效应减弱',
                  fontsize=11, fontweight='bold', pad=8)
    ax3.legend(fontsize=9, framealpha=0.85, loc='lower left')
    fmt_ax(ax3)

    # 在相关性图上叠加信号0区域（灰色竖条）
    sig = df['signal']
    in_avoid = False
    seg_start = None
    for dt, sv in sig.items():
        if sv == 0 and not in_avoid:
            in_avoid = True
            seg_start = dt
        elif sv != 0 and in_avoid:
            ax3.axvspan(seg_start, dt, alpha=0.18, color='#888888', zorder=1)
            in_avoid = False
    if in_avoid:
        ax3.axvspan(seg_start, dates[-1], alpha=0.18, color='#888888', zorder=1)
    from matplotlib.patches import Patch
    ax3.legend(
        handles=ax3.get_legend_handles_labels()[0] + [
            Patch(facecolor='#888888', alpha=0.35, label='信号0（回避区）')
        ],
        labels=ax3.get_legend_handles_labels()[1] + ['信号0（回避区）'],
        fontsize=9, framealpha=0.85, loc='lower left',
    )

    # ── 子图4: 相关系数分布（直方图）──────────────────────────────── #
    ax4 = fig.add_subplot(gs[3])
    rc_valid = rc.dropna()
    bins = np.linspace(-1, 1, 41)
    counts, edges = np.histogram(rc_valid, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_c = ['#2ca02c' if c >= 0 else '#d73027' for c in centers]
    ax4.bar(centers, counts, width=edges[1]-edges[0]*0.9,
            color=bar_c, alpha=0.75, linewidth=0)
    ax4.axvline(CORR_HIGH, color='#2ca02c', linewidth=1.2,
                linestyle='--', label=f'趋同阈值 {CORR_HIGH}')
    ax4.axvline(CORR_LOW,  color='#d73027', linewidth=1.2,
                linestyle='--', label=f'背离阈值 {CORR_LOW}')
    ax4.axvline(rc_valid.mean(), color='#e6550d', linewidth=1.5,
                label=f'均值 {rc_valid.mean():.3f}')
    ax4.axvline(rc_valid.iloc[-1], color='black', linewidth=1.5,
                linestyle=':', label=f'最新值 {rc_valid.iloc[-1]:.3f}')
    ax4.set_xlabel('相关系数', fontsize=10)
    ax4.set_ylabel('频次', fontsize=10)
    ax4.set_title('滚动相关系数历史分布（正相关=绿，负相关=红）',
                  fontsize=11, fontweight='bold', pad=8)
    ax4.legend(fontsize=8.5, framealpha=0.85, loc='upper left', ncol=2)
    ax4.set_facecolor('#f0f0f0')
    ax4.grid(axis='y', color='white', linewidth=0.5)

    # ── 子图5: 风险信号时序条 ─────────────────────────────────────── #
    ax5 = fig.add_subplot(gs[4])
    sig_vals = df['signal'].values
    # 绘制填充：信号0区域填灰，信号1区域填绿
    ax5.fill_between(dates, 0, 1,
                     where=(sig_vals == 1), step='post',
                     color='#2ca02c', alpha=0.45, label='信号1（正常）')
    ax5.fill_between(dates, 0, 1,
                     where=(sig_vals == 0), step='post',
                     color='#888888', alpha=0.55, label='信号0（回避）')
    # 在 ax1、ax2 上也叠加信号0的灰色背景，方便比对净值
    for ax_top in (ax1, ax2):
        in_avoid = False
        seg_start = None
        for dt, sv in df['signal'].items():
            if sv == 0 and not in_avoid:
                in_avoid = True
                seg_start = dt
            elif sv != 0 and in_avoid:
                ax_top.axvspan(seg_start, dt, alpha=0.12, color='#888888', zorder=0)
                in_avoid = False
        if in_avoid:
            ax_top.axvspan(seg_start, dates[-1], alpha=0.12, color='#888888', zorder=0)

    n_avoid = (sig_vals == 0).sum()
    ax5.set_ylim(0, 1)
    ax5.set_yticks([])
    ax5.set_title(
        f'风险信号（灰=0回避 / 绿=1正常）\n'
        f'触发条件：滚动{ROLL_WINDOW}日相关>0.75 且 两指数5日累计收益均为负'
        f'   共触发 {n_avoid} 天（占比 {n_avoid/max(len(sig_vals),1):.1%}）',
        fontsize=10, fontweight='bold', pad=6,
    )
    ax5.legend(fontsize=8.5, framealpha=0.85, loc='upper right', ncol=2)
    fmt_ax(ax5)

    fig.suptitle(f'{SMALL_NAME} vs {LARGE_NAME}\n大小盘风格趋同监测',
                 fontsize=14, fontweight='bold', y=1.008)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'图表已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  CSV 输出                                                            #
# ------------------------------------------------------------------ #

def save_csv(df: pd.DataFrame, save_path: str) -> None:
    out = df[['small', 'large', 'roll_corr', 'spread', 'spread_cum',
              'regime', 'small_5d_cum', 'large_5d_cum', 'signal']].copy()
    out.columns = ['small_ret', 'large_ret', f'roll_corr_{ROLL_WINDOW}d',
                   'daily_spread', 'cum_spread', 'regime',
                   'small_5d_cum_ret', 'large_5d_cum_ret', 'signal']
    out.index.name = 'trade_date'
    out.to_csv(save_path, float_format='%.6f')
    print(f'数据已保存 → {save_path}')


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else 'analyse'
    os.makedirs(out_dir, exist_ok=True)

    df = build_dataset()
    print_summary(df)

    save_csv(df, os.path.join(out_dir, 'style_convergence.csv'))
    plot(df,     os.path.join(out_dir, 'style_convergence.png'))


if __name__ == '__main__':
    main()
