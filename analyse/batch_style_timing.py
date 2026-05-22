"""
analyse/batch_style_timing.py

分析批次风格（大盘/小盘偏向）与盲窗口指数走势的交互效应。

核心问题：
  当大小盘风格分化时（corr_5d 低），模型可能已自动切换到上涨的那一侧。
  此时 BlindWindowTiming 因 corr 低而空仓，可能错过收益。
  本脚本验证：批次的市值偏向 × 对应指数的盲窗口走势，是否能预测批次收益。

用法：
    python analyse/batch_style_timing.py <task目录路径>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr

plt.rcParams['font.family'] = ['STHeiti', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

LOOKAHEAD = 6
SIGNAL_WINDOW = 7

SMALL_FILE = 'data/raw/index_daily/932000.CSI.csv'
LARGE_FILE  = 'data/raw/index_daily/000510.CSI.csv'
CALENDAR_FILE = 'data/raw/trade_cal.csv'
SECTION_DIR   = 'data/section/'

FIG_BG = '#f8f8f8'
AX_BG  = '#f0f0f0'


# ------------------------------------------------------------------ #
#  数据加载                                                            #
# ------------------------------------------------------------------ #

def load_trade_dates():
    cal = pd.read_csv(CALENDAR_FILE)
    if 'is_open' in cal.columns:
        cal = cal[cal['is_open'] == 1]
    return sorted(cal['cal_date'].astype(str).tolist())


def load_index(path):
    df = pd.read_csv(path)
    df['trade_date'] = df['trade_date'].astype(str)
    return df.sort_values('trade_date').set_index('trade_date')['close'].astype(float)


def compute_batch_returns(task_dir):
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
        next_bd = buy_dates[i + 1] if i + 1 < len(buy_dates) else '99999999'
        raw = sells[(sells['sell_date'] > bd) & (sells['sell_date'] <= next_bd)]
        if raw.empty:
            continue
        agg = (raw.assign(amt=lambda x: x['sell_price'] * x['sell_shares'])
                  .groupby('ts_code', as_index=False)
                  .agg(amt=('amt', 'sum'), sh=('sell_shares', 'sum')))
        agg['sell_price'] = agg['amt'] / agg['sh']
        merged = batch.merge(agg[['ts_code', 'sell_price']], on='ts_code', how='left').dropna(subset=['sell_price'])
        if merged.empty:
            continue
        merged['cost'] = merged['shares'] * merged['buy_price']
        total = merged['cost'].sum()
        if total <= 0:
            continue
        merged['w'] = merged['cost'] / total
        merged['ret'] = (merged['sell_price'] - merged['buy_price']) / merged['buy_price']
        rows.append({
            'buy_date':     bd,
            'batch_return': (merged['ret'] * merged['w']).sum(),
            'stocks':       merged['ts_code'].tolist(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  批次风格：信号日 T 截面中被选股票的 size_factor 均值               #
# ------------------------------------------------------------------ #

def compute_batch_style(batch_df, trade_dates):
    styles = []
    td_set = set(trade_dates)
    for _, row in batch_df.iterrows():
        bd = row['buy_date']
        if bd not in td_set:
            styles.append(np.nan); continue
        t1_idx = trade_dates.index(bd)
        if t1_idx < 1:
            styles.append(np.nan); continue
        signal_day = trade_dates[t1_idx - 1]
        path = os.path.join(SECTION_DIR, f'{signal_day}.csv')
        if not os.path.exists(path):
            styles.append(np.nan); continue
        df = pd.read_csv(path)
        if 'ts_code' not in df.columns or 'size_factor_standard' not in df.columns:
            styles.append(np.nan); continue
        sel = df[df['ts_code'].isin(row['stocks'])]
        styles.append(sel['size_factor_standard'].mean() if not sel.empty else np.nan)
    batch_df = batch_df.copy()
    batch_df['batch_style'] = styles
    return batch_df


# ------------------------------------------------------------------ #
#  盲窗口指数特征                                                       #
# ------------------------------------------------------------------ #

def compute_blind_features(batch_df, trade_dates, small_close, large_close):
    s_ret = small_close.pct_change()
    l_ret = large_close.pct_change()
    td_set = set(trade_dates)

    rows = []
    for _, row in batch_df.iterrows():
        bd = row['buy_date']
        if bd not in td_set:
            rows.append({}); continue
        t1_idx = trade_dates.index(bd)
        if t1_idx < 1 + LOOKAHEAD + SIGNAL_WINDOW:
            rows.append({}); continue

        signal_day = trade_dates[t1_idx - 1]          # T
        pre_day    = trade_dates[t1_idx - 1 - LOOKAHEAD]  # T-6

        # 累计收益：close[T] / close[T-6] - 1
        sc = small_close.get(signal_day, np.nan)
        sp = small_close.get(pre_day,    np.nan)
        lc = large_close.get(signal_day, np.nan)
        lp = large_close.get(pre_day,    np.nan)
        small_cum = sc / sp - 1 if (sc == sc and sp == sp and sp != 0) else np.nan
        large_cum = lc / lp - 1 if (lc == lc and lp == lp and lp != 0) else np.nan

        # corr_5d / l_vol：SIGNAL_WINDOW 个日收益
        win_start = t1_idx - SIGNAL_WINDOW
        win_dates = trade_dates[win_start: t1_idx + 1]
        s_arr = np.array([s_ret.get(d, np.nan) for d in win_dates[1:]], dtype=float)
        l_arr = np.array([l_ret.get(d, np.nan) for d in win_dates[1:]], dtype=float)
        valid = ~(np.isnan(s_arr) | np.isnan(l_arr))
        if valid.sum() >= 3:
            corr_5d = float(np.corrcoef(s_arr[valid], l_arr[valid])[0, 1])
            l_vol   = float(np.std(l_arr[valid], ddof=1))
        else:
            corr_5d = l_vol = np.nan

        rows.append({'small_cum': small_cum, 'large_cum': large_cum,
                     'corr_5d': corr_5d, 'l_vol': l_vol})

    feat_df = pd.DataFrame(rows, index=batch_df.index)
    out = pd.concat([batch_df, feat_df], axis=1)

    # style_aligned_cum：按批次风格选对应指数
    def _aligned(r):
        if any(pd.isna(r.get(k)) for k in ['batch_style', 'small_cum', 'large_cum']):
            return np.nan
        return r['small_cum'] if r['batch_style'] > 0 else r['large_cum']

    out['style_aligned_cum'] = out.apply(_aligned, axis=1)
    return out


# ------------------------------------------------------------------ #
#  文本分析                                                            #
# ------------------------------------------------------------------ #

def print_analysis(df):
    features = ['small_cum', 'large_cum', 'corr_5d', 'l_vol',
                'batch_style', 'style_aligned_cum']
    valid = df.dropna(subset=['batch_return'])

    print('=' * 62)
    print('  盲窗口特征 × 批次收益相关性')
    print('=' * 62)
    print(f'{"特征":<25} {"Spearman r":>12} {"p值":>10}  sig')
    print('-' * 62)
    for feat in features:
        sub = valid.dropna(subset=[feat])
        if len(sub) < 10:
            continue
        r_s, p_s = spearmanr(sub[feat], sub['batch_return'])
        sig = '***' if p_s < 0.01 else ('**' if p_s < 0.05 else ('*' if p_s < 0.1 else ''))
        print(f'{feat:<25} {r_s:>+12.4f} {p_s:>10.4f}  {sig}')

    print()
    print('── 条件分析：corr_5d 低（风格分化）vs 高（趋同）时 style_aligned_cum 的预测力 ──')
    df2 = valid.dropna(subset=['style_aligned_cum', 'corr_5d'])
    med = df2['corr_5d'].median()
    for mask, label in [(df2['corr_5d'] < med, f'低corr(<{med:.2f}) n={( df2["corr_5d"]<med).sum()}'),
                        (df2['corr_5d'] >= med, f'高corr(≥{med:.2f}) n={(df2["corr_5d"]>=med).sum()}')]:
        sub = df2[mask]
        if len(sub) < 5: continue
        r_s, p_s = spearmanr(sub['style_aligned_cum'], sub['batch_return'])
        print(f'  {label}: r={r_s:+.4f}  p={p_s:.4f}')

    print()
    print('── 四象限：corr高/低 × 对齐指数涨/跌 ──')
    df3 = df2.copy()
    for corr_label, cmask in [('低corr', df3['corr_5d'] < med), ('高corr', df3['corr_5d'] >= med)]:
        for cum_label, dmask in [('对齐涨', df3['style_aligned_cum'] >= 0),
                                  ('对齐跌', df3['style_aligned_cum'] < 0)]:
            sub = df3[cmask & dmask]
            if len(sub) < 3: continue
            wr = (sub['batch_return'] >= 0).mean()
            mr = sub['batch_return'].mean()
            print(f'  {corr_label} × {cum_label}: n={len(sub):3d}  胜率={wr:.1%}  均值={mr:+.4f}')
    print()


# ------------------------------------------------------------------ #
#  可视化                                                              #
# ------------------------------------------------------------------ #

def plot_analysis(df, save_path):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(FIG_BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32, figure=fig)

    # 左上：批次风格随时间变化
    ax1 = fig.add_subplot(gs[0, 0])
    d1 = df.dropna(subset=['batch_style', 'batch_return']).sort_values('buy_date')
    colors = ['#2ca02c' if r >= 0 else '#d62728' for r in d1['batch_return']]
    ax1.bar(range(len(d1)), d1['batch_style'], color=colors, alpha=0.75, linewidth=0)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xlabel('批次（时间顺序）', fontsize=10)
    ax1.set_ylabel('size_factor均值\n正=小盘偏向，负=大盘偏向', fontsize=9)
    ax1.set_title('批次市值风格随时间变化\n绿=上涨批次，红=下跌批次', fontsize=10, fontweight='bold')
    ax1.set_facecolor(AX_BG)
    ax1.grid(axis='y', color='white', linewidth=0.5)

    # 右上：风格分化时，对齐指数走势 vs 批次收益
    ax2 = fig.add_subplot(gs[0, 1])
    d2 = df.dropna(subset=['style_aligned_cum', 'batch_return', 'corr_5d'])
    med = d2['corr_5d'].median()
    low  = d2[d2['corr_5d'] <  med]
    high = d2[d2['corr_5d'] >= med]
    ax2.scatter(low['style_aligned_cum']  * 100, low['batch_return']  * 100,
                alpha=0.6, s=30, color='#ff7f0e', label=f'低corr(分化) n={len(low)}')
    ax2.scatter(high['style_aligned_cum'] * 100, high['batch_return'] * 100,
                alpha=0.6, s=30, color='#1f77b4', label=f'高corr(趋同) n={len(high)}')
    ax2.axhline(0, color='black', linewidth=0.6)
    ax2.axvline(0, color='black', linewidth=0.6)
    r_low,  p_low  = spearmanr(low['style_aligned_cum'],  low['batch_return'])
    r_high, p_high = spearmanr(high['style_aligned_cum'], high['batch_return'])
    ax2.text(0.03, 0.97,
             f'低corr: r={r_low:+.3f} p={p_low:.3f}\n高corr: r={r_high:+.3f} p={p_high:.3f}',
             transform=ax2.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax2.set_xlabel('对齐指数盲窗口累计收益 (%)', fontsize=10)
    ax2.set_ylabel('批次收益率 (%)', fontsize=10)
    ax2.set_title('对齐指数走势 vs 批次收益\n按大小盘相关性分组', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, framealpha=0.85)
    ax2.set_facecolor(AX_BG)
    ax2.grid(color='white', linewidth=0.5)

    # 左下：四象限胜率热图
    ax3 = fig.add_subplot(gs[1, 0])
    d3 = df.dropna(subset=['style_aligned_cum', 'batch_return', 'corr_5d']).copy()
    cq = [d3['corr_5d'].quantile(0.33), d3['corr_5d'].quantile(0.67)]
    dq = [d3['style_aligned_cum'].quantile(0.33), d3['style_aligned_cum'].quantile(0.67)]

    def _bin3(x, q): return 0 if x < q[0] else (1 if x < q[1] else 2)
    d3['cb'] = d3['corr_5d'].apply(lambda x: _bin3(x, cq))
    d3['db'] = d3['style_aligned_cum'].apply(lambda x: _bin3(x, dq))

    win_mat  = np.full((3, 3), np.nan)
    mean_mat = np.full((3, 3), np.nan)
    cnt_mat  = np.zeros((3, 3), dtype=int)
    for ci in range(3):
        for di in range(3):
            sub = d3[(d3['cb'] == ci) & (d3['db'] == di)]
            if len(sub) >= 3:
                win_mat[ci, di]  = (sub['batch_return'] >= 0).mean()
                mean_mat[ci, di] = sub['batch_return'].mean()
                cnt_mat[ci, di]  = len(sub)

    im = ax3.imshow(win_mat, cmap='RdYlGn', vmin=0.3, vmax=0.7, aspect='auto')
    ax3.set_xticks(range(3))
    ax3.set_yticks(range(3))
    ax3.set_xticklabels(['对齐跌', '对齐平', '对齐涨'], fontsize=9)
    ax3.set_yticklabels(['低corr', '中corr', '高corr'], fontsize=9)
    for ci in range(3):
        for di in range(3):
            if not np.isnan(win_mat[ci, di]):
                ax3.text(di, ci,
                         f'{win_mat[ci,di]:.0%}\n{mean_mat[ci,di]:+.3f}\nn={cnt_mat[ci,di]}',
                         ha='center', va='center', fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='胜率')
    ax3.set_title('胜率热图：corr × 对齐指数走势\n格内：胜率 / 均值收益 / 样本数',
                  fontsize=10, fontweight='bold')

    # 右下：批次风格 vs 大小盘收益差（验证模型是否跟上了风格）
    ax4 = fig.add_subplot(gs[1, 1])
    d4 = df.dropna(subset=['batch_style', 'small_cum', 'large_cum', 'batch_return']).copy()
    d4['style_spread'] = d4['small_cum'] - d4['large_cum']
    sc = ax4.scatter(d4['style_spread'] * 100, d4['batch_style'],
                     c=d4['batch_return'] * 100, cmap='RdYlGn',
                     vmin=-3, vmax=3, alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax4, label='批次收益率 (%)')
    ax4.axhline(0, color='black', linewidth=0.6)
    ax4.axvline(0, color='black', linewidth=0.6)
    r_s, p_s = spearmanr(d4['style_spread'], d4['batch_style'])
    ax4.text(0.03, 0.97, f'Spearman r={r_s:+.3f} p={p_s:.3f}',
             transform=ax4.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax4.set_xlabel('小盘-大盘盲窗口收益差 (%)\n正=小盘跑赢', fontsize=9)
    ax4.set_ylabel('批次风格 (size_factor均值)\n正=小盘偏向', fontsize=9)
    ax4.set_title('模型风格 vs 市场风格分化\n颜色=批次收益，验证模型是否跟上风格切换',
                  fontsize=10, fontweight='bold')
    ax4.set_facecolor(AX_BG)
    ax4.grid(color='white', linewidth=0.5)

    fig.suptitle('批次风格 × 盲窗口指数走势深度分析', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    print(f'图已保存 → {save_path}')
    plt.show()


# ------------------------------------------------------------------ #
#  入口                                                                #
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print('用法: python analyse/batch_style_timing.py <task目录路径>')
        sys.exit(1)

    task_dir = sys.argv[1]
    print(f'读取：{task_dir}')

    trade_dates = load_trade_dates()
    small_close = load_index(SMALL_FILE)
    large_close = load_index(LARGE_FILE)

    print('计算批次收益...')
    batch_df = compute_batch_returns(task_dir)
    print(f'共 {len(batch_df)} 批次')

    print('计算批次风格（信号日截面 size_factor）...')
    batch_df = compute_batch_style(batch_df, trade_dates)

    print('计算盲窗口指数特征...')
    batch_df = compute_blind_features(batch_df, trade_dates, small_close, large_close)

    print_analysis(batch_df)

    out_dir = os.path.abspath(task_dir)
    batch_df.drop(columns=['stocks'], errors='ignore').to_csv(
        os.path.join(out_dir, 'batch_style_features.csv'), index=False, float_format='%.6f'
    )
    print(f'数据已保存 → {os.path.join(out_dir, "batch_style_features.csv")}')
    plot_analysis(batch_df, os.path.join(out_dir, 'batch_style_timing.png'))


if __name__ == '__main__':
    main()
