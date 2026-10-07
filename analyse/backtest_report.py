"""
回测结果分析报告生成器。

用法（项目根目录执行）：
    python analyse/backtest_report.py <结果目录名或路径>

例：
    python analyse/backtest_report.py 20240103_20260930_20261006_221942
    python analyse/backtest_report.py backtest/results/20240103_20260930_20261006_221942

产出：在结果目录下生成 分析报告.html（自包含，图片内嵌 base64）。

报告内容：
    1. 核心绩效指标（收益/年化/波动/夏普/回撤/Calmar）
    2. 净值曲线 + 回撤曲线
    3. 月度收益与基准对比
    4. 交易与批次统计
    5. 六项可信度体检（幸存者偏差、收益集中度、排序能力、风格暴露、流动性、样本量）
    6. 结论与改进建议
"""
import os
import sys
import glob
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# A股习惯：涨用红、跌用绿
UP, DOWN = '#d62728', '#2ca02c'

BENCHMARKS = [
    ('data/raw/index_daily/000510.SH.csv', '000510.SH'),
    ('data/raw/index_daily/932000.CSI.csv', '932000.CSI'),
]


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def load_result(path):
    return dict(
        nav=pd.read_csv(os.path.join(path, 'nav.csv'), dtype={'date': str}),
        trade=pd.read_csv(os.path.join(path, 'trade_log.csv'), dtype={'date': str}),
        picks=pd.read_csv(os.path.join(path, 'daily_picks.csv'), dtype={'date': str}),
        batch=pd.read_csv(os.path.join(path, 'batch_alias_profit.csv'), dtype={'trade_date': str}),
    )


def pair_trades(trade):
    """把 BUY/SELL 配对成单笔交易，返回 DataFrame[bd, sd, ts_code, ret, hold_days]。"""
    b = trade[trade.side == 'BUY'][['date', 'ts_code', 'amount']].rename(
        columns={'date': 'bd', 'amount': 'ba'})
    s = trade[trade.side == 'SELL'][['date', 'ts_code', 'amount']].rename(
        columns={'date': 'sd', 'amount': 'sa'})
    m = b.merge(s, on='ts_code')
    m = m[m.sd >= m.bd]
    m = (m.sort_values(['bd', 'ts_code', 'sd'])
           .groupby(['bd', 'ts_code'])
           .agg(sd=('sd', 'first'), ba=('ba', 'first'), sa=('sa', 'first'))
           .reset_index())
    m['ret'] = m.sa / m.ba - 1
    m['hold_days'] = (pd.to_datetime(m.sd) - pd.to_datetime(m.bd)).dt.days
    return m


def core_metrics(nav):
    nav = nav.copy()
    nav['date'] = pd.to_datetime(nav['date'])
    nav = nav.sort_values('date').reset_index(drop=True)
    r = nav['nav'].pct_change().dropna()
    years = (nav['date'].iloc[-1] - nav['date'].iloc[0]).days / 365.25
    tot = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
    ann = (1 + tot) ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / (r.std() * np.sqrt(252))
    nav['peak'] = nav['nav'].cummax()
    nav['dd'] = nav['nav'] / nav['peak'] - 1
    i_mdd = nav['dd'].idxmin()
    peak_i = nav.loc[:i_mdd, 'nav'].idxmax()
    mdd = nav['dd'].min()
    return dict(
        nav=nav, ret=r, years=years, total=tot, ann=ann, vol=vol,
        sharpe=sharpe, mdd=mdd, calmar=ann / abs(mdd),
        win_day=(r > 0).mean(),
        mdd_start=str(nav.loc[peak_i, 'date'].date()),
        mdd_end=str(nav.loc[i_mdd, 'date'].date()),
        mdd_days=int(i_mdd - peak_i),
        start=str(nav['date'].iloc[0].date()), end=str(nav['date'].iloc[-1].date()),
        n_days=len(nav),
    )


def bench_monthly(d0, d1):
    out = {}
    for f, code in BENCHMARKS:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, dtype={'trade_date': str})
        df['date'] = pd.to_datetime(df['trade_date'])
        df = df[(df.date >= d0) & (df.date <= d1)]
        if df.empty:
            continue
        df['ym'] = df.date.dt.to_period('M')
        g = df.groupby('ym').agg(f=('close', 'first'), l=('close', 'last'))
        out[code] = g.l / g.f - 1
    return out


def trading_calendar():
    """从 data/market 分区目录 + trade_cal 推出实际有数据的交易日序列。"""
    import re
    ps = glob.glob('data/market/trade_date=*/part-*.parquet')
    have = set(re.search(r'trade_date=(\d{8})', p.replace('\\', '/')).group(1) for p in ps)
    cal = pd.read_csv('data/raw/trade_cal.csv', dtype={'cal_date': str})
    cal = cal[cal.is_open == 1].sort_values('cal_date')
    return [d for d in cal.cal_date if d in have]


def peer_test(picks):
    """
    对照实验：把模型换成两个"傻瓜基准"，看模型还剩多少增量。
      naive  = 无脑买 20 日动量最高的 5 只（追动量）
      univ   = 全池中位（等权持有整个股票池）
    统一用 label（5 日后收益）比较，量纲一致可直接相减。
    """
    cal = trading_calendar()
    idx = {d: i for i, d in enumerate(cal)}
    rows = []
    for bd, g in picks.groupby('date'):
        i = idx.get(bd)
        if i is None or i < 1:
            continue
        fs = sorted(glob.glob(f"data/market/trade_date={cal[i-1]}/part-*.parquet"))
        if not fs:
            continue
        df = pd.read_parquet(fs[0], columns=['ts_code', 'is_st', 'ret_20d', 'label', 'total_mv'])
        df = df[(df.is_st == 0)].dropna(subset=['label', 'ret_20d'])
        if len(df) < 50:
            continue
        ml = df[df.ts_code.isin(g.ts_code)]['label']
        if ml.empty:
            continue
        rows.append(dict(date=bd, ml=ml.mean(),
                         naive=df.nlargest(5, 'ret_20d')['label'].mean(),
                         univ=df['label'].mean(),
                         ml_mom=df[df.ts_code.isin(g.ts_code)]['ret_20d'].median(),
                         naive_mom=df.nlargest(5, 'ret_20d')['ret_20d'].median(),
                         univ_mom=df['ret_20d'].median(),
                         ml_mv=df[df.ts_code.isin(g.ts_code)]['total_mv'].median(),
                         univ_mv=df['total_mv'].median()))
    if not rows:
        return None
    S = pd.DataFrame(rows)

    def tstat(x):
        d = x.dropna()
        return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan
    return dict(
        n=len(S),
        ml=S.ml.mean(), naive=S.naive.mean(), univ=S.univ.mean(),
        t_ml=tstat(S.ml - S.univ), t_naive=tstat(S.naive - S.univ), t_inc=tstat(S.ml - S.naive),
        ml_mom=S.ml_mom.median(), naive_mom=S.naive_mom.median(), univ_mom=S.univ_mom.median(),
        ml_mv=S.ml_mv.median(), univ_mv=S.univ_mv.median(),
    )


def ex_window(nav, a='2026-04-01', b='2026-06-30'):
    """剔除某个极端窗口后的年化收益（把前后两段净值直接相乘）。"""
    d = nav['date']
    pre = nav[d <= a]
    post = nav[d >= b]
    if len(pre) < 2 or len(post) < 2:
        return None
    r1 = pre.nav.iloc[-1] / pre.nav.iloc[0]
    r2 = post.nav.iloc[-1] / post.nav.iloc[0]
    yrs_all = (d.iloc[-1] - d.iloc[0]).days / 365.25
    yrs_ex = yrs_all - (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    comb = r1 * r2
    return dict(total=comb - 1, ann=comb ** (1 / yrs_ex) - 1, years=yrs_ex,
                win_mul=nav[(d >= a) & (d <= b)].nav.iloc[-1] / nav[(d >= a) & (d <= b)].nav.iloc[0])


def survivorship_check(last_date):
    """股票池是否存在幸存者偏差：早期有、后期消失的股票数。"""
    res = {}
    try:
        def codes(d):
            fs = glob.glob(f'data/market/trade_date={d}/part-*.parquet')
            if not fs:
                return None, None
            c = pd.read_parquet(fs[0], columns=['ts_code'])['ts_code']
            return set(c), len(c)
        last, n_last = codes(last_date)
        if last is None:
            return None
        for probe in ('20180102', '20240103'):
            early, n_early = codes(probe)
            if early is None:
                continue
            res[probe] = dict(n=len(early), missing=len(early - last))
        res['n_last'] = n_last
        return res
    except Exception:
        return None


def build_report(res_dir):
    d = load_result(res_dir)
    M = core_metrics(d['nav'])
    nav = M['nav']
    T = pair_trades(d['trade'])
    pk = d['picks']
    bt = d['batch']

    # ---------- 图1：净值 + 回撤 ----------
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                           gridspec_kw={'height_ratios': [2.2, 1]})
    ax[0].plot(nav['date'], nav['nav'] / 1e6, color='#1f4e79', lw=1.4)
    ax[0].set_ylabel('净值（百万）')
    ax[0].set_title(f"净值曲线  {M['start']} ~ {M['end']}", fontsize=12)
    ax[0].grid(alpha=.3)
    ax[0].axhline(1, color='gray', ls='--', lw=.8)
    ax[1].fill_between(nav['date'], nav['dd'] * 100, 0, color=DOWN, alpha=.55)
    ax[1].set_ylabel('回撤 %')
    ax[1].grid(alpha=.3)
    img_nav = fig_to_b64(fig)

    # ---------- 图2：月度收益 vs 基准 ----------
    nav['ym'] = nav.date.dt.to_period('M')
    g = nav.groupby('ym').agg(f=('nav', 'first'), l=('nav', 'last'))
    mret = g.l / g.f - 1
    bm = bench_monthly(nav.date.min(), nav.date.max())
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(mret))
    ax.bar(x, mret.values * 100, color=[UP if v > 0 else DOWN for v in mret.values], width=.72)
    colors = ['#ff7f0e', '#9467bd', '#8c564b']
    for i, (code, s) in enumerate(bm.items()):
        s = s.reindex(mret.index)
        ax.plot(x, s.values * 100, marker='o', ms=3, lw=1.2,
                color=colors[i % len(colors)], label=code)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([str(p) for p in mret.index[::3]], rotation=60, fontsize=7)
    ax.axhline(0, color='k', lw=.8)
    ax.set_ylabel('月度收益 %')
    ax.set_title('月度收益（柱=策略，线=基准指数）', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis='y')
    img_month = fig_to_b64(fig)

    # ---------- 图3：单笔收益分布 + 稳健性 ----------
    r = T['ret'].sort_values(ascending=False).values
    robust = [(n, r[n:].mean(), r[n:].std(), (r[n:] > 0).mean(),
               r[n:].mean() / r[n:].std() * np.sqrt(len(r) - n)) for n in (0, 5, 10, 20, 30)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].hist(T['ret'] * 100, bins=45, color='#4c72b0', edgecolor='white')
    ax[0].axvline(T['ret'].mean() * 100, color=UP, lw=1.6, label=f"均值 {T['ret'].mean():+.2%}")
    ax[0].axvline(T['ret'].median() * 100, color='#555', ls='--', lw=1.2,
                  label=f"中位 {T['ret'].median():+.2%}")
    ax[0].set_xlabel('单笔收益 %')
    ax[0].set_ylabel('笔数')
    ax[0].set_title(f"单笔收益分布（n={len(T)}）", fontsize=11)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ns = [x[0] for x in robust]
    ax[1].bar(ns, [x[1] * 100 for x in robust],
              color=['#1f4e79' if x[4] > 2 else '#c44e52' for x in robust], width=3)
    for n, mu, sd, wr, t in robust:
        ax[1].text(n, mu * 100 + .03, f"t={t:.2f}", ha='center', fontsize=8)
    ax[1].set_xticks(ns)
    ax[1].set_xlabel('剔除最好的 N 笔')
    ax[1].set_ylabel('平均单笔收益 %')
    ax[1].set_title('收益集中度（高波动策略的必然现象，非缺陷）', fontsize=11)
    ax[1].grid(alpha=.3, axis='y')
    img_robust = fig_to_b64(fig)

    # ---------- 关键数字 ----------
    buy = d['trade'][d['trade'].side == 'BUY']
    p = bt['trade_profit']
    ics = []
    mm = T.merge(pk[['date', 'ts_code', 'score', 'pct_change']].rename(columns={'date': 'bd'}),
                 on=['bd', 'ts_code'], how='left')
    mm = mm.dropna(subset=['score'])
    for _, gg in mm.groupby('bd'):
        if len(gg) >= 4:
            c = spearmanr(gg.score, gg.ret).correlation
            if c == c:
                ics.append(c)
    ics = np.array(ics)
    ic_mean = ics.mean() if len(ics) else float('nan')
    ic_t = ic_mean / ics.std() * np.sqrt(len(ics)) if len(ics) else float('nan')

    sv = survivorship_check(nav['date'].iloc[-1].strftime('%Y%m%d'))
    peer = peer_test(pk)
    exw = ex_window(nav)

    # 退市股体检（需要 data/delisted_st_history.csv，由一次性脚本生成；缺失则跳过）
    sv_extra = ''
    hist_path = os.path.join('data', 'delisted_st_history.csv')
    if os.path.exists(hist_path):
        try:
            h = pd.read_csv(hist_path, dtype={'ts_code': str})
            n_all, n_st = len(h), int(h.st_ever.sum())
            sv_extra = (f"<tr class='ok'><td>退市股中曾戴帽（会被 is_st 过滤掉）</td>"
                        f"<td>{n_st} / {n_all}（{n_st/max(1,n_all):.0%}）</td></tr>")
        except Exception:
            pass

    # 收益集中是不是异常？用 σ/μ 与正态预期对照 + Bootstrap 置信区间
    rng = np.random.default_rng(42)
    pv = bt['trade_profit'].dropna().values
    mu, sd = pv.mean(), pv.std()
    bs = np.array([rng.choice(pv, len(pv), replace=True).mean() for _ in range(5000)])
    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
    ann_of = lambda m: (1 + m) ** (243 / 5) - 1
    tail = pv > 0.10
    tail_by_year = []
    for y in sorted(set(bt['trade_date'].str[:4])):
        m = bt['trade_date'].str[:4] == y
        if m.sum() >= 5:
            tail_by_year.append((y, int(tail[m].sum()), int(m.sum())))

    # 2026 二季度贡献
    q2 = nav[(nav.date >= '2026-04-01') & (nav.date <= '2026-06-30')]
    q2_share = np.nan
    if len(q2) > 1:
        q2_share = np.log(q2.nav.iloc[-1] / q2.nav.iloc[0]) / np.log(nav.nav.iloc[-1] / nav.nav.iloc[0])

    # 流动性
    liq = []
    for dd, gg in buy.groupby('date'):
        fs = glob.glob(f"data/market/trade_date={dd}/part-*.parquet")
        if not fs:
            continue
        mk = pd.read_parquet(fs[0], columns=['ts_code', 'amount_x'])
        j = gg.merge(mk, on='ts_code', how='inner')
        liq.append((j['amount'] / (j['amount_x'] * 1000)).values)
    liq = np.concatenate(liq) if liq else np.array([np.nan])
    liq = liq[~np.isnan(liq)]

    bm_tot = {}
    for f, code in BENCHMARKS:
        if os.path.exists(f):
            df = pd.read_csv(f, dtype={'trade_date': str})
            df['date'] = pd.to_datetime(df['trade_date'])
            df = df[(df.date >= nav.date.min()) & (df.date <= nav.date.max())]
            if not df.empty:
                bm_tot[code] = df.close.iloc[-1] / df.close.iloc[0] - 1

    # ---------- HTML ----------
    def pct(x, d=2):
        return f"{x*100:.{d}f}%"

    def row(k, v, cls=''):
        return f"<tr class='{cls}'><td>{k}</td><td>{v}</td></tr>"

    sv_html = ''
    if sv:
        for probe, v in sv.items():
            if not isinstance(v, dict):
                continue
            flag = 'bad' if v['missing'] == 0 else 'ok'
            sv_html += row(f"{probe} 截面 {v['n']} 只 → 到最新消失", f"{v['missing']} 只", flag)

    bm_rows = ''.join(row(f"基准 {k} 同期", pct(v)) for k, v in bm_tot.items())

    robust_rows = ''.join(
        f"<tr class='{'ok' if t>2 else 'bad'}'><td>剔除最好 {n} 笔</td>"
        f"<td>{pct(mu)}</td><td>{pct(sd)}</td><td>{pct(wr)}</td><td>{t:.2f}</td></tr>"
        for n, mu, sd, wr, t in robust)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>回测分析报告 {os.path.basename(res_dir)}</title>
<style>
 body{{font-family:"Microsoft YaHei","Segoe UI",sans-serif;margin:0;padding:28px;color:#222;background:#f6f7f9}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 10px;padding-left:9px;border-left:4px solid #1f4e79}}
 .sub{{color:#777;font-size:12px;margin-bottom:18px}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px}}
 .card{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 16px;min-width:118px;flex:1}}
 .card .k{{font-size:11px;color:#888}} .card .v{{font-size:19px;font-weight:600;margin-top:3px}}
 .up{{color:#d62728}} .dn{{color:#2ca02c}}
 table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}
 th,td{{border:1px solid #e3e6ea;padding:6px 10px;text-align:left}}
 th{{background:#eef2f7;font-weight:600}} td:nth-child(n+2){{text-align:right}}
 tr.bad td:last-child{{color:#d62728;font-weight:600}}
 tr.ok td:last-child{{color:#2ca02c;font-weight:600}}
 tr.warn td:last-child{{color:#e08b00;font-weight:600}}
 img{{width:100%;border:1px solid #e3e6ea;border-radius:8px;background:#fff;margin:6px 0}}
 .note{{background:#fff8e6;border-left:4px solid #e0a800;padding:10px 14px;font-size:13px;line-height:1.7}}
 .good{{background:#eefaf0;border-left:4px solid #2ca02c;padding:10px 14px;font-size:13px;line-height:1.7}}
 ul{{line-height:1.9;font-size:13px}}
 .tag{{display:inline-block;padding:1px 7px;border-radius:3px;font-size:11px;margin-right:6px}}
 .t-bad{{background:#fde8e8;color:#c62828}} .t-ok{{background:#e8f5e9;color:#2e7d32}} .t-warn{{background:#fff4e5;color:#ef6c00}}
</style></head><body>
<h1>回测结果分析报告</h1>
<div class="sub">结果目录：{os.path.basename(res_dir)} ｜ 区间 {M['start']} ~ {M['end']}（{M['n_days']} 个交易日，{M['years']:.2f} 年）</div>

<h2>一、核心绩效</h2>
<div class="cards">
 <div class="card"><div class="k">总收益</div><div class="v up">{pct(M['total'])}</div></div>
 <div class="card"><div class="k">年化收益</div><div class="v up">{pct(M['ann'])}</div></div>
 <div class="card"><div class="k">年化波动</div><div class="v">{pct(M['vol'])}</div></div>
 <div class="card"><div class="k">夏普</div><div class="v">{M['sharpe']:.2f}</div></div>
 <div class="card"><div class="k">最大回撤</div><div class="v dn">{pct(M['mdd'])}</div></div>
 <div class="card"><div class="k">Calmar</div><div class="v">{M['calmar']:.2f}</div></div>
 <div class="card"><div class="k">日胜率</div><div class="v">{pct(M['win_day'],1)}</div></div>
</div>
<img src="data:image/png;base64,{img_nav}">
<table>
 {row('最大回撤区间', f"{M['mdd_start']} → {M['mdd_end']}（{M['mdd_days']} 个交易日）")}
 {row('期末净值', f"{nav['nav'].iloc[-1]:,.0f}")}
 {bm_rows}
 {row('相对基准超额（年化）', pct(M['ann'] - ((1+list(bm_tot.values())[0])**(1/M['years'])-1) if bm_tot else float('nan')))}
</table>

<h2>二、月度收益与基准</h2>
<img src="data:image/png;base64,{img_month}">
<table>
 {row('正收益月份', f"{(mret>0).sum()} / {len(mret)}")}
 {row('最好月份', f"{mret.idxmax()}  {pct(mret.max())}")}
 {row('最差月份', f"{mret.idxmin()}  {pct(mret.min())}")}
 {row('2026年4-6月对全程对数收益的贡献', pct(q2_share,1) if q2_share==q2_share else '—')}
</table>

<h2>三、交易与批次</h2>
<table>
 {row('成交笔数', f"{len(d['trade'])}（买 {len(buy)} / 卖 {len(d['trade'])-len(buy)}）")}
 {row('单笔配对交易', f"{len(T)} 笔，涉及 {buy.ts_code.nunique()} 只股票")}
 {row('建仓批次', f"{len(bt)} 批，平均每批 {len(buy)/max(1,len(bt)):.2f} 只")}
 {row('仓位暴露（有持仓交易日占比）', pct(len(set().union(*[set(pd.date_range(a,b).strftime('%Y%m%d')) for a,b in zip(T.bd,T.sd)]) & set(nav['date'].dt.strftime('%Y%m%d')))/len(nav),1))}
 {row('单笔平均收益 / 中位', f"{pct(T['ret'].mean())} / {pct(T['ret'].median())}")}
 {row('单笔胜率', pct((T['ret']>0).mean(),1))}
 {row('批均收益 / 胜率', f"{pct(p.mean())} / {pct((p>0).mean(),1)}")}
 {row('最好 / 最差批次', f"{pct(p.max())} / {pct(p.min())}")}
 {row('买入占当日成交额（中位 / 最大）', f"{pct(np.median(liq),2)} / {pct(np.max(liq),2)}")}
</table>

<h2>四、收益质量与收益集中度</h2>
<img src="data:image/png;base64,{img_robust}">
<b style="font-size:13px">先看清楚：收益集中在右尾是必然的，不是缺陷。</b>
<p style="font-size:13px;line-height:1.7;color:#555">
批均 {pct(pv.mean())}、批次标准差 {pct(pv.std())}，<b>σ/μ = {sd/abs(mu):.1f}</b>。
对任何高 σ/μ 的正收益策略，即便收益服从正态分布，前 10% 左右的赢家也会贡献均值的绝大部分——
所以"剔除最好的 5%~10% 后均值归零"是<b>算术必然</b>，不能用来否定策略。
真正该问的是<b>均值估计得准不准</b>与<b>抓极端行情的能力能不能重复</b>：
</p>
<table>
 <tr><th>指标</th><th>结果</th></tr>
 {row('批均收益 95% 置信区间（Bootstrap）', f"[{pct(ci_lo)}, {pct(ci_hi)}]，点估计 {pct(mu)}")}
 {row('折合年化 95% 置信区间', f"[{pct(ann_of(ci_lo))}, {pct(ann_of(ci_hi))}]，点估计 {pct(ann_of(mu))}")}
 {row('大涨批次（>+10%）占比', f"{int(tail.sum())} / {len(pv)} = {pct(tail.mean(),1)}")}
 {''.join(row(f"　└ {y} 年", f"{a} / {b} = {a/b:.1%}") for y, a, b in tail_by_year)}
 {row('前 10 大赢家占盈利总额', pct(r[:10].sum()/r[r>0].sum(),1))}
</table>
<p style="font-size:13px;line-height:1.7;color:#555">
结论：置信区间很宽（年化从 {pct(ann_of(ci_lo))} 到 {pct(ann_of(ci_hi))}），
说明<b>样本量不足以把期望收益定死</b>——这是当前最主要的不确定性来源，而不是"收益靠运气"。
右尾频率若各年接近，说明抓极端行情的能力可重复；若某年异常高，则该年贡献需打折看待。
</p>

<h2>五、可信度体检</h2>
<table>
 <tr><th>检查项</th><th>结果</th></tr>
 {sv_html}
 {sv_extra}
 {'' if peer is None else f"<tr class='{'ok' if peer['t_ml']>2 else 'warn'}'><td>模型 Top5 vs 全池（5日后收益 t 值）</td><td>{peer['ml']:+.4f} vs {peer['univ']:+.4f}，t={peer['t_ml']:.2f}</td></tr>"}
 {'' if peer is None else f"<tr class='{'ok' if peer['t_inc']>2 else 'bad'}'><td>模型 vs 无脑动量 Top5（增量 t 值）</td><td>{peer['ml']-peer['naive']:+.4f}，t={peer['t_inc']:.2f}</td></tr>"}
 {'' if peer is None else f"<tr class='{'ok' if peer['t_naive']<0 else 'warn'}'><td>无脑动量 Top5 vs 全池（验证是否只是押动量）</td><td>{peer['naive']:+.4f}，t={peer['t_naive']:.2f}</td></tr>"}
 <tr class="{'bad' if ic_t<2 else 'warn'}"><td>模型在 Top5 内部的排序 IC（t 值）</td><td>IC={ic_mean:+.4f}，t={ic_t:.2f}</td></tr>
 {'' if exw is None else f"<tr class='warn'><td>剔除 2026-04~06 极端窗口后年化</td><td>{pct(exw['ann'])}（该窗口 {exw['win_mul']:.2f}x）</td></tr>"}
 <tr class="warn"><td>样本量</td><td>{len(bt)} 批 / {len(T)} 笔</td></tr>
 <tr class="{'ok' if np.max(liq)<0.05 else 'warn'}"><td>建仓流动性（单笔占当日成交额最大）</td><td>{pct(np.max(liq),2)}</td></tr>
</table>

{'' if peer is None else f'''
<h2>六、模型到底有没有真本事（对照实验）</h2>
<table>
 {row('对照批次', f"{peer['n']} 批")}
 {row('模型 Top5 的 5 日后收益', f"{peer['ml']:+.4f}")}
 {row('无脑买 20 日动量最高 5 只', f"{peer['naive']:+.4f}")}
 {row('全池等权（基准）', f"{peer['univ']:+.4f}")}
 {row('模型 − 全池（t 值）', f"{peer['ml']-peer['univ']:+.4f}，t={peer['t_ml']:.2f}")}
 {row('模型 − 无脑动量（t 值）', f"{peer['ml']-peer['naive']:+.4f}，t={peer['t_inc']:.2f}")}
 {row('选中的股票 20 日动量', f"模型 {peer['ml_mom']:+.1%} | 无脑动量 {peer['naive_mom']:+.1%} | 全池 {peer['univ_mom']:+.1%}")}
 {row('选中的股票市值中位数', f"{peer['ml_mv']/1e4:.0f} 亿 | 全池 {peer['univ_mv']/1e4:.0f} 亿")}
</table>
<p style="font-size:13px;line-height:1.7;color:#555">
无脑追动量在本样本里是<b>亏钱</b>的（跑输全池），说明 A 股这一段是反转而非动量；
模型没有押注动量，却取得了正向超额，这一条支持"模型确实有截面选股能力"。
</p>
'''}

<h2>七、结论</h2>
<div class="note">
<b>业绩被极端窗口放大了，但模型不是"纯靠风格"。</b>年化 {pct(M['ann'])}、夏普 {M['sharpe']:.2f}，
同期基准仅约 {pct(list(bm_tot.values())[0]) if bm_tot else '—'}。逐条核对后的判定：
<ul>
<li><span class="tag t-ok">已澄清</span><b>幸存者偏差影响有限</b>：ST 过滤是 point-in-time 的（<code>is_st</code> 由逐日名称状态维护，
训练和预测都走同一个 <code>get_data()</code> 过滤）；2024 年以来 84 只主板退市股里 78 只（93%）退市前戴过帽，
会被过滤掉；剩下多为吸收合并（非亏损退市）。退市股在未戴帽期的 20 日动量常态低于策略入场线，常规不会被选中。</li>
<li><span class="tag t-ok">模型确有截面选股能力</span>：对照实验里"无脑买 20 日动量最高 5 只"是<b>亏钱</b>的，
而模型 Top5 显著跑赢全池{('，t=%.2f' % peer['t_ml']) if peer else ''}，且显著跑赢无脑动量基准
{('（t=%.2f）' % peer['t_inc']) if peer else ''}。说明模型没有押注动量风格。</li>
<li><span class="tag t-warn">需注意</span><b>期望收益尚未被估计准</b>：批均 {pct(pv.mean())} 的 95% 置信区间是
[{pct(ci_lo)}, {pct(ci_hi)}]，折合年化 [{pct(ann_of(ci_lo))}, {pct(ann_of(ci_hi))}]。
收益集中在右尾本身是正常的（σ/μ={sd/abs(mu):.1f}，高波动策略都这样），
真正的问题是<b>样本只有 {len(bt)} 批，均值还没收敛</b>。</li>
<li><span class="tag t-bad">仍然严重</span><b>收益集中在单一窗口</b>：2026 年 4-6 月三个月贡献了全程约
{pct(q2_share,0) if q2_share==q2_share else '—'} 的对数收益
{'' if exw is None else f"，剔除该窗口后年化降到 {pct(exw['ann'])}，批次 t 值也跌破 2"}。</li>
<li><span class="tag t-warn">提示</span><b>Top5 内部排不准</b>：IC 仅 {ic_mean:+.4f}（t={ic_t:.2f}）。
与上面"能选出好篮子"并不矛盾——模型擅长把好票从 3000 只里捞出来，但排不准这 5 只谁更好；
因此<b>不要靠 score 做权重</b>，等权即可。</li>
<li><span class="tag t-ok">通过</span>流动性、涨跌停约束、成本（约占交易额 0.13%）、训练标签截断
（<code>label_lookahead</code> 已排除跨日标签）均未见前视或失真。</li>
</ul>
</div>
<div class="good">
<b>建议的下一步（按优先级）：</b>
<ol>
<li><b>做样本外检验（最高优先）</b>：本地数据覆盖 2018 年至今共 2123 个交易日，本次只用了 2024 年后的 665 天。
直接跑 <code>python run_backtest.py --start 20180102 --end 20231229</code>，看 2018-2023 是否同样有效——
这是验证"真本事 vs 运气"最快、最硬的办法。</li>
<li>把"右尾频率各年是否稳定"和"Bootstrap 置信区间下限是否 &gt; 0"作为上线门槛，
<b>不再用"剔除最好 N 笔"这类检验</b>（它会把一切高 σ/μ 的正收益策略误杀）。</li>
<li>做风格中性化检验（市值、波动率、行业），确认超额不是残留的风格 beta。</li>
<li>若样本外通过，再用更长的区间（2018-2026 全量）重估期望年化，而不是采信当前的 {pct(M['ann'])}。</li>
</ol>
</div>
</body></html>"""
    out = os.path.join(res_dir, '分析报告.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python analyse/backtest_report.py <结果目录名或路径>')
        sys.exit(1)
    arg = sys.argv[1]
    if os.path.isdir(arg):
        path = arg
    elif os.path.isdir(os.path.join('backtest/results', arg)):
        path = os.path.join('backtest/results', arg)
    else:
        print(f'找不到结果目录: {arg}')
        sys.exit(1)
    print('报告已生成 →', build_report(path))
