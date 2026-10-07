"""增量更新逻辑的离线自测（不联网、不污染真实数据目录之外的文件）。

1. build_st_flag：基于 stock_list 的增量 ST 标识
   - 历史行必须一条不少
   - 新日期按当前名称写入 is_st
   - 名称变更能被检测出来
2. merge_basic_daily_data：只跑 SQL 拼装与视图过滤（不落盘宽表）
"""
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_data as md

RAW = md.raw_data_path

# ---------------------------------------------------------------- 1. ST
print('=' * 70)
print('[测试 1] build_st_flag —— 增量追加 + 名称比对')
print('=' * 70)

tmp_dir = 'data/_inctest'
os.makedirs(tmp_dir, exist_ok=True)
test_st = os.path.join(tmp_dir, 'st_flag.parquet')
test_state = os.path.join(tmp_dir, 'name_state.csv')
shutil.copyfile('data/st_flag.parquet', test_st)

md.ST_FLAG_PATH = test_st
md.NAME_STATE_PATH = test_state

before = pd.read_parquet(test_st)
print(f'历史 st_flag: {len(before)} 行, 最新日期 {before.trade_date.max()}, ST {int(before.is_st.sum())} 行')

new = md.build_st_flag(dates=['20260925', '20260926'])
after = pd.read_parquet(test_st)
hist = after[~after.trade_date.isin(['20260925', '20260926'])]

assert new is not None and len(new) == 3196 * 2, (new is None, len(new) if new is not None else 0)
assert len(hist) == len(before), f'历史行被改动了: {len(hist)} != {len(before)}'
assert after.trade_date.max() == '20260926'
print(f'追加后: {len(after)} 行（新增 {len(new)}，历史保留 {len(hist)}）')
print(f'新增中 ST 行数: {int(new.is_st.sum())}')

sl = pd.read_csv(os.path.join(RAW, 'stock_list/stock_list.csv'))
st_codes = set(sl.loc[sl['name'].astype(str).str.upper().str.contains('ST'), 'ts_code'])
got = set(new.loc[new.is_st == 1, 'ts_code'])
print(f'stock_list 里 ST 只数: {len(st_codes)}，写入一致: {got == st_codes}')
assert got == st_codes

state = pd.read_csv(test_state)
print(f'name_state.csv: {len(state)} 行, 列 {list(state.columns)}')
assert len(state) == len(sl)

# 第二次跑：制造一次名称变更（把某只股票改名成 ST），看能否检测出来
print('\n--- 模拟一次戴帽/摘帽 ---')
tmp_raw = os.path.join(tmp_dir, 'raw')
os.makedirs(os.path.join(tmp_raw, 'stock_list'), exist_ok=True)
os.makedirs(os.path.join(tmp_raw, 'namechange'), exist_ok=True)
sl2 = sl.copy()
is_st_name = sl2['name'].astype(str).str.upper().str.contains('ST')
code_a = sl2.loc[~is_st_name, 'ts_code'].iloc[0]
code_b = sl2.loc[is_st_name, 'ts_code'].iloc[0]
old_a = sl2.loc[sl2.ts_code == code_a, 'name'].iloc[0]
old_b = sl2.loc[sl2.ts_code == code_b, 'name'].iloc[0]
sl2.loc[sl2.ts_code == code_a, 'name'] = '*ST测试A'     # 戴帽
sl2.loc[sl2.ts_code == code_b, 'name'] = old_b.replace('ST', '').replace('*', '')  # 摘帽
sl2.to_csv(os.path.join(tmp_raw, 'stock_list/stock_list.csv'), index=False)
shutil.copyfile('data/raw/trade_cal.csv', os.path.join(tmp_raw, 'trade_cal.csv'))

md.raw_data_path = tmp_raw + '/'
new2 = md.build_st_flag(dates=['20260927'])
a_row = new2[new2.ts_code == code_a].iloc[0]
b_row = new2[new2.ts_code == code_b].iloc[0]
print(f'  {code_a}: {old_a} → *ST测试A  写入 is_st={a_row.is_st}（期望 1）')
print(f'  {code_b}: {old_b} → {sl2.loc[sl2.ts_code==code_b,"name"].iloc[0]}  写入 is_st={b_row.is_st}（期望 0）')
assert a_row.is_st == 1 and b_row.is_st == 0

after2 = pd.read_parquet(test_st)
print(f'  文件最终 {len(after2)} 行（应为 {len(after)} + 3196）')
assert len(after2) == len(after) + 3196
md.raw_data_path = RAW

# ---------------------------------------------------------------- 2. merge
print()
print('=' * 70)
print('[测试 2] merge_basic_daily_data —— 增量过滤（只跑视图层，不落盘）')
print('=' * 70)
print('增量起点 since=20260920 时各数据源应只剩 20260921 之后的少量行：')
import duckdb
con = duckdb.connect()
con.execute("SET threads=4")
for sec in ['stock_data', 'adj_factor', 'daily_basic_data', 'margin_detail', 'moneyflow', 'top_list']:
    files = sorted(__import__('glob').glob(RAW + sec + '/*.csv'))
    if sec in {'daily_basic_data', 'margin_detail', 'moneyflow', 'top_list'}:
        picked = [f for f in files if os.path.basename(f)[:-4].isdigit() and os.path.basename(f)[:-4] > '20260920']
    else:
        picked = files
    src = "[" + ", ".join("'" + os.path.abspath(p).replace('\\', '/') + "'" for p in picked) + "]"
    con.execute(f"CREATE OR REPLACE VIEW _s AS SELECT * FROM read_csv_auto({src}, union_by_name=true, "
                f"types={{'ts_code':'VARCHAR','trade_date':'VARCHAR'}})")
    n = con.execute("SELECT count(*) FROM _s WHERE CAST(trade_date AS VARCHAR) > '20260920'").fetchone()[0]
    print(f'  {sec:18s} 文件 {len(picked):5d} 个 → 新日期行 {n}')
con.close()

print('\n全部断言通过 ✓')
