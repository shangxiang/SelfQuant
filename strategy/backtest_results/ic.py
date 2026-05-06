import pandas as pd
import numpy as np

# ==================== 1. 读取数据 ====================
df = pd.read_csv('ic_series.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ==================== 2. 非重叠5日采样 ====================
# 假设每个交易日IC对应的是该日做出的「未来5日」预测，
# 取第1日、第6日、第11日…… 的IC，这些预测区间完全不重叠
step = 5
df_sample = df.iloc[::step].copy()

# ==================== 3. 统计对比 ====================
def ic_stats(series, label):
    mean_ic = series.mean()
    std_ic  = series.std(ddof=1)
    ir      = mean_ic / std_ic if std_ic != 0 else np.nan
    ratio   = (series > 0).sum() / len(series)
    print(f'{label:　<12s}  样本数: {len(series):5d}  '
          f'IC均值: {mean_ic:.4f}  IC标准差: {std_ic:.4f}  '
          f'IR: {ir:.4f}  IC>0: {ratio:.2%}')

print('========== IC统计对比 ==========')
ic_stats(df['ic'], '全量（天频）')
ic_stats(df_sample['ic'], '非重叠5日')

# 如果需要保存降采样后的数据
# df_sample.to_csv('ic_5day_nonoverlap.csv', index=False)