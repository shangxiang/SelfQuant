import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_DIR = '../data/section/'  # CSV 文件所在目录
START_DATE = '20230103'  # 回测开始日期（需保证有足够历史数据）
END_DATE = '20230401'  # 回测结束日期
WINDOW = 10  # 训练窗口长度（交易日）
FACTOR_COLS = ['pe_standard', 'mfi_standard', 'macd_standard', 'pe_ttm_standard',
               'cci_standard', 'J_standard', "turnover_rate_x_standard"]  # 因子列名
LABEL_COL = 'label'  # 标签列名
DATE_FORMAT = '%Y%m%d'  # 文件命名日期格式，如 20240102.csv
TRADE_CAL_FILE = '../data/raw/trade_cal.csv'   # 交易日历文件路径
CAL_DATE_COL = 'cal_date'          # 日期列名
SMOOTH = False  # 是否对权重做EMA平滑
SMOOTH_ALPHA = 0.2  # 平滑系数


# ==================== 工具函数 ====================
def load_data(date_str):
    """加载某个日期的数据，date_str 格式如 '20240102'"""
    path = f"{DATA_DIR}{date_str}.csv"
    df = pd.read_csv(path, parse_dates=False)
    # 假设文件中已有日期列 'trade_date' 或无需额外列
    # 确保因子列为数值
    df[FACTOR_COLS] = df[FACTOR_COLS].astype(float)
    df[LABEL_COL] = df[LABEL_COL].astype(float)
    return df


def get_trading_dates(start, end):
    """根据实际文件存在生成交易日列表，或直接用日期范围"""
    # 这里简化为按 pandas 日期范围生成，实际应读取目录下所有 csv 文件名
    dates = pd.bdate_range(start=START_DATE, end=END_DATE).strftime('%Y%m%d').tolist()
    # 可在此处过滤掉无数据的日期
    return dates


def load_trading_dates(cal_file, start=None, end=None):
    cal = pd.read_csv(cal_file, dtype={CAL_DATE_COL: str})
    dates = cal[CAL_DATE_COL].dropna().str.strip().tolist()
    dates = sorted(dates)
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    return dates


# ==================== 滚动权重计算 ====================
#all_dates = get_trading_dates(START_DATE, END_DATE)
all_dates = load_trading_dates(TRADE_CAL_FILE, START_DATE, END_DATE)
# 确保前 WINDOW 天有足够数据，至少需要 WINDOW+5 天才能有标签
# 初始化权重平滑存储
previous_weights = None
weights_history = {}  # 存储每日权重，方便后续分析
score_history = {}  # 存储每日各股票得分

for i, today in enumerate(all_dates):
    # 窗口日期列表：从今天往前 WINDOW-1 天（不含今天）到今天
    window_start_idx = i - WINDOW + 1
    if window_start_idx < 0:
        continue  # 不足窗口长度，跳过
    window_dates = all_dates[window_start_idx: i + 1]  # 包含今天

    # ---------- 构建训练集 ----------
    X_list, y_list = [], []
    valid_sample_count = 0
    for t_date in window_dates:
        try:
            df_t = load_data(t_date)
        except FileNotFoundError:
            continue

        # 检查该日的 label 是否可用：要求 t_date 往后推5个交易日不超过今天
        # 这里简化为比较 t_date + 5 个自然日，准确应用交易日列表索引
        t_date_dt = pd.to_datetime(t_date, format=DATE_FORMAT)
        # 找到 t_date 在 all_dates 中的位置
        try:
            t_idx = all_dates.index(t_date)
        except ValueError:
            continue  # 当前日期不在交易日列表中
        if t_idx + 5 <= i:  # t+5 日应 <= 今天
            X_list.append(df_t[FACTOR_COLS].values)
            y_list.append(df_t[LABEL_COL].values)
            valid_sample_count += df_t.shape[0]
        # 如果 t+5 > 今天，则标签不可用，不加入训练

    if len(X_list) == 0:
        print(f"日期 {today}: 无有效训练样本，跳过")
        continue

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    print(f"日期 {today}: 训练样本数 {X_train.shape[0]}, 因子数 {X_train.shape[1]}")

    # 1. 处理 y 的缺失：直接丢弃（标签缺失，该样本必须舍弃）
    valid_y = ~np.isnan(y_train)
    X_train = X_train[valid_y]
    y_train = y_train[valid_y]

    # 2. 处理 X 的缺失：NaN 全部置 0（即标准化均值）
    X_train = np.nan_to_num(X_train, nan=0.0)

    print(f"日期 {today}: 有效样本 {X_train.shape[0]}")


    # ---------- 弹性网络训练 ----------
    # 注意：数据已中心化，但不同因子量纲可能仍有差异，建议再次标准化
    # 由于用户声称已做好处理，这里省略标准化步骤，直接训练
    # 若需标准化，取消下面两行注释
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    # 使用弹性网络，网格搜索 L1_ratio 和 alpha
    model = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, 1],
        alphas=None,  # 默认自动生成 alpha 序列
        cv=5,  # 5折交叉验证
        max_iter=5000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    weights = model.coef_.copy()
    print(f"  最佳 alpha: {model.alpha_:.6f}, l1_ratio: {model.l1_ratio_:.3f}")
    print(f"  非零权重因子数: {np.sum(np.abs(weights) > 1e-6)}")

    # ---------- 权重平滑（可选） ----------
    if SMOOTH and previous_weights is not None:
        weights = SMOOTH_ALPHA * weights + (1 - SMOOTH_ALPHA) * previous_weights
    previous_weights = weights
    weights_history[today] = dict(zip(FACTOR_COLS, weights))

    # ---------- 对最新一天打分 ----------
    df_today = load_data(today)
    # 同样，如果训练时用了标准化，这里也需要用同一个 scaler 变换
    X_today = df_today[FACTOR_COLS].values
    # 若前面使用了 scaler，则：X_today = scaler.transform(X_today)
    score = X_today.dot(weights)  # (n_stocks,)

    # 保存得分，可附加股票代码（如果文件包含 stock_code 列）
    score_df = pd.DataFrame({
        'score': score
    })
    if 'stock_code' in df_today.columns:
        score_df['stock_code'] = df_today['stock_code'].values
    score_history[today] = score_df

# ==================== 输出结果 ====================
# 查看某一天的权重
example_date = all_dates[WINDOW]  # 第一个有结果的日期
print(f"\n====== {example_date} 因子权重 ======")
for k, v in weights_history.get(example_date, {}).items():
    print(f"{k}: {v:.6f}")

# 权重历史保存为 DataFrame
weights_df = pd.DataFrame(weights_history).T  # 行是日期，列是因子
print("\n权重统计摘要：")
print(weights_df.describe())

# 可选：保存权重与得分
weights_df.to_csv('factor_weights_daily.csv')
# 如果要将得分保存，可将 score_history 中的每个 DataFrame 合并后写入文件


# 对于某个交易日 today
score_df = score_history[today]    # DataFrame，包含 'score' 和 'stock_code'
# 按得分降序排序，得分越高，预期未来5日涨幅越大
score_df = score_df.sort_values('score', ascending=False)
# 选取前 N 只股票作为买入组合，例如前10%（或固定100只）
top_n = int(len(score_df) * 0.1)
buy_list = score_df.head(top_n)['stock_code'].tolist()
# 如果你也做空（得分最低的），同样取尾部
short_list = score_df.tail(top_n)['stock_code'].tolist()