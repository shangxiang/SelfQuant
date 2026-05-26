from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseTimingStrategy(ABC):
    """
    择时策略抽象基类。

    择时策略决定每次建仓时使用多少比例的可用资金，与选股策略正交：
      - 选股策略回答"买哪些股票"
      - 择时策略回答"这次投入多少钱"

    引擎在回测开始前调用 prepare() 进行预计算，
    之后在每个信号生成日调用 get_position_ratio() 获取当日建仓比例。
    注意：get_position_ratio() 返回的是 T 日的仓位判断，用于 T+1 日的实际买入，
    因此内部使用 T 日数据（而非 T+1）不构成未来函数问题。
    """

    def prepare(self, start_date: str, end_date: str) -> None:
        """
        回测开始前的预计算（如读取指数文件、计算均线序列）。
        子类可选覆盖，默认空实现（表示不需要预计算）。

        Parameters
        ----------
        start_date : str  回测开始日期 YYYYMMDD
        end_date   : str  回测结束日期 YYYYMMDD
        """
        pass

    @abstractmethod
    def get_position_ratio(self, date_str: str) -> float:
        """
        返回 date_str 当日对应的建仓比例。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期

        Returns
        -------
        float  0.0 = 不建仓（空仓），1.0 = 全仓，0.6 = 用 60% 可用资金建仓
        """
        pass

    def on_batch_sold(self, profit: float) -> None:
        """
        每批持仓全部卖出后由引擎回调，传入本批已实现收益率。

        profit = (卖出总收益 - 买入总成本) / 买入总成本

        子类可覆盖此方法以更新内部状态（如记录上期亏损）。
        默认空实现，对不需要历史信息的择时策略无影响。
        """
        pass


class MATiming(BaseTimingStrategy):
    """
    均线择时：以指数收盘价相对于 N 日均线的位置判断牛熊。

    判断逻辑（使用前一日数据，避免用当日收盘价做当日决策）：
      昨日收盘 > 昨日 N 日均线 → 牛市，满仓（1.0）
      昨日收盘 ≤ 昨日 N 日均线 → 熊市，空仓（0.0）

    若需要渐进式仓位（如距均线越远仓位越低），可继承此类并覆盖 get_position_ratio()。
    """

    def __init__(self, index_file: str, ma_period: int = 60):
        """
        Parameters
        ----------
        index_file : str  指数日线 CSV 路径，需含 trade_date、close 列
        ma_period  : int  均线周期（交易日数），默认 60 日
        """
        self.index_file = index_file
        self.ma_period = ma_period
        # 预计算结果缓存：{date_str: position_ratio}
        # 在 prepare() 中一次性填充，get_position_ratio() 直接查表
        self._map: dict[str, float] = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        """
        读取指数日线，预计算全区间内每日的牛熊状态并存入 self._map。

        使用 shift(1) 取前一日数据：T 日的牛熊判断基于 T-1 日收盘，
        与引擎的信号→成交时序一致（T 日判断 → T+1 日买入）。
        """
        index_df = pd.read_csv(self.index_file, parse_dates=['trade_date'])
        index_df = index_df.sort_values('trade_date')
        close = index_df['close']
        ma = close.rolling(self.ma_period).mean()

        # shift(1)：用昨日收盘和昨日均线判断今日牛熊，避免使用今日收盘做今日决策
        is_bull = (close.shift(1) > ma.shift(1)).fillna(False)

        self._map = dict(zip(
            index_df['trade_date'].dt.strftime('%Y%m%d'),
            is_bull.map({True: 1.0, False: 0.0})
        ))

    def get_position_ratio(self, date_str: str) -> float:
        """
        查表返回当日建仓比例。

        若 date_str 不在预计算结果中（如数据缺失），默认返回 1.0（满仓），
        以避免因数据缺口意外跳过建仓。
        """
        # return self._map.get(date_str, 1.0)
        return 1.0


class StyleConvergenceTiming(BaseTimingStrategy):
    """
    大小盘风格趋同择时。

    当小微盘（932000.CSI）与大中盘（000510.CSI）同时满足以下条件时，
    判定市场处于"系统性下行"状态，将建仓比例降为 avoid_ratio（默认空仓）：

        条件1：滚动 roll_window 日相关系数 > corr_threshold（两者高度趋同）
        条件2：小盘指数过去 roll_window 日累计收益 < 0
        条件3：大盘指数过去 roll_window 日累计收益 < 0

    逻辑含义：
        大小盘趋同说明风格轮动失效，同步下行说明整体风险偏好在收缩。
        此时选股 alpha 很难抵御 beta 下行，持股收益往往偏差。

    信号使用 T 日收盘数据，引擎在 T 日生成信号、T+1 日执行买入，不存在未来函数。
    """

    def __init__(
        self,
        small_file: str = 'data/raw/index_daily/932000.CSI.csv',
        large_file: str = 'data/raw/index_daily/000510.CSI.csv',
        roll_window: int = 5,
        corr_threshold: float = 0.75,
        avoid_ratio: float = 0.0,
    ):
        """
        Parameters
        ----------
        small_file       : 小微盘指数日线 CSV（需含 trade_date、pct_chg 列）
        large_file       : 大中盘指数日线 CSV
        roll_window      : 滚动窗口天数，同时用于相关性和累计收益计算
        corr_threshold   : 相关系数触发阈值，超过则认为趋同
        avoid_ratio      : 信号0时的建仓比例，0.0=空仓，0.5=半仓
        """
        self.small_file     = small_file
        self.large_file     = large_file
        self.roll_window    = roll_window
        self.corr_threshold = corr_threshold
        self.avoid_ratio    = avoid_ratio
        self._map: dict[str, float] = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        """
        读取两只指数日线，预计算全区间信号并存入 self._map。
        由于需要 roll_window 天的历史才能输出第一个信号，
        实际读取范围从文件最早日期起，不受 start_date 限制。
        """
        def _load(path: str) -> pd.Series:
            df = pd.read_csv(path, index_col=0)
            df['trade_date'] = pd.to_datetime(
                df['trade_date'].astype(str), format='%Y%m%d'
            )
            return (
                df.sort_values('trade_date')
                  .set_index('trade_date')['pct_chg'] / 100
            )

        small = _load(self.small_file)
        large = _load(self.large_file)
        df = pd.concat([small.rename('small'), large.rename('large')],
                       axis=1).dropna()

        df['roll_corr'] = df['small'].rolling(self.roll_window).corr(df['large'])
        df['small_cum'] = df['small'].rolling(self.roll_window).sum()
        df['large_cum'] = df['large'].rolling(self.roll_window).sum()

        avoid = (
            (df['roll_corr'] > self.corr_threshold)
            & (df['small_cum'] < 0)
            & (df['large_cum'] < 0)
        )
        ratio = avoid.map({True: self.avoid_ratio, False: 1.0}).fillna(1.0)

        self._map = dict(zip(df.index.strftime('%Y%m%d'), ratio))

    def get_position_ratio(self, date_str: str) -> float:
        """
        返回当日建仓比例。
        数据不足 roll_window 天时（回测初期）默认满仓，不强制跳过建仓。
        """
        return self._map.get(date_str, 1.0)


class LastBatchTiming(BaseTimingStrategy):
    """
    上期收益择时：根据连续亏损次数动态递减仓位，盈利后立即恢复满仓。

    规则：
        上期盈利（profit >= 0）→ 仓位重置为 1.0（满仓）
        上期亏损（profit <  0）→ 当前仓位 × half_ratio（每亏一次打一折）
        首次入场（无历史）     → 满仓

    示例（half_ratio=0.5）：
        第1次亏损 → 0.5 仓
        第2次连亏 → 0.25 仓
        第3次连亏 → 0.125 仓
        中间任意一次盈利 → 重置回 1.0 仓

    Parameters
    ----------
    half_ratio : float  每次亏损后仓位的乘数，默认 0.5（减半）
    min_ratio  : float  仓位下限，防止连续亏损后仓位趋近于零，默认 0.0（不限制）
    """

    def __init__(self, half_ratio: float = 0.5, min_ratio: float = 0.0):
        self.half_ratio  = half_ratio
        self.min_ratio   = min_ratio
        self._ratio: float = 1.0   # 当前仓位，随每期结果动态更新

    def on_batch_sold(self, profit: float) -> None:
        if profit >= 0:
            self._ratio = 1.0
        else:
            self._ratio = max(self._ratio * self.half_ratio, self.min_ratio)

    def get_position_ratio(self, date_str: str) -> float:
        return self._ratio


class BlindWindowTiming(BaseTimingStrategy):
    """
    盲窗口行情择时：根据买入信号日前 T-6~T 的大小盘市场特征决定仓位。

    ElasticNet 模型训练集截止到 T-6（label_lookahead=6），T-6 到 T 这段行情
    是模型的"盲窗口"——模型未见过这段数据。实证发现，当这段窗口内：
      - 大小盘日收益相关性高（corr_5d 高）：大小盘同向，因子信号有效性更强
      - 大盘日波动率高（l_vol 高）：市场活跃，因子区分度更高
    两者同时满足时批次胜率约 67%，均值收益 +1.57%；
    任一偏低时胜率降至 43%，均值收益 -0.19%，不如空仓。

    仓位规则（加入趋势滤波后）：
      corr 低 AND vol 低             →  avoid_ratio（双重确认，无论趋势）
      corr 低 AND vol 正常            →  avoid_ratio（大小盘背离，可靠熊市信号）
      corr 正常 AND vol 低 AND 在趋势上方  →  1.0（牛市低波动是正常现象，不空仓）
      corr 正常 AND vol 低 AND 在趋势下方  →  avoid_ratio（死市，无方向，空仓）
      corr 正常 AND vol 正常           →  1.0

    趋势判断：小微盘指数昨日收盘 > 昨日 trend_ma 日均线（shift(1) 避免前视）。

    阈值使用滚动历史分位数（rolling_window 期），避免使用未来数据。

    Parameters
    ----------
    small_file      : 小微盘指数日线 CSV（932000.CSI，需含 trade_date、close 列）
    large_file      : 大中盘指数日线 CSV（000510.CSI，需含 trade_date、close 列）
    signal_window   : 计算 corr/vol 所用的滚动窗口天数，默认 7（取约 6 个日收益）
    corr_pct        : corr_5d 的历史分位数阈值，低于此分位数时触发空仓，默认 20
    vol_pct         : l_vol 的历史分位数阈值，低于此分位数时触发空仓，默认 33
    rolling_window  : 计算分位数阈值所用的历史样本数（滚动），默认 80（与 ElasticNet 训练窗口对齐）
    avoid_ratio     : 触发条件时的仓位，默认 0.0（空仓）
    trend_ma        : 趋势判断均线周期（交易日数），默认 20；设为 0 则禁用趋势滤波，
                      退化为原始 OR 逻辑
    """

    def __init__(
        self,
        small_file: str = 'data/raw/index_daily/932000.CSI.csv',
        large_file: str = 'data/raw/index_daily/000510.CSI.csv',
        signal_window: int = 7,
        corr_pct: float = 20.0,
        vol_pct: float = 33.0,
        rolling_window: int = 80,
        avoid_ratio: float = 0.0,
        trend_ma: int = 20,
    ):
        self.small_file     = small_file
        self.large_file     = large_file
        self.signal_window  = signal_window
        self.corr_pct       = corr_pct
        self.vol_pct        = vol_pct
        # rolling_window 与 ElasticNet 训练窗口对齐（默认 80 交易日 ≈ 16 批次）：
        # 用模型"见过"的同等长度历史来判断当前市场环境是否异常，逻辑自洽。
        # 过短（<40）噪声大，过长（>120）阈值过于保守，80 在敏感性分析中表现稳健。
        self.rolling_window = rolling_window
        self.avoid_ratio    = avoid_ratio
        self.trend_ma       = trend_ma
        self._map: dict[str, float] = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        """
        读取两只指数日线，逐日计算 corr_5d 和 l_vol，
        再用滚动历史分位数确定阈值，最终生成每日仓位映射。
        """
        import numpy as np

        def _load_close(path: str) -> pd.Series:
            df = pd.read_csv(path)
            df['trade_date'] = df['trade_date'].astype(str)
            return df.sort_values('trade_date').set_index('trade_date')['close'].astype(float)

        small_close = _load_close(self.small_file)
        large_close = _load_close(self.large_file)

        # 对齐日期，计算日收益率
        idx = small_close.index.intersection(large_close.index)
        s_ret = small_close.loc[idx].pct_change()
        l_ret = large_close.loc[idx].pct_change()

        df = pd.DataFrame({'s_ret': s_ret, 'l_ret': l_ret,
                           's_close': small_close.loc[idx]}).dropna()

        # 滚动计算 corr_5d 和 l_vol（基于 signal_window 个日收益）
        w = self.signal_window
        df['corr_5d'] = df['s_ret'].rolling(w).corr(df['l_ret'])
        df['l_vol']   = df['l_ret'].rolling(w).std()
        df = df.dropna()

        # 滚动分位数阈值：用过去 rolling_window 个观测值计算，避免未来函数
        rw = self.rolling_window
        df['corr_thresh'] = df['corr_5d'].rolling(rw, min_periods=rw // 2).quantile(
            self.corr_pct / 100
        )
        df['vol_thresh'] = df['l_vol'].rolling(rw, min_periods=rw // 2).quantile(
            self.vol_pct / 100
        )

        # 基础触发：corr 和 vol 各自是否低于阈值
        avoid_corr = df['corr_5d'] < df['corr_thresh']
        avoid_vol  = df['l_vol']   < df['vol_thresh']

        if self.trend_ma > 0:
            # 趋势滤波：小微盘昨日收盘是否在 trend_ma 日均线上方
            # shift(1) 使用昨日数据，与 prepare 整体的时序惯例一致
            ma = df['s_close'].rolling(self.trend_ma).mean()
            above_trend = (df['s_close'].shift(1) > ma.shift(1)).fillna(False)

            # 仅 vol 触发（corr 正常）且处于上升趋势时，不避仓：
            #   低波动在牛市中是正常的"平静上涨"，不是危险信号
            # corr 触发（大小盘背离）或两者同时触发时，无论趋势都避仓：
            #   大小盘背离意味着系统性风险或结构性转折，是更可靠的熊市信号
            only_vol_in_uptrend = avoid_vol & ~avoid_corr & above_trend
            avoid = (avoid_corr | avoid_vol) & ~only_vol_in_uptrend
        else:
            # trend_ma=0：禁用趋势滤波，退化为原始 OR 逻辑
            avoid = avoid_corr | avoid_vol

        ratio = avoid.map({True: self.avoid_ratio, False: 1.0}).fillna(1.0)

        # 时序对齐：engine 在 T+1 日买入时调用 get_position_ratio(T+1)，
        # 但仓位判断必须基于 T 日收盘数据（T+1 收盘价是买入当天的未来数据）。
        # 将 ratio 整体向后移一位：T 日计算的信号存入 T+1 日的 key，
        # 使 engine 查 T+1 时拿到的是 T 日的判断，不引入未来函数。
        ratio_shifted = ratio.shift(1)

        self._map = dict(zip(df.index, ratio_shifted.values))

    def get_position_ratio(self, date_str: str) -> float:
        """
        返回当日建仓比例。
        历史数据不足（回测初期）或 shift 导致的 NaN 时默认满仓，不强制跳过建仓。
        """
        v = self._map.get(date_str, 1.0)
        import math
        return 1.0 if (v != v) else v  # NaN check: NaN != NaN


class LGBMDriftTiming(BaseTimingStrategy):
    """
    LGBM 因子漂移择时。

    核心逻辑：
      LGBM 模型在 [T-window, T-label_lookahead] 窗口训练，
      盲窗口 [T-label_lookahead+1, T] 的因子形态是模型从未见过的。
      若盲窗口内各因子的截面均值相对于训练窗口发生了显著漂移，
      说明模型的推断规则可能已不适用当前市场，应降低仓位。

    漂移分数计算：
      1. 训练窗口：[T-window, T-label_lookahead]，逐日计算各因子的截面均值
         → 得到每因子在训练窗口内的均值 μ_train 和标准差 σ_train
      2. 盲窗口：  [T-label_lookahead+1, T]，逐日计算各因子截面均值
         → 得到每因子在盲窗口内的均值 μ_blind
      3. 单因子 z-score = |μ_blind - μ_train| / (σ_train + ε)
      4. 漂移总分 = 各因子 z-score 的平均值（剔除 NaN）
      漂移分数越高 = 盲窗口因子环境与训练窗口差异越大 = 模型可靠性越低

    仓位规则：
      漂移分数 < 历史 drift_high_pct 分位数  →  1.0（环境稳定，满仓）
      漂移分数 ≥ 历史 drift_high_pct 分位数  →  avoid_ratio（环境漂移，降仓）

    Parameters
    ----------
    data_dir        : 截面数据目录（与 LGBMConfig.data_dir 相同）
    calendar_file   : 交易日历 CSV
    factor_cols     : 与 LGBMConfig.factor_cols 相同的因子列表
    window          : 训练窗口长度（与 LGBMConfig.window 对齐）
    label_lookahead : 盲窗口长度（与 LGBMConfig.label_lookahead 对齐）
    rolling_window  : 计算漂移分数历史分位数的窗口（期数），默认 40
    drift_high_pct  : 漂移分数触发空仓的历史分位数（%），默认 75
    avoid_ratio     : 触发漂移时的建仓比例，默认 0.0
    """

    def __init__(
        self,
        data_dir: str = 'data/section/',
        calendar_file: str = 'data/raw/trade_cal.csv',
        factor_cols: list = None,
        window: int = 40,
        label_lookahead: int = 6,
        rolling_window: int = 40,
        drift_high_pct: float = 75.0,
        avoid_ratio: float = 0.0,
    ):
        self.data_dir        = data_dir
        self.calendar_file   = calendar_file
        self.factor_cols     = factor_cols or []
        self.window          = window
        self.label_lookahead = label_lookahead
        self.rolling_window  = rolling_window
        self.drift_high_pct  = drift_high_pct
        self.avoid_ratio     = avoid_ratio
        self._map: dict[str, float] = {}

    def _load_cross_means(self, dates: list) -> pd.DataFrame:
        """
        批量读取截面文件，计算每日各因子截面均值。
        返回 DataFrame，index=date_str，columns=factor_cols。
        """
        import os
        import numpy as np
        rows = {}
        for d in dates:
            path = os.path.join(self.data_dir, f'{d}.csv')
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, usecols=lambda c: c in self.factor_cols or c == 'ts_code')
            except Exception:
                continue
            available = [c for c in self.factor_cols if c in df.columns]
            if not available:
                continue
            rows[d] = df[available].mean(numeric_only=True)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).T  # index=date, columns=factors

    def prepare(self, start_date: str, end_date: str) -> None:
        import numpy as np
        import os

        cal = pd.read_csv(self.calendar_file, dtype={'cal_date': str})
        all_dates = sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())
        # 只处理回测区间
        dates_in_range = [d for d in all_dates if start_date <= d <= end_date]

        # 一次性预读所有需要的截面日均值（训练窗口 + 盲窗口覆盖的所有日期）
        # 取足够早的起点以支持第一个训练窗口
        start_idx = max(0, all_dates.index(start_date) - self.window - 5)
        needed_dates = all_dates[start_idx: all_dates.index(end_date) + 1]
        print(f'  [LGBMDriftTiming] 预读 {len(needed_dates)} 天截面均值...')
        means_df = self._load_cross_means(needed_dates)
        if means_df.empty:
            print('  [LGBMDriftTiming] 截面数据为空，所有日期默认满仓')
            return

        # 计算每个信号日 T 的漂移分数
        drift_series = {}
        for t in dates_in_range:
            if t not in all_dates:
                continue
            t_idx = all_dates.index(t)

            # 训练窗口：[T-window, T-label_lookahead]
            train_start_idx = max(0, t_idx - self.window + 1)
            train_end_idx   = t_idx - self.label_lookahead
            if train_end_idx < train_start_idx:
                continue
            train_dates = all_dates[train_start_idx: train_end_idx + 1]

            # 盲窗口：[T-label_lookahead+1, T]
            blind_dates = all_dates[t_idx - self.label_lookahead + 1: t_idx + 1]

            train_sub = means_df.loc[means_df.index.isin(train_dates)]
            blind_sub = means_df.loc[means_df.index.isin(blind_dates)]
            if train_sub.empty or blind_sub.empty:
                continue

            mu_train  = train_sub.mean()
            std_train = train_sub.std().clip(lower=1e-8)
            mu_blind  = blind_sub.mean()

            z_scores = ((mu_blind - mu_train) / std_train).abs()
            drift_series[t] = float(z_scores.mean(skipna=True))

        if not drift_series:
            return

        drift = pd.Series(drift_series)

        # 滚动历史分位数阈值
        rw = self.rolling_window
        high_thresh = drift.rolling(rw, min_periods=max(5, rw // 4)).quantile(
            self.drift_high_pct / 100
        )

        # 漂移 >= 阈值 → avoid；否则 → 满仓
        ratio = pd.Series(
            np.where(drift >= high_thresh, self.avoid_ratio, 1.0),
            index=drift.index,
        )

        # shift(1)：T 日信号存入 T+1 key，engine 在 T+1 日买入时查到 T 的判断
        ratio_shifted = ratio.shift(1)
        self._map = dict(zip(ratio_shifted.index, ratio_shifted.values))

    def get_position_ratio(self, date_str: str) -> float:
        v = self._map.get(date_str, 1.0)
        return 1.0 if (v != v) else float(v)


class BlindWindowVolTiming(BaseTimingStrategy):
    """
    盲窗口波动率择时（专为 LGBM 小微盘策略设计）。

    LGBM 训练截止 T-label_lookahead（默认 T-6），T-5 到 T 这 6 个交易日是模型的
    "盲窗口"——模型完全无法使用这段数据训练。实证发现：
      - 盲窗口内波动率高 → 批次胜率 85%，均收益 +1.55%
      - 盲窗口内波动率低 → 批次胜率 51%，均收益 -0.03%（与随机无异）

    判断逻辑：
      1. 计算信号日 T 过去 signal_window 个交易日的指数日收益率标准差（即盲窗口波动率）
      2. 与滚动历史 rolling_window 期的第 vol_low_pct 和 vol_high_pct 分位数比较：
           vol < low_thresh  → avoid_ratio（低波动，市场无催化，空仓/减仓）
           vol ≥ high_thresh → 1.0（高波动，市场活跃，满仓）
           中间区间           → mid_ratio（半仓观望）

    时序说明：
      信号在 T 日计算，存入 T+1 日的 key（shift(1)），因此 engine 在 T+1 日
      调用 get_position_ratio(T+1) 时取到的是基于 T 日数据的判断，无未来函数。

    Parameters
    ----------
    index_file     : 指数日线 CSV（需含 trade_date、pct_chg 列），默认中证 2000
    signal_window  : 盲窗口长度（交易日数），与 label_lookahead 对齐，默认 6
    rolling_window : 历史分位数计算窗口（交易日数），默认 80
    vol_low_pct    : 低波动触发阈值（历史分位数 %），低于此空仓/减仓，默认 33
    vol_high_pct   : 高波动阈值（历史分位数 %），高于此满仓，默认 67
    avoid_ratio    : 低波动时的建仓比例，默认 0.0（空仓）
    mid_ratio      : 中间波动时的建仓比例，默认 0.5（半仓）
    """

    def __init__(
        self,
        index_file: str = 'data/raw/index_daily/932000.CSI.csv',
        signal_window: int = 6,
        rolling_window: int = 80,
        vol_low_pct: float = 33.0,
        vol_high_pct: float = 67.0,
        avoid_ratio: float = 0.0,
        mid_ratio: float = 0.5,
    ):
        self.index_file    = index_file
        self.signal_window = signal_window
        self.rolling_window = rolling_window
        self.vol_low_pct   = vol_low_pct
        self.vol_high_pct  = vol_high_pct
        self.avoid_ratio   = avoid_ratio
        self.mid_ratio     = mid_ratio
        self._map: dict[str, float] = {}

    def prepare(self, start_date: str, end_date: str) -> None:
        import numpy as np

        df = pd.read_csv(self.index_file)
        df['trade_date'] = df['trade_date'].astype(str)
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['ret'] = df['pct_chg'] / 100

        # 盲窗口波动率：过去 signal_window 日日收益的标准差
        df['blind_vol'] = df['ret'].rolling(self.signal_window).std()
        df = df.dropna(subset=['blind_vol'])

        # 滚动历史分位数阈值，避免未来函数
        rw = self.rolling_window
        df['low_thresh'] = (
            df['blind_vol']
            .rolling(rw, min_periods=rw // 2)
            .quantile(self.vol_low_pct / 100)
        )
        df['high_thresh'] = (
            df['blind_vol']
            .rolling(rw, min_periods=rw // 2)
            .quantile(self.vol_high_pct / 100)
        )

        def _ratio(row):
            if pd.isna(row['low_thresh']) or pd.isna(row['high_thresh']):
                return 1.0
            if row['blind_vol'] < row['low_thresh']:
                return self.avoid_ratio
            if row['blind_vol'] >= row['high_thresh']:
                return 1.0
            return self.mid_ratio

        df['ratio'] = df.apply(_ratio, axis=1)

        # shift(1)：T 日计算的信号存入 T+1 日 key，engine 查 T+1 时取到 T 的判断
        df['ratio_shifted'] = df['ratio'].shift(1)

        self._map = dict(zip(df['trade_date'], df['ratio_shifted']))

    def get_position_ratio(self, date_str: str) -> float:
        import math
        v = self._map.get(date_str, 1.0)
        return 1.0 if (v != v) else float(v)  # NaN → 满仓（数据不足时不强制跳过）


class HybridTiming(BaseTimingStrategy):
    """
    盲窗口 + 风格趋同融合择时。

    仓位矩阵（b=BlindWindow信号, s=StyleConvergence信号）：
      b=1, s=1  →  1.0          双重确认，全仓
      b=1, s=0  →  partial_style  StyleConvergence 预警趋同下行，主动降仓
      b=0, s=1  →  partial_blind  BlindWindow 谨慎但无系统性下行，保留反弹仓位
      b=0, s=0  →  0.0          双重回避，空仓

    设计动机：
      - BlindWindow  在单边熊市防守强，但其他行情上涨乏力（OR 触发过于保守）
      - StyleConvergence 能识别反转最佳切入点，但下跌市场回撤偏大（AND 触发过于迟钝）
      - 融合后：BlindWindow 保留主防守线；StyleConvergence 提供双向修正：
          * BlindWindow 空仓但 StyleConvergence 无趋同信号 → 允许 partial_blind 仓位，
            捕捉 BlindWindow 因低相关/低波动而错过的风格分化牛市行情
          * BlindWindow 满仓但 StyleConvergence 检测到趋同下行 → 降至 partial_style，
            在系统性下行早期阶段主动减少暴露

    Parameters
    ----------
    small_file     : 小微盘指数日线 CSV（932000.CSI，需含 trade_date, close, pct_chg 列）
    large_file     : 大中盘指数日线 CSV（000510.CSI）
    signal_window  : BlindWindow 滚动窗口（corr/vol 计算，默认 7）
    corr_pct       : BlindWindow 相关系数历史分位数阈值（默认 20）
    vol_pct        : BlindWindow 波动率历史分位数阈值（默认 33）
    rolling_window : BlindWindow 分位数计算用的历史样本数（默认 80）
    roll_window    : StyleConvergence 滚动窗口（默认 5）
    corr_threshold : StyleConvergence 相关系数触发阈值（默认 0.75）
    partial_blind  : BlindWindow=避 & StyleConvergence=投时的仓位（默认 0.3）
    partial_style  : BlindWindow=投 & StyleConvergence=避时的仓位（默认 0.5）
    """

    def __init__(
        self,
        small_file: str = 'data/raw/index_daily/932000.CSI.csv',
        large_file: str = 'data/raw/index_daily/000510.CSI.csv',
        signal_window: int = 7,
        corr_pct: float = 20.0,
        vol_pct: float = 33.0,
        rolling_window: int = 80,
        roll_window: int = 5,
        corr_threshold: float = 0.75,
        partial_blind: float = 0.3,
        partial_style: float = 0.5,
    ):
        self._blind = BlindWindowTiming(
            small_file=small_file,
            large_file=large_file,
            signal_window=signal_window,
            corr_pct=corr_pct,
            vol_pct=vol_pct,
            rolling_window=rolling_window,
        )
        self._style = StyleConvergenceTiming(
            small_file=small_file,
            large_file=large_file,
            roll_window=roll_window,
            corr_threshold=corr_threshold,
        )
        self.partial_blind = partial_blind
        self.partial_style = partial_style

    def prepare(self, start_date: str, end_date: str) -> None:
        self._blind.prepare(start_date, end_date)
        self._style.prepare(start_date, end_date)

    def get_position_ratio(self, date_str: str) -> float:
        b = self._blind.get_position_ratio(date_str)
        s = self._style.get_position_ratio(date_str)
        b_invest = b >= 1.0
        s_invest = s >= 1.0
        if b_invest and s_invest:
            return 1.0
        elif b_invest:
            return self.partial_style
        elif s_invest:
            return self.partial_blind
        else:
            return 0.0
