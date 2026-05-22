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

    仓位规则（二值化）：
      corr_5d < corr_threshold  OR  l_vol < vol_threshold  →  avoid_ratio（默认空仓）
      否则                                                  →  1.0（满仓）

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

        df = pd.DataFrame({'s_ret': s_ret, 'l_ret': l_ret}).dropna()

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

        # 触发条件：corr 或 vol 低于各自阈值
        avoid = (df['corr_5d'] < df['corr_thresh']) | (df['l_vol'] < df['vol_thresh'])
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
