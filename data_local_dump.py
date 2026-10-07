import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from data_api.tushareApi import TushareDataSource
from merge_data import is_main_board   # 主板口径的唯一来源

START_DATE = "20180101"
# 动态取今日：写死日期会让交易日历过期，导致新交易日下载不到（增量更新静默失效）
END_DATE   = datetime.today().strftime('%Y%m%d')

# 多线程配置
ENABLE_MULTITHREADING = True  # 是否启用多线程并行下载
# 需要多次请求的任务（按股票或交易日循环下载）放在并行列表
# namechange 只用于首次全量建库时给历史 ST 标识打底；日常增量更新不再需要
# （ST 判定已改为用 stock_list 的当前名称与本地记录的上一次名称比对）
PARALLEL_TASKS = ['stock_data', 'adj_factor', 'income', 'balancesheet', 'cashflow', 'namechange',
                  'daily_basic_data', 'moneyflow', 'margin_detail', 'top_list']
MAX_WORKERS = 5  # 最大线程数（根据Tushare API频率限制调整）

class DownloadData:
    """
    从 Tushare 下载原始数据到本地，供后续因子计算和回测使用。

    所有路径沿用现有约定（data/raw/...），目录不存在时自动创建。
    重复运行时已存在的文件会被跳过，支持断点续传。
    """

    # 各类数据的本地存储路径
    STOCK_LIST_PATH     = "data/raw/stock_list/stock_list.csv"
    TRADE_CAL_PATH      = "data/raw/trade_cal.csv"
    STOCK_DATA_DIR      = "data/raw/stock_data/"
    ADJ_FACTOR_DIR      = "data/raw/adj_factor/"
    DAILY_BASIC_DIR     = "data/raw/daily_basic_data/"
    MONEYFLOW_DIR       = "data/raw/moneyflow/"
    MARGIN_DETAIL_DIR   = "data/raw/margin_detail/"
    TOP_LIST_DIR        = "data/raw/top_list/"
    INCOME_DIR          = "data/raw/income/"
    BALANCESHEET_DIR    = "data/raw/balancesheet/"
    CASHFLOW_DIR        = "data/raw/cashflow/"
    INDEX_BASIC_DIR     = "data/raw/index_basic/"
    INDEX_DAILY_DIR     = "data/raw/index_daily/"
    NAMECHANGE_DIR      = "data/raw/namechange/"

    # 精选指数日线清单：{ts_code: (名称, 分类)}
    # 原 index_daily() 会遍历 SSE.csv 的全部指数（数千个）逐个下载，绝大多数用不上。
    # 这里只保留择时与回测分析真正需要的三类：
    #   scale —— 规模宽基：判断市场状态、大小盘轮动，同时作为回测的收益基准
    #   style —— 风格：价值/成长轮动
    #   div   —— 红利：防御风格，用于风险偏好判断
    # 注意：中证A500 的正确代码是 000510.SH；项目早期误写成 000510.CSI（该代码不存在）。
    INDEX_SELECTED: dict = {
        # ── 规模宽基（策略选股市值中位约 99 亿，中证1000/2000 是最贴近的基准）──
        '000001.SH':  ('上证指数',  'scale'),
        '000300.SH':  ('沪深300',   'scale'),
        '000905.SH':  ('中证500',   'scale'),
        '000852.SH':  ('中证1000',  'scale'),
        '932000.CSI': ('中证2000',  'scale'),
        '000985.CSI': ('中证全指',  'scale'),
        '000510.SH':  ('中证A500',  'scale'),
        '000016.SH':  ('上证50',    'scale'),
        '399006.SZ':  ('创业板指',  'scale'),
        '000688.SH':  ('科创50',    'scale'),
        # ── 风格 ──
        '399370.SZ':  ('国证成长',  'style'),
        '399371.SZ':  ('国证价值',  'style'),
        '399376.SZ':  ('小盘成长',  'style'),
        '399377.SZ':  ('小盘价值',  'style'),
        # ── 红利 ──
        '000922.CSI': ('中证红利',  'div'),
        '000015.SH':  ('红利指数',  'div'),
    }

    # True = 只下载/更新主板股票（SH 6xxxxx 非688、SZ 0xxxxx 非300/301）
    MAIN_BOARD_ONLY: bool = True

    def __init__(self, api=TushareDataSource):
        self.api = api()

    # ------------------------------------------------------------------ #
    #  工具方法                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _call_with_retry(fn, *args, max_retries: int = 5, base_sleep: float = 30.0, **kwargs):
        """调用 fn(*args, **kwargs)，遇到频率超限时指数退避重试。"""
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(str(e))
                msg = str(e)
                if '频率超限' in msg or 'exceed' in msg.lower():
                    wait = base_sleep * (2 ** attempt)
                    print(f"\n  [限速] 等待 {wait:.0f}s 后重试（第 {attempt + 1}/{max_retries} 次）...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"超过最大重试次数 {max_retries}，放弃。")

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """路径不存在时递归创建，存在时不报错。"""
        os.makedirs(path, exist_ok=True)

    def _get_trading_dates(self, start: str = START_DATE, end: str = END_DATE) -> list[str]:
        """
        从本地交易日历读取指定区间内的交易日列表（仅 is_open=1 的日期）。
        调用前需确保 TRADE_CAL_PATH 已下载。
        """
        df = pd.read_csv(self.TRADE_CAL_PATH, dtype={'cal_date': str, 'is_open': int})
        mask = (df['cal_date'] >= start) & (df['cal_date'] <= end) & (df['is_open'] == 1)
        return sorted(df.loc[mask, 'cal_date'].tolist())

    def _get_stock_list(self) -> list[str]:
        """从本地股票列表文件读取 ts_code。MAIN_BOARD_ONLY=True 时只返回主板股票。"""
        df = pd.read_csv(self.STOCK_LIST_PATH)
        if self.MAIN_BOARD_ONLY:
            df = df[df['ts_code'].apply(self._is_main_board)]
        return df['ts_code'].tolist()

    @staticmethod
    def _is_main_board(ts_code: str) -> bool:
        """主板判断：规则统一走 merge_data.is_main_board，避免两处口径漂移。"""
        return is_main_board(ts_code)

    # ------------------------------------------------------------------ #
    #  基础元数据                                                           #
    # ------------------------------------------------------------------ #

    def trade_cal(self, start: str = "20100101", end: str = None) -> None:
        """
        下载交易日历到本地。覆盖写入（每次获取全量）。

        默认拉到今天之后 30 天：只拉到今天的话，遇到国庆/春节这种长假，
        "下一交易日"就会落在日历之外，run_signal 的 T+1 判断会失效。
        """
        end = end or (datetime.today() + timedelta(days=30)).strftime('%Y%m%d')
        self._ensure_dir(os.path.dirname(self.TRADE_CAL_PATH))
        print(f"下载交易日历 {start} ~ {end} ...")
        df = self.api.get_trade_calender(startdate=start, enddate=end)
        df.to_csv(self.TRADE_CAL_PATH, index=False)
        print(f"  已保存 → {self.TRADE_CAL_PATH}")

    def stock_list(self) -> None:
        """下载当前上市 A 股列表（主板）。覆盖写入。"""
        self._ensure_dir(os.path.dirname(self.STOCK_LIST_PATH))
        print("下载股票列表 ...")
        df = self.api.get_stock_list()
        df.to_csv(self.STOCK_LIST_PATH, index=False)
        print(f"  已保存 → {self.STOCK_LIST_PATH}（{len(df)} 只）")

    def index_basic(self) -> None:
        """下载各市场的指数基础信息（SW / CSI / SSE / SZSE / MSCI）。"""
        self._ensure_dir(self.INDEX_BASIC_DIR)
        for market in ["SW", "CSI", "SSE", "SZSE", "MSCI"]:
            path = os.path.join(self.INDEX_BASIC_DIR, f"{market}.csv")
            print(f"下载指数基础信息 market={market} ...")
            df = self.api.get_index_basic(market=market)
            df.to_csv(path, index=False)
            time.sleep(0.4)

    # ------------------------------------------------------------------ #
    #  按日期下载（每个交易日一个文件）                                     #
    # ------------------------------------------------------------------ #

    def daily_basic_data(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每个交易日的股票基本面快照（PE/PB/市值等）。"""
        self._ensure_dir(self.DAILY_BASIC_DIR)
        dates = self._get_trading_dates(start, end)
        for date in dates:
            path = os.path.join(self.DAILY_BASIC_DIR, f"{date}.csv")
            if os.path.exists(path):
                continue   # 已下载，跳过
            print(f"  daily_basic {date}")
            df = self.api.get_daily_basic_data(date=date)
            if df is None or df.empty:
                continue
            df.to_csv(path, index=False)
            time.sleep(0.4)

    def moneyflow(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每个交易日的大中小单净流入数据。"""
        self._ensure_dir(self.MONEYFLOW_DIR)
        dates = self._get_trading_dates(start, end)
        for date in dates:
            path = os.path.join(self.MONEYFLOW_DIR, f"{date}.csv")
            if os.path.exists(path):
                continue
            print(f"  moneyflow {date}")
            df = self.api.get_moneyflow(date=date)
            if df is None or df.empty:
                continue
            df.to_csv(path, index=False)
            time.sleep(0.4)

    def margin_detail(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每个交易日的融资融券明细。"""
        self._ensure_dir(self.MARGIN_DETAIL_DIR)
        dates = self._get_trading_dates(start, end)
        for date in dates:
            path = os.path.join(self.MARGIN_DETAIL_DIR, f"{date}.csv")
            if os.path.exists(path):
                continue
            print(f"  margin_detail {date}")
            df = self.api.get_margin_detail(date=date)
            if df is None or df.empty:
                continue
            df.to_csv(path, index=False)
            time.sleep(0.4)

    def top_list(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每个交易日的龙虎榜数据。"""
        self._ensure_dir(self.TOP_LIST_DIR)
        dates = self._get_trading_dates(start, end)
        for date in dates:
            path = os.path.join(self.TOP_LIST_DIR, f"{date}.csv")
            if os.path.exists(path):
                continue
            print(f"  top_list {date}")
            df = self.api.get_top_list(date=date)
            if df is None or df.empty:
                continue
            df.to_csv(path, index=False)
            time.sleep(0.4)

    # ------------------------------------------------------------------ #
    #  按股票下载（每只股票一个文件）                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _incremental_start(path: str, date_col: str) -> Optional[str]:
        """
        读取已有文件中 date_col 列的最大值，返回其后一天作为增量起点。
        文件不存在或读取失败时返回 None（表示全量下载）。
        """
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, usecols=[date_col], dtype={date_col: str})
            latest = df[date_col].dropna().max()
            if not latest:
                return None
            # 返回最新日期的次日（字符串 YYYYMMDD），API 会自动对齐到下一个交易日
            dt = pd.to_datetime(latest, format='%Y%m%d') + pd.Timedelta(days=1)
            return dt.strftime('%Y%m%d')
        except Exception:
            return None

    @staticmethod
    def _append_and_save(path: str, existing_df: Optional[pd.DataFrame],
                         new_df: pd.DataFrame, dedup_cols: list[str]) -> None:
        """合并新旧数据，按 dedup_cols 去重后写回文件。"""
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined = combined.drop_duplicates(subset=dedup_cols)
        combined.to_csv(path, index=False)

    def stock_data(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每只股票的不复权日线行情，支持增量更新。"""
        self._ensure_dir(self.STOCK_DATA_DIR)
        for ts_code in self._get_stock_list():
            path = os.path.join(self.STOCK_DATA_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'trade_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue   # 已是最新，无需更新
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'trade_date': str})
            df = self._call_with_retry(self.api.get_stock_data, code=ts_code, startdate=fetch_start, enddate=end, adj=None)
            time.sleep(0.4)
            if df is None or df.empty:
                continue
            print(f"  stock_data {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'trade_date'])

    def adj_factor(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每只股票的复权因子，支持增量更新。"""
        self._ensure_dir(self.ADJ_FACTOR_DIR)
        for ts_code in self._get_stock_list():
            path = os.path.join(self.ADJ_FACTOR_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'trade_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue   # 已是最新，无需更新
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'trade_date': str})
            df = self._call_with_retry(self.api.get_adj_factor, code=ts_code, startdate=fetch_start, enddate=end)
            time.sleep(0.4)
            if df is None or df.empty:
                continue
            print(f"  adj_factor {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'trade_date'])

    def income(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每只股票的利润表，支持增量更新。"""
        self._ensure_dir(self.INCOME_DIR)
        for ts_code in self._get_stock_list():
            path = os.path.join(self.INCOME_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'ann_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'ann_date': str})
            df = self._call_with_retry(self.api.get_income, code=ts_code, startdate=fetch_start, enddate=end)
            time.sleep(0.4)
            if df is None or df.empty:
                continue
            print(f"  income {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'ann_date', 'end_date'])
            

    def balancesheet(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每只股票的资产负债表，支持增量更新。"""
        self._ensure_dir(self.BALANCESHEET_DIR)
        for ts_code in self._get_stock_list():
            path = os.path.join(self.BALANCESHEET_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'ann_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'ann_date': str})
            df = self._call_with_retry(self.api.get_balancesheet, code=ts_code, startdate=fetch_start, enddate=end)
            time.sleep(0.4)
            if df is None or df.empty:
                continue
            print(f"  balancesheet {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'ann_date', 'end_date'])

    def cashflow(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载每只股票的现金流量表，支持增量更新。"""
        self._ensure_dir(self.CASHFLOW_DIR)
        for ts_code in self._get_stock_list():
            path = os.path.join(self.CASHFLOW_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'ann_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'ann_date': str})
            df = self._call_with_retry(self.api.get_cashflow, code=ts_code, startdate=fetch_start, enddate=end)
            time.sleep(0.4)
            if df is None or df.empty:
                continue
            print(f"  cashflow {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'ann_date', 'end_date'])

    # ------------------------------------------------------------------ #
    #  名称变更（ST股票识别）                                               #
    # ------------------------------------------------------------------ #

    def namechange(self, start: str = "20100101", end: str = END_DATE) -> None:
        """
        下载股票名称变更记录。

        注意：日常增量更新不再需要它 —— ST 标识改为用 stock_list 的当前名称
        与本地记录的上一次名称比对（见 merge_data.build_st_flag），
        不再调用 get_namechange 接口。这里保留仅供首次全量建库时给历史
        ST 标识打底（data/raw/namechange 有数据时 build_st_flag 会自动用一次）。
        """
        self._ensure_dir(self.NAMECHANGE_DIR)
        for ts_code in self._get_stock_list():
            path = os.path.join(self.NAMECHANGE_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'start_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'start_date': str})
            df = self._call_with_retry(self.api.get_namechange, code=ts_code, startdate=fetch_start, enddate=end)
            time.sleep(0.5)
            if df is None or df.empty:
                continue
            print(f"  namechange {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'name', 'start_date'])

    # ------------------------------------------------------------------ #
    #  多线程并行执行                                                       #
    # ------------------------------------------------------------------ #

    # 任务名称 -> 方法映射
    TASK_MAP = {
        'daily_basic_data': 'daily_basic_data',
        'moneyflow': 'moneyflow',
        'margin_detail': 'margin_detail',
        'top_list': 'top_list',
        'stock_data': 'stock_data',
        'adj_factor': 'adj_factor',
        'income': 'income',
        'balancesheet': 'balancesheet',
        'cashflow': 'cashflow',
        #'namechange': 'namechange',
        'index_daily': 'index_daily',
    }

    def run_parallel(self, tasks: list[str] = None, start: str = START_DATE, end: str = END_DATE) -> None:
        """
        多线程并行执行指定的下载任务。

        Parameters
        ----------
        tasks : list[str]
            要并行执行的任务名称列表，如 ['stock_data', 'adj_factor', 'income']。
            为 None 时使用全局配置 PARALLEL_TASKS。
        start : str
            起始日期
        end : str
            结束日期
        """
        if tasks is None:
            tasks = PARALLEL_TASKS

        # 验证任务名称
        invalid = [t for t in tasks if t not in self.TASK_MAP]
        if invalid:
            raise ValueError(f"无效的任务名称: {invalid}，可选: {list(self.TASK_MAP.keys())}")

        print(f"\n{'='*60}")
        print(f"多线程并行下载: {tasks}")
        print(f"线程数: {MAX_WORKERS}")
        print(f"日期范围: {start} ~ {end}")
        print(f"{'='*60}\n")

        def _run_task(task_name: str):
            method = getattr(self, self.TASK_MAP[task_name])
            print(f"[{task_name}] 开始下载...")
            t0 = time.time()
            method(start=start, end=end)
            elapsed = time.time() - t0
            print(f"[{task_name}] 完成，耗时 {elapsed:.1f}s")
            return task_name, elapsed

        results = {}
        errors = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    name, elapsed = future.result()
                    results[name] = elapsed
                except Exception as e:
                    print(f"\n[{task_name}] 失败: {e}")
                    errors.append((task_name, str(e)))

        # 汇总
        print(f"\n{'='*60}")
        print("下载汇总:")
        for name, elapsed in sorted(results.items()):
            print(f"  ✓ {name}: {elapsed:.1f}s")
        if errors:
            print(f"\n失败任务 ({len(errors)}):")
            for name, err in errors:
                print(f"  ✗ {name}: {err}")
        print(f"{'='*60}\n")

    def index_daily(self, start: str = START_DATE, end: str = END_DATE) -> None:
        """下载 SSE 全部指数的日线行情，支持增量更新。"""
        self._ensure_dir(self.INDEX_DAILY_DIR)
        index_list_path = os.path.join(self.INDEX_BASIC_DIR, "SSE.csv")
        code_list = pd.read_csv(index_list_path)['ts_code'].tolist()
        for ts_code in code_list:
            path = os.path.join(self.INDEX_DAILY_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'trade_date')
            if inc_start is None:
                fetch_start = start
                existing = None
            elif inc_start > end:
                continue
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'trade_date': str})
            df = self._call_with_retry(self.api.get_index_daily, ts_code=ts_code, startdate=fetch_start, enddate=end)
            time.sleep(0.4)
            if df is None or df.empty:
                continue
            print(f"  index_daily {ts_code}  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'trade_date'])

    def index_daily_selected(
        self, start: str = START_DATE, end: str = END_DATE,
        categories: Optional[list] = None, codes: Optional[list] = None,
    ) -> None:
        """
        只下载 INDEX_SELECTED 中的精选指数日线，支持增量更新。

        与 index_daily() 的区别：后者遍历 SSE.csv 的全部指数（数千个，绝大多数用不上），
        这里只下择时与回测分析需要的十几个，且覆盖沪深两市与中证系列。

        Parameters
        ----------
        categories : list[str] | None   只下指定分类：'scale' / 'style' / 'div'
        codes      : list[str] | None   只下指定代码，优先级高于 categories
        """
        self._ensure_dir(self.INDEX_DAILY_DIR)

        if codes:
            targets = [(c, self.INDEX_SELECTED.get(c, (c, '-'))) for c in codes]
        else:
            targets = [(c, v) for c, v in self.INDEX_SELECTED.items()
                       if categories is None or v[1] in categories]

        for ts_code, (name, cat) in targets:
            path = os.path.join(self.INDEX_DAILY_DIR, f"{ts_code}.csv")
            inc_start = self._incremental_start(path, 'trade_date')
            if inc_start is None:
                fetch_start, existing = start, None
            elif inc_start > end:
                continue                      # 已是最新的，跳过
            else:
                fetch_start = inc_start
                existing = pd.read_csv(path, dtype={'trade_date': str})

            df = self._call_with_retry(
                self.api.get_index_daily, ts_code=ts_code,
                startdate=fetch_start, enddate=end)
            time.sleep(0.4)
            if df is None or df.empty:
                print(f"  index_daily {ts_code} {name}  无数据")
                continue
            print(f"  index_daily {ts_code} {name}[{cat}]  {fetch_start}~{end}  +{len(df)}行")
            self._append_and_save(path, existing, df, dedup_cols=['ts_code', 'trade_date'])


if __name__ == '__main__':
    d = DownloadData()  # MAIN_BOARD_ONLY=True，默认只处理主板

    # ① 基础元数据（其他方法依赖这三个，必须先跑）
    # d.trade_cal()       # 交易日历（全量，从2010年起）
    # d.stock_list()      # 股票列表（含全板块，过滤由 _get_stock_list 控制）
    # d.index_basic()     # 指数基础信息

    # ② 需要多次请求的任务（按股票或交易日循环下载）放在并行列表中
    if ENABLE_MULTITHREADING and PARALLEL_TASKS:
        d.run_parallel()
    else:
        # 单线程顺序下载
        d.stock_data()      # 不复权日线行情
        d.adj_factor()      # 复权因子（用于计算后复权价格）
        d.income()
        d.balancesheet()
        d.cashflow()
        # d.namechange()      # 股票名称变更（用于识别ST股）
        d.daily_basic_data()
        d.moneyflow()
        d.margin_detail()
        d.top_list()

    # ③ 指数日线（依赖 index_basic 已下载，暂时不做）
    # d.index_daily()
