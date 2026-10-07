"""
数据管理模块 - 基于 DuckDB + Parquet 的统一数据存储

存储布局
--------
data/
├── raw/                                   # 原始 CSV（不变）
├── series/<ts_code>.parquet               # 时序层：逐股票，供因子计算（滚动窗口沿时间轴）
└── market/trade_date=YYYYMMDD/part-*.parquet   # 截面层：按交易日分区，供标准化 / 回测

为什么保留两层
--------------
- 因子计算是「逐股票时序」访问（rolling / ewm 必须沿时间轴），需要数据按股票聚拢 → series 层。
- 建模与回测是「逐交易日截面」访问，需要数据按日期聚拢 → market 层。
两者互为转置，一套物理布局无法同时高效服务两种访问模式，因此保留两层：
series 是含因子的时序真源，market 是由其派生、供回测消费的截面层。

关键的分区设计
--------------
分区键就是 trade_date。DuckDB 只对 **hive 分区列** 做文件裁剪，
因此所有查询必须把过滤条件写在分区列上（本模块内部已保证），
否则一次 40 天窗口查询就要打开全部两千多个分区文件。
实测：命中分区列 0.23s / 未命中 1.93s（300 天样本），差距随数据量线性放大。

文件内不再冗余存储 trade_date 列（由 hive 分区提供），读取时由 DataStore
统一补回 YYYYMMDD 字符串形式，与上层（strategy / backtest）保持类型一致。
"""

import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------- #
#  常量
# ---------------------------------------------------------------------- #

PART_COL = 'trade_date'          # hive 分区列名，同时也是业务上的交易日列名
MARKET_DIRNAME = 'market'        # 截面层目录（旧版叫 market.parquet，易被误认为文件）
SERIES_DIRNAME = 'series'        # 时序层目录

_DATE_RE = re.compile(r'^\d{8}$')
_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def normalize_date(value) -> str:
    """把 '2024-01-02' / 20240102 / '20240102' 统一成 '20240102'。"""
    s = str(value).strip().replace('-', '').replace('/', '')
    if not _DATE_RE.match(s):
        raise ValueError(f'非法交易日: {value!r}，应为 YYYYMMDD')
    return s


def _q(identifier: str) -> str:
    """列名加引号，避免与 SQL 关键字冲突。"""
    if not _IDENT_RE.match(str(identifier)):
        raise ValueError(f'非法列名: {identifier!r}')
    return f'"{identifier}"'


class DataStore:
    """
    统一数据管理模块（截面层 market + 时序层 series）。

    对外暴露的最小接口：
        write_section(date, df)          写入/覆盖某交易日截面
        read_section(date, cols=None)    读取某交易日截面
        read_window(start, end, cols)    读取日期区间（命中分区裁剪）
        read_stock(code, start, end)     读取单只股票历史
        get_all_dates()                  所有已落盘的交易日
        build_sections_from_series()     从 series 流式派生截面层
    """

    def __init__(self, base_path: str = 'data', market_dir: str = MARKET_DIRNAME):
        self.base_path = Path(base_path)
        self.market_path = self.base_path / market_dir
        self.series_path = self.base_path / SERIES_DIRNAME
        self.market_path.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    # ------------------------------------------------------------------ #
    #  连接
    # ------------------------------------------------------------------ #

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ #
    #  路径 / SQL 片段
    # ------------------------------------------------------------------ #

    def _partition_dir(self, trade_date: str) -> Path:
        return self.market_path / f'{PART_COL}={normalize_date(trade_date)}'

    def _glob_pattern(self) -> str:
        # DuckDB 的 glob 用正斜杠最稳，Windows 下需转换
        return str(self.market_path / f'{PART_COL}=*' / '*.parquet').replace('\\', '/')

    def _scan(self) -> str:
        """生成 read_parquet(...) 片段：开启 hive 分区、固定分区类型为 BIGINT、允许列并集。"""
        return (
            f"read_parquet('{self._glob_pattern()}', "
            f"hive_partitioning=true, "
            f"hive_types={{'{PART_COL}':'BIGINT'}}, "
            f"union_by_name=true)"
        )

    @staticmethod
    def _select_clause(columns: Optional[Sequence[str]]) -> str:
        if not columns:
            return '*'
        cols = [_q(PART_COL)] + [_q(c) for c in columns if c != PART_COL]
        return ', '.join(cols)

    # ------------------------------------------------------------------ #
    #  元数据
    # ------------------------------------------------------------------ #

    def get_all_dates(self) -> List[str]:
        """已落盘的所有交易日（升序，YYYYMMDD 字符串）。"""
        if not self.market_path.exists():
            return []
        out = []
        for d in self.market_path.iterdir():
            if not d.is_dir() or '=' not in d.name:
                continue
            key, _, val = d.name.partition('=')
            if key == PART_COL and _DATE_RE.match(val):
                out.append(val)
        return sorted(out)

    def get_latest_date(self) -> Optional[str]:
        dates = self.get_all_dates()
        return dates[-1] if dates else None

    def has_date(self, trade_date: str) -> bool:
        d = self._partition_dir(trade_date)
        return d.is_dir() and any(d.glob('*.parquet'))

    def table_exists(self) -> bool:
        return len(self.get_all_dates()) > 0

    def get_all_stocks(self, trade_date: str = None) -> List[str]:
        """股票代码列表。指定日期时只读该分区，否则走全表 DISTINCT。"""
        if trade_date:
            df = self.read_section(trade_date, columns=['ts_code'])
            if df is None or 'ts_code' not in df.columns:
                return []
            return df['ts_code'].dropna().unique().tolist()
        if not self.table_exists():
            return []
        sql = f'SELECT DISTINCT ts_code FROM {self._scan()} ORDER BY ts_code'
        try:
            return self.conn.execute(sql).df()['ts_code'].tolist()
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  写入
    # ------------------------------------------------------------------ #

    def _write_part(self, trade_date: str, df: pd.DataFrame, part: Optional[int] = None) -> Path:
        """写入分区内的一个 part 文件。df 中的 trade_date 列会被剔除（由 hive 提供）。"""
        if df is None or df.empty:
            raise ValueError('空 DataFrame，拒绝写入')
        d = normalize_date(trade_date)
        pdir = self._partition_dir(d)
        pdir.mkdir(parents=True, exist_ok=True)
        payload = df.drop(columns=[PART_COL], errors='ignore')
        name = 'part-0.parquet' if part is None else f'part-{part}.parquet'
        out = pdir / name
        payload.to_parquet(out, index=False, engine='pyarrow', compression='snappy')
        return out

    def reset(self) -> None:
        """
        清空整个截面层。

        用目录 rename 而不是删除：全量重算要丢弃六千多个 part 文件，
        批量删除不安全也不好回滚，改名保留后确认无误再手动清理即可。
        """
        if self.market_path.exists() and any(self.market_path.iterdir()):
            self.market_path.rename(self.base_path / f'{self.market_path.name}_old_{int(time.time())}')
        self.market_path.mkdir(parents=True, exist_ok=True)

    def write_section(self, trade_date: str, df: pd.DataFrame) -> None:
        """
        覆盖写入某交易日截面。会先清空该分区目录下的旧 part 文件，
        保证每个交易日最终只有一个 part-0.parquet。
        """
        if df is None or df.empty:
            return
        pdir = self._partition_dir(trade_date)
        if pdir.exists():
            for f in pdir.glob('*.parquet'):
                try:
                    f.unlink()
                except OSError:
                    pass
        else:
            pdir.mkdir(parents=True, exist_ok=True)
        self._write_part(trade_date, df)

    # 兼容旧接口名
    write_day_data = write_section

    def append_day_data(self, trade_date: str, df: pd.DataFrame) -> None:
        """追加数据到某交易日（按 ts_code 去重，新数据优先）。"""
        if df is None or df.empty:
            return
        existing = self.read_section(trade_date)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['ts_code'], keep='last')
        self.write_section(trade_date, df)

    # ------------------------------------------------------------------ #
    #  读取
    # ------------------------------------------------------------------ #

    @staticmethod
    def _attach_date(df: pd.DataFrame, trade_date: str = None) -> pd.DataFrame:
        """把 hive 分区列（BIGINT）还原成 YYYYMMDD 字符串，并放到首列。"""
        if df is None or df.empty:
            return df
        if PART_COL in df.columns:
            df[PART_COL] = df[PART_COL].astype('int64').astype(str)
            col = df.pop(PART_COL)
            df.insert(0, PART_COL, col)
        elif trade_date is not None:
            df.insert(0, PART_COL, normalize_date(trade_date))
        return df

    def read_section(self, trade_date: str,
                     columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
        """
        读取单个交易日截面；不存在返回 None。

        这里直接用 pyarrow 读该分区目录，而不是走 DuckDB：
        DuckDB 的 read_parquet(..., union_by_name=true) 需要先把 **全部** 分区文件的
        schema 并一遍才能确定输出列（6736 个文件时约 6 秒），既慢又会把少数已标准化
        日期才有的 _standard/_neutral 列混进所有日期的 schema，
        让下游误以为该日期已标准化过。
        分区内文件由同一次构建写入，schema 天然一致，直读目录即可。
        """
        d = normalize_date(trade_date)
        pdir = self._partition_dir(d)
        if not pdir.exists():
            return None

        try:
            files = sorted(pdir.glob('*.parquet'))
            if columns:
                cols = [c for c in columns if c != PART_COL]
                frames = [pq.read_table(f, columns=cols).to_pandas() for f in files]
            else:
                frames = [pq.read_table(f).to_pandas() for f in files]
            # 逐文件读再 concat：分区内文件 schema 可能有细微差异（不同批次写入），
            # 交给 pandas 做列并集比要求 Arrow schema 严格一致更稳
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        except Exception as e:
            print(f'[DataStore] 读取 {d} 失败: {e}')
            return None

        return self._attach_date(df, d)

    # 兼容旧接口名
    read_day_data = read_section

    def read_window(self, start_date: str, end_date: str,
                    columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        读取日期区间内的所有截面（闭区间）。

        过滤条件写在分区列上，触发 DuckDB 的 hive 裁剪，
        只打开区间内的分区文件。
        """
        s, e = normalize_date(start_date), normalize_date(end_date)
        if s > e:
            s, e = e, s
        if not self.table_exists():
            return pd.DataFrame()
        sql = (f'SELECT {self._select_clause(columns)} FROM {self._scan()} '
               f'WHERE {_q(PART_COL)} BETWEEN {int(s)} AND {int(e)} '
               f'ORDER BY {_q(PART_COL)}')
        try:
            df = self.conn.execute(sql).df()
        except Exception as ex:
            print(f'[DataStore] 区间查询失败: {ex}')
            return pd.DataFrame()
        return self._attach_date(df)

    # 兼容旧接口名
    read_date_range = read_window

    def read_stock(self, ts_code: str, start_date: str = None,
                   end_date: str = None,
                   columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """读取单只股票的历史时序（会扫描全部日期分区，慎用）。"""
        if not self.table_exists():
            return pd.DataFrame()
        sql = (f'SELECT {self._select_clause(columns)} FROM {self._scan()} '
               f"WHERE ts_code = '{ts_code}'")
        if start_date:
            sql += f' AND {_q(PART_COL)} >= {int(normalize_date(start_date))}'
        if end_date:
            sql += f' AND {_q(PART_COL)} <= {int(normalize_date(end_date))}'
        sql += f' ORDER BY {_q(PART_COL)}'
        try:
            df = self.conn.execute(sql).df()
        except Exception as ex:
            print(f'[DataStore] 个股查询失败: {ex}')
            return pd.DataFrame()
        return self._attach_date(df)

    # 兼容旧接口名
    read_stock_data = read_stock

    def iter_sections(self, dates: Optional[Sequence[str]] = None,
                      columns: Optional[Sequence[str]] = None) -> Iterator[tuple]:
        """逐个交易日迭代 (date, DataFrame)，内存只保留一天。"""
        target = list(dates) if dates else self.get_all_dates()
        for d in target:
            df = self.read_section(d, columns=columns)
            if df is not None and not df.empty:
                yield d, df

    # ------------------------------------------------------------------ #
    #  从 series 流式派生截面层
    # ------------------------------------------------------------------ #

    def build_sections_from_series(self, series_dir: str = None,
                                   chunk_size: int = 200,
                                   main_board_only: bool = True,
                                   stock_list_file: str = None,
                                   only_dates: Optional[Sequence[str]] = None,
                                   overwrite: bool = True,
                                   verbose: bool = True) -> int:
        """
        从时序层（series）派生截面层（market）。

        按股票分块读取 → 每块按交易日切分 → 以 part-<块序号>.parquet 追加写入对应分区。
        内存占用被限制在「一块股票 × 全部交易日」，避免一次性 concat 全量数据导致 OOM
        （全量约 677 万行 × 数百列，float64 下十几 GB）。

        Returns
        -------
        int  写入的截面行数
        """
        sdir = Path(series_dir) if series_dir else self.series_path
        if not sdir.exists():
            print(f'[DataStore] series 目录不存在: {sdir}')
            return 0

        files = sorted(sdir.glob('[0-9]*.parquet')) or sorted(sdir.glob('[0-9]*.csv'))
        files = [f for f in files if not f.name.startswith('_')]
        if not files:
            print(f'[DataStore] {sdir} 下没有时序文件')
            return 0

        keep_codes = None
        if main_board_only:
            sl_path = stock_list_file or str(self.base_path / 'raw' / 'stock_list' / 'stock_list.csv')
            if os.path.exists(sl_path):
                keep_codes = set(pd.read_csv(sl_path, usecols=['ts_code'])['ts_code'])
            else:
                print(f'[DataStore] 股票列表不存在，跳过主板过滤: {sl_path}')

        date_filter = set(only_dates) if only_dates else None

        if overwrite:
            if date_filter is None:
                self.reset()
            else:
                for d in date_filter:
                    pdir = self._partition_dir(d)
                    if pdir.exists():
                        pdir.rename(self.market_path / f'_old_{d}_{int(time.time())}')

        total_rows = 0
        n_chunks = math.ceil(len(files) / chunk_size)
        if verbose:
            print(f'[DataStore] 流式构建截面：{len(files)} 个时序文件，分 {n_chunks} 块'
                  + (f'，限定 {len(date_filter)} 个交易日' if date_filter else ''))

        for ci in range(n_chunks):
            chunk = files[ci * chunk_size: (ci + 1) * chunk_size]
            frames = []
            for f in chunk:
                try:
                    df = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f)
                except Exception as ex:
                    print(f'\n[DataStore] 读取失败 {f.name}: {ex}')
                    continue
                if PART_COL not in df.columns:
                    continue
                df[PART_COL] = df[PART_COL].astype(str)
                if date_filter is not None:
                    df = df[df[PART_COL].isin(date_filter)]
                if keep_codes is not None and 'ts_code' in df.columns:
                    df = df[df['ts_code'].isin(keep_codes)]
                if not df.empty:
                    frames.append(df)

            if not frames:
                continue

            big = pd.concat(frames, ignore_index=True)
            del frames
            for d, g in big.groupby(PART_COL, sort=False):
                g = g.sort_values('ts_code') if 'ts_code' in g.columns else g
                g = g.drop_duplicates(subset='ts_code', keep='first', ignore_index=True)
                self._write_part(d, g, part=ci)
                total_rows += len(g)
            del big

            if verbose:
                print(f'\r  构建进度: {ci + 1}/{n_chunks} 块', end='', flush=True)

        if verbose:
            dates = self.get_all_dates()
            print(f'\n[DataStore] 截面构建完成：{len(dates)} 个交易日，共 {total_rows} 行')
        return total_rows

    # ------------------------------------------------------------------ #
    #  维护
    # ------------------------------------------------------------------ #

    def compact_date(self, trade_date: str) -> None:
        """把某交易日分区内的多个 part 文件合并成单个 part-0.parquet。"""
        df = self.read_section(trade_date)
        if df is not None and not df.empty:
            self.write_section(trade_date, df)

    def drop_date(self, trade_date: str) -> None:
        pdir = self._partition_dir(trade_date)
        if pdir.exists():
            pdir.rename(self.market_path / f'_old_{trade_date}_{int(time.time())}')

    def drop_all(self) -> None:
        self.reset()

    def stats(self) -> dict:
        dates = self.get_all_dates()
        n_files = sum(len(list(self._partition_dir(d).glob('*.parquet'))) for d in dates)
        size = sum(f.stat().st_size
                   for d in dates for f in self._partition_dir(d).glob('*.parquet'))
        return {'dates': len(dates), 'files': n_files,
                'size_mb': round(size / 1024 / 1024, 1),
                'latest': dates[-1] if dates else None}


def swap_market_dir(base_path: str, new_dir: str, target: str = MARKET_DIRNAME) -> str:
    """
    用 new_dir 整体替换 target（两次 rename，不做删除）。

    Returns
    -------
    str  被换下来的旧目录路径
    """
    base = Path(base_path)
    target_path, new_path = base / target, base / new_dir
    old_path = base / f'{target}_old_{int(time.time())}'
    if target_path.exists():
        target_path.rename(old_path)
    new_path.rename(target_path)
    return str(old_path)


# ---------------------------------------------------------------------- #
#  便捷函数
# ---------------------------------------------------------------------- #

def get_data_store(base_path: str = 'data') -> DataStore:
    return DataStore(base_path)


def read_section(date: str, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    """兼容旧接口的便捷读法。"""
    with DataStore() as store:
        return store.read_section(date, columns=columns)


def write_section(date: str, df: pd.DataFrame) -> None:
    """兼容旧接口的便捷写法。"""
    with DataStore() as store:
        store.write_section(date, df)


if __name__ == '__main__':
    store = DataStore()
    demo = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        PART_COL: ['20240101', '20240101'],
        'close': [10.5, 20.3],
        'vol': [1000000, 2000000],
    })
    store.write_section('20240101', demo)
    print(store.read_section('20240101'))
    print('日期:', store.get_all_dates())
    print('统计:', store.stats())
    store.drop_all()
    store.close()
