from abc import ABC, abstractmethod
import os
import pandas as pd
from datetime import datetime
from typing import Optional


class BaseStrategy(ABC):
    """
    截面选股策略抽象基类。

    子类只需实现 fit() 和 generate_signals()，引擎负责调用时序和资金分配。
    设计约束：
      - fit() 和 generate_signals() 均只能使用 date_str 当日及之前的数据，
        严禁引用未来信息（point-in-time 原则）。
      - generate_signals() 不依赖 label 列，使训练（fit）和推理（generate_signals）
        在实盘环境中可以分离运行。
    """

    @abstractmethod
    def fit(self, date_str: str) -> bool:
        """
        使用 date_str 当日及之前的历史数据训练或更新模型。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期

        Returns
        -------
        bool  True 表示训练成功，False 表示数据不足或该日非交易日
        """
        pass

    @abstractmethod
    def generate_signals(self, date_str: str) -> Optional[pd.DataFrame]:
        """
        用当前已训练的模型对 date_str 当日全量股票打分。

        返回的 DataFrame 至少包含两列：
          ts_code : str    股票代码
          score   : float  得分，越高表示越值得买入

        按 score 降序排列，引擎取 top_n 行建仓。
        返回 None 表示当日无法生成信号（模型未训练、数据缺失等）。
        注意：此方法不得使用 label 列，以保证实盘可用性。

        Parameters
        ----------
        date_str : str  YYYYMMDD 格式的日期
        """
        pass

    def reset(self) -> None:
        """
        重置策略内部状态（如已训练的权重、缓存等）。
        引擎在每次独立回测开始前调用，防止多次 run() 之间状态污染。
        子类若有持久状态（模型权重等），需覆盖此方法清空。
        """
        pass

    def simple_backtest(self, start_date: str, end_date: str) -> dict:
        """
        纯信号质量评估（不涉及资金和交易），计算 Rank IC、分层多空收益等指标。
        子类可选覆盖以提供具体实现，默认抛出 NotImplementedError。

        Parameters
        ----------
        start_date : str  回测开始日期 YYYYMMDD
        end_date   : str  回测结束日期 YYYYMMDD

        Returns
        -------
        dict  评估结果，具体结构由子类定义
        """
        raise NotImplementedError

    def report_dump(self, result: dict, output_dir: str = None, label_period: int = 1) -> str:
        """
        将 simple_backtest() 的返回结果持久化到本地目录。

        默认在 strategy/backtest_results/<start>_<end>_<timestamp>/ 下写入：
          ic_series.csv          — 每日 Rank IC
          long_short_returns.csv — 多空分层日收益（全量，含重叠）
          factor_weights.csv     — 每日因子权重（列 = 因子名）
          summary.txt            — 统计摘要（多空收益采用非重叠采样）

        Parameters
        ----------
        result       : dict  simple_backtest() 返回值，期望包含
                             ic_df / ls_df / weights_df 三个 DataFrame
        output_dir   : str   指定输出目录；None 时自动生成带时间戳的子目录
        label_period : int   label 所代表的持有期天数（默认 1）；
                             用于非重叠采样和年化系数修正，避免重叠 label
                             导致累计收益和夏普虚高

        Returns
        -------
        str  实际输出目录路径
        """
        if output_dir is None:
            ic_df = result.get('ic_df')
            if ic_df is not None and not ic_df.empty:
                start = str(ic_df.index[0])
                end   = str(ic_df.index[-1])
            else:
                start = end = 'unknown'
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join('strategy', 'backtest_results',
                                      f'{start}_{end}_{ts}')

        os.makedirs(output_dir, exist_ok=True)

        if 'ic_df' in result and result['ic_df'] is not None:
            result['ic_df'].to_csv(os.path.join(output_dir, 'ic_series.csv'))

        if 'ls_df' in result and result['ls_df'] is not None:
            result['ls_df'].to_csv(os.path.join(output_dir, 'long_short_returns.csv'))

        if 'weights_df' in result and result['weights_df'] is not None:
            result['weights_df'].to_csv(os.path.join(output_dir, 'factor_weights.csv'))

        # 将统计摘要写为文本文件
        lines = []
        ic_df = result.get('ic_df')
        if ic_df is not None and not ic_df.empty and 'ic' in ic_df.columns:
            ic = ic_df['ic']
            std_ic = ic.std()
            lines += [
                '===== Rank IC 统计 =====',
                f'IC 均值:    {ic.mean():.4f}',
                f'IC 标准差:  {std_ic:.4f}',
                f'IR:         {ic.mean() / std_ic:.4f}' if std_ic != 0 else 'IR:  N/A',
                f'IC>0 比例:  {(ic > 0).mean():.2%}',
            ]

        ls_df = result.get('ls_df')
        if ls_df is not None and not ls_df.empty and 'spread' in ls_df.columns:
            # 非重叠采样：每 label_period 行取一次，消除 N 日 label 的重叠效应
            ls_nonoverlap = ls_df.iloc[::label_period]
            ann_obs = 252 / label_period
            sp = ls_nonoverlap['spread']
            cum = (1 + sp).prod() - 1
            sharpe = (sp.mean() / sp.std()) * (ann_obs ** 0.5) if sp.std() != 0 else 0.0
            lines += [
                '',
                f'===== 多空收益统计（非重叠采样，持有期={label_period}d）=====',
                ls_nonoverlap.describe().to_string(),
                f'多空累计收益: {cum:.4%}',
                f'年化夏普:     {sharpe:.4f}',
            ]

        weights_df = result.get('weights_df')
        if weights_df is not None and not weights_df.empty:
            lines += [
                '',
                '===== 因子权重均值 =====',
                weights_df.mean().sort_values(ascending=False).to_string(),
            ]

        fa_df = result.get('factor_analysis_df')
        if fa_df is not None and not fa_df.empty and 'ic_diff' in fa_df.columns:
            ic_thr = 0.01
            lr_thr = 0.0
            has_lr = 'long_ret_diff' in fa_df.columns
            harmful  = fa_df[(fa_df['ic_diff'] < -ic_thr) & (fa_df['long_ret_diff'] < lr_thr if has_lr else True)].sort_values('ic_diff')
            tradeoff = fa_df[(fa_df['ic_diff'] < -ic_thr) & (fa_df['long_ret_diff'] >= lr_thr)] if has_lr else fa_df.iloc[0:0]
            positive = fa_df[fa_df['ic_diff'] >  ic_thr].sort_values('ic_diff', ascending=False)
            lines += [
                '',
                '===== 因子效果分类（IC差分 + 多头收益差分）=====',
                '方法：ic_diff = 因子活跃时 RankIC 均值 − 未使用时 RankIC 均值',
                '      long_ret_diff = 因子活跃时 Q5实际收益均值 − 未使用时 Q5实际收益均值',
                '判断：仅当 ic_diff<0 且 long_ret_diff<0 时才认为真正有害',
                '',
                f'正面因子（ic_diff > +{ic_thr}，共 {len(positive)} 个）：',
            ]
            for f, row in positive.iterrows():
                lr_str = f"  lr_diff={row['long_ret_diff']:+.5f}" if has_lr else ''
                lines.append(f"  {f:<40s}  ic_diff={row['ic_diff']:+.4f}{lr_str}  imp={row['mean_importance']:.1f}")
            lines += ['', f'真正有害（ic_diff<-{ic_thr} 且 long_ret_diff<0，共 {len(harmful)} 个，建议删除）：']
            for f, row in harmful.iterrows():
                lr_str = f"  lr_diff={row['long_ret_diff']:+.5f}" if has_lr else ''
                lines.append(f"  {f:<40s}  ic_diff={row['ic_diff']:+.4f}{lr_str}  imp={row['mean_importance']:.1f}")
            if not tradeoff.empty:
                lines += ['', f'以排序换头部收益（ic_diff负但lr_diff正，共 {len(tradeoff)} 个，谨慎删除）：']
                for f, row in tradeoff.sort_values('long_ret_diff', ascending=False).iterrows():
                    lines.append(f"  {f:<40s}  ic_diff={row['ic_diff']:+.4f}  lr_diff={row['long_ret_diff']:+.5f}  imp={row['mean_importance']:.1f}")

        with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f'结果已保存 → {output_dir}')
        return output_dir
