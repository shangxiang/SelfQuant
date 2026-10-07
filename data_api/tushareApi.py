import tushare as ts
import pandas as pd
import time
import datetime
from .data_source import DataSource


class TushareDataSource(DataSource):
    def __init__(self):
        super(TushareDataSource, self).__init__()
        ts.set_token('a365a01b820fb9460a24b6f0ee81468daddf4903486e8b2717144dd3')
        self.pro = ts.pro_api()

    def get_stock_list(self):
        """
        获取日期下的股票列表
        """
        return self.pro.stock_basic(exchange='', list_status='L', market='主板')

    def get_stock_data(self, code, startdate, enddate, adj=None):
        """
        获取某只股票的日线行情数据。
        默认返回不复权数据（adj=None），适用于回测交易模拟。
        如需复权数据，可传入 adj='qfq'（前复权）或 adj='hfq'（后复权）。
        """
        if adj is None:
            # 不复权：直接使用 daily 接口，返回真实交易价格
            return self.pro.daily(ts_code=code, start_date=startdate, end_date=enddate)
        else:
            # 复权：使用 pro_bar 接口
            return ts.pro_bar(ts_code=code, start_date=startdate, end_date=enddate, adj=adj)

    def get_trade_calender(self, startdate, enddate, exchange='SSE', is_open='1'):
        """
        获取交易日历
        :param startdate:
        :param enddate:
        :param exchange:
        :param is_open:
        :return:
        """
        return self.pro.trade_cal(start_date=startdate, end_date=enddate, exchange=exchange, is_open=is_open)

    def get_daily_basic_data(self, date, code='', startdate='', enddate=''):
        """
        获取股票的详细数据
        :param date:
        :return:
        """
        return self.pro.daily_basic(trade_date=date, ts_code=code, start_date=startdate, end_date=enddate)

    def get_moneyflow(self, date, code='', startdate='', enddate=''):
        """
        获取大中小单流向
        :param date:
        :return:
        """
        return self.pro.moneyflow(trade_date=date, ts_code=code, start_date=startdate, end_date=enddate)

    def get_margin_detail(self, date, code='', startdate='', enddate=''):
        """
        获取融资融券数据
        :param date:
        :return:
        """
        return self.pro.margin_detail(trade_date=date, ts_code=code, start_date=startdate, end_date=enddate)

    def get_top_list(self, date, code=''):
        """
        获取龙虎榜单
        :param date:
        :return:
        """
        return self.pro.top_list(trade_date=date, ts_code=code)
        pass

    def get_income(self, code, startdate='', enddate=''):
        """
        获取利润表
        :param date:
        :param code:
        :return:
        """
        return self.pro.income(ts_code=code, start_date=startdate, end_date=enddate)
        pass

    def get_balancesheet(self, code, startdate='', enddate=''):
        """
        获取资产负债表
        :param date:
        :param code:
        :return:
        """
        return self.pro.balancesheet(ts_code=code, start_date=startdate, end_date=enddate)
        pass

    def get_cashflow(self, code, startdate='', enddate=''):
        """
        获取现金流量表
        :param date:
        :param code:
        :return:
        """
        return self.pro.cashflow(ts_code=code, start_date=startdate, end_date=enddate)
        pass

    def get_index_basic(self, market):
        """
        获取市场指数信息
        :param date:
        :return:
        """
        return self.pro.index_basic(market=market)
        pass

    def get_index_daily(self, ts_code, startdate='', enddate=''):
        """
        获取指数信息
        :param market:
        :return:
        """
        return self.pro.index_daily(ts_code=ts_code, start_date=startdate, end_date=enddate)
        pass

    def get_adj_factor(self, code, startdate='', enddate=''):
        """
        获取股票复权因子
        :param code: 股票代码
        :param startdate: 开始日期
        :param enddate: 结束日期
        :return: 复权因子DataFrame，包含trade_date和adj_factor列
        """
        return self.pro.adj_factor(ts_code=code, start_date=startdate, end_date=enddate)

    def get_namechange(self, code='', startdate='', enddate=''):
        """
        获取股票名称变更记录，用于识别ST股票
        :param code: 股票代码（可选，不传则获取全部）
        :param startdate: 开始日期
        :param enddate: 结束日期
        :return: 名称变更DataFrame，包含ts_code, name, start_date, end_date, ann_title等
        """
        return self.pro.namechange(ts_code=code, start_date=startdate, end_date=enddate, fields='ts_code,name,start_date,end_date,ann_title,ann_type')

    def get_st_stocks_by_date(self, date):
        """
        获取指定日期的ST股票列表
        通过namechange接口获取该日期所有名称包含ST的股票
        :param date: 日期，格式YYYYMMDD
        :return: ST股票代码列表
        """
        # 获取该日期之前所有名称变更记录
        df = self.pro.namechange(start_date='20100101', end_date=date, 
                                 fields='ts_code,name,start_date,end_date')
        if df is None or df.empty:
            return []
        
        # 筛选出在指定日期名称包含ST的股票
        # 条件：start_date <= date 且 (end_date > date 或 end_date为空)
        df['start_date'] = pd.to_datetime(df['start_date'], format='%Y%m%d')
        df['end_date'] = pd.to_datetime(df['end_date'], format='%Y%m%d', errors='coerce')
        target_date = pd.to_datetime(date, format='%Y%m%d')
        
        mask = (df['start_date'] <= target_date) & \
               ((df['end_date'] > target_date) | df['end_date'].isna()) & \
               (df['name'].str.contains('ST', case=False, na=False))
        
        st_df = df[mask]
        return st_df['ts_code'].unique().tolist()
