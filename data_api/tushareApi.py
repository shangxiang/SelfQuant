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

    def get_stock_data(self, code, startdate, enddate, adj='qfq'):
        """
        获取某只股票的基础数据
        """
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
