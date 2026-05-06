from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

class DataSource(ABC):
    """
    数据源抽象类
    """

    @abstractmethod
    def get_stock_list(self):
        """
        获取日期下的股票列表
        """
        pass

    @abstractmethod
    def get_stock_data(self, code, startdate, enddate):
        """
        获取某只股票的基础数据
        """
        pass

    @abstractmethod
    def get_daily_basic_data(self, date, code, startdate, enddate):
        """
        获取股票的详细数据
        :param date:
        :return:
        """
        pass

    @abstractmethod
    def get_moneyflow(self, date, code, startdate, enddate):
        """
        获取大中小单流向
        :param date:
        :return:
        """
        pass

    @abstractmethod
    def get_margin_detail(self, date, code, startdate, enddate):
        """
        获取融资融券数据
        :param date:
        :return:
        """
        pass

    @abstractmethod
    def get_top_list(self, date, code):
        """
        获取龙虎榜单
        :param date:
        :return:
        """
        pass


    @abstractmethod
    def get_income(self, code, startdate='', enddate=''):
        """
        获取利润表
        :param date:
        :param code:
        :return:
        """
        pass

    @abstractmethod
    def get_balancesheet(self, code, startdate='', enddate=''):
        """
        获取资产负债表
        :param date:
        :param code:
        :return:
        """
        pass

    @abstractmethod
    def get_cashflow(self, code, startdate='', enddate=''):
        """
        获取现金流量表
        :param date:
        :param code:
        :return:
        """
        pass

    @abstractmethod
    def get_index_basic(self, market):
        """
        获取指数信息
        :param market:
        :return:
        """
        pass

    @abstractmethod
    def get_index_daily(self, trade_date):
        """
        获取指数信息
        :param market:
        :return:
        """
        pass