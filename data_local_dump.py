import os

import tushare as ts
import pandas as pd
from tushare import margin_detail
from tushare.coins import market
from data_api.tushareApi import TushareDataSource
from datetime import datetime, timedelta
import time


class DownloadData():
    """
    用于下载线上数据到本地进行回测
    """
    def __init__(self, api=TushareDataSource):
        self.api = api()
        self.stock_list_path = "data/raw/stock_list/stock_list.csv"

    def get_local_calender(self):
        trade_cal_path = "data/raw/trade_cal.csv"
        df = pd.read_csv(trade_cal_path)
        trade_date = df['cal_date']
        return trade_date

    def stock_list(self):
        df = self.api.get_stock_list()
        df.to_csv(self.stock_list_path)

    def stock_data(self):
        stock_list_df = pd.read_csv(self.stock_list_path)
        stock_list = stock_list_df['ts_code'].tolist()
        stock_data_path = "data/raw/stock_data/"
        for ts_code in stock_list:
            df = self.api.get_stock_data(code=ts_code,startdate="20230101", enddate=datetime.today().date().strftime("%Y%m%d"))
            df.to_csv(stock_data_path + ts_code + ".csv")
            time.sleep(0.1)

    def daily_basic_data(self):
        trade_cal = self.get_local_calender()
        #trade_cal.to_csv("data/raw/trade_cal.csv")
        stock_data_path = "data/raw/daily_basic_data/"
        for date in trade_cal['cal_date']:
            print(date)
            df = self.api.get_daily_basic_data(date=date)
            df.to_csv(stock_data_path + date + ".csv")
            time.sleep(0.1)

    def moneyflow(self):
        trade_date = self.get_local_calender()
        moneyflow_path = "data/raw/moneyflow/"
        for date in trade_date:
            date = str(date)
            print(date)
            df = self.api.get_moneyflow(date=date)
            df.to_csv(moneyflow_path + date + ".csv")
            time.sleep(0.2)

    def margin_detail(self):
        trade_date = self.get_local_calender()
        margin_detail_path = "data/raw/margin_detail/"
        for date in trade_date:
            date = str(date)
            print(date)
            df = self.api.get_margin_detail(date=date)
            df.to_csv(margin_detail_path + date + ".csv")
            time.sleep(0.2)

    def top_list(self):
        trade_date = self.get_local_calender()
        top_list_path = "data/raw/top_list/"
        for date in trade_date:
            date = str(date)
            print(date)
            df = self.api.get_top_list(date=date)
            df.to_csv(top_list_path + date + ".csv")
            time.sleep(0.4)

    def income(self):
        stock_list_df = pd.read_csv(self.stock_list_path)
        stock_list = stock_list_df['ts_code'].tolist()
        stock_data_path = "data/raw/income/"
        for ts_code in stock_list:
            print(ts_code)

            df = self.api.get_income(code=ts_code, startdate="20230101",
                                         enddate="20260407")
            df.to_csv(stock_data_path + ts_code + ".csv")
            time.sleep(0.4)

    def balancesheet(self):
        stock_list_df = pd.read_csv(self.stock_list_path)
        stock_list = stock_list_df['ts_code'].tolist()
        stock_data_path = "data/raw/balancesheet/"
        for ts_code in stock_list:
            print(ts_code)
            df = self.api.get_balancesheet(code=ts_code, startdate="20230101",
                                         enddate="20260407")
            df.to_csv(stock_data_path + ts_code + ".csv")
            time.sleep(0.4)

    def cashflow(self):
        stock_list_df = pd.read_csv(self.stock_list_path)
        stock_list = stock_list_df['ts_code'].tolist()
        stock_data_path = "data/raw/cashflow/"
        for ts_code in stock_list:
            print(ts_code)
            df = self.api.get_cashflow(code=ts_code, startdate="20230101",
                                         enddate="20260407")
            df.to_csv(stock_data_path + ts_code + ".csv")
            time.sleep(0.4)

    def index_basic(self):
        stock_data_path = "data/raw/index_basic/"
        market_list = ["SW", "CSI", "SSE", "SZSE", "MSCI"]
        for market in market_list:
            df = self.api.get_index_basic(market=market)
            df.to_csv(stock_data_path + market + ".csv")
            time.sleep(0.4)

    def index_daily(self):
        SW_code_list = pd.read_csv("data/raw/index_basic/SZSE.csv")
        code_list = SW_code_list['ts_code'].tolist()
        stock_data_path = "data/raw/index_daily/"
        for code in code_list:
            df = self.api.get_index_daily(ts_code=code, startdate="20230101", enddate="20260407")
            df.to_csv(stock_data_path + code + ".csv")
            time.sleep(0.4)



if __name__ == '__main__':
    download = DownloadData()
    download.index_daily()