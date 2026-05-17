import tushare as ts
import pandas as pd
from tushare import margin_detail
from tushare.coins import market
from data_api.tushareApi import TushareDataSource
import glob

start = '20230101'
end = '20260516'

api = TushareDataSource()
df = api.get_index_daily(ts_code='000510.CSI', startdate=start, enddate=end)
df.to_csv('data/raw/index_daily/000510.CSI.csv')
print(df)
