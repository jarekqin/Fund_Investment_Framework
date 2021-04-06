
import pickle
import time
import re
import pandas as pd
import numpy as np

import os
import requests
import warnings

from datetime import datetime, timedelta
import json

import xlrd

warnings.filterwarnings('ignore')


class CHINAEXCHANGERZRQ:

    @staticmethod
    def get_data_from_exchange(date, ex_name=None, read_path1='H:\\RZRQ_csv\\sz', read_path2='H:\\RZRQ_csv\\sh'):
        if ex_name is None:
            raise ValueError('交易所简称不能为空!')
        elif ex_name not in ['sz', 'SZ', 'sh', 'SH']:
            raise ValueError('交易所简称输入只能是sz(SZ)/sh(SH)!')
        if ex_name in ['sz', 'SZ']:
            data = CHINAEXCHANGERZRQ()._get_data_from_sz_ex(date, read_path1)
        elif ex_name in ['sh', 'SH']:
            data = CHINAEXCHANGERZRQ()._get_data_from_sh_ex(date, read_path2)
        else:
            raise ValueError('交易所简称输入只能是sz(SZ)/sh(SH)!')
        return data

    def _get_data_from_sz_ex(self, date, read_path='H:\\RZRQ_csv\\sz'):
        data = pd.read_excel(os.path.join(read_path, '%ssz_rzrq_data.csv' % date))
        data['证券代码'] = [('0' * (6 - len(str(x))) + str(x) + '.SZ') if len(str(x)) < 6 else str(x) + '.SZ' for x in
                        data['证券代码']]
        if data.shape[0] < 2:
            return None
        data['证券简称'] = data['证券简称'].apply(lambda x: x.replace(' ', ''))
        col = ['security_code', 'security_institution', 'rz_buy_amounts', 'rz_remained_amounts', 'rq_short_shares',
               'rq_remained_shares', 'rq_remained_amounts', 'rzrq_remained_amounts']
        data.columns = col
        data['as_of_date'] = pd.to_datetime(date).date()
        data['rq_returned_shares'] = np.nan
        data['rz_returned_amounts'] = np.nan

        data = data[['as_of_date', 'security_code', 'security_institution', 'rz_buy_amounts',
                     'rz_remained_amounts', 'rq_short_shares', 'rq_remained_shares',
                     'rq_remained_amounts', 'rzrq_remained_amounts', 'rq_returned_shares',
                     'rz_returned_amounts']]
        data['rz_buy_amounts'] = data['rz_buy_amounts'].apply(lambda x: float(x.replace(',', '')))
        data['rz_remained_amounts'] = data['rz_remained_amounts'].apply(lambda x: float(x.replace(',', '')))
        data['rq_remained_amounts'] = data['rq_remained_amounts'].apply(lambda x: float(x.replace(',', '')))
        data['rzrq_remained_amounts'] = data['rzrq_remained_amounts'].apply(lambda x: float(x.replace(',', '')))
        data['rq_short_shares'] = data['rq_short_shares'].apply(lambda x: float(x.replace(',', '')))
        data['rq_remained_shares'] = data['rq_remained_shares'].apply(lambda x: float(x.replace(',', '')))
        return data

    def _get_data_from_sh_ex(self, date, read_path='H:\\RZRQ_csv\\sh'):
        try:
            data = pd.read_excel(os.path.join(read_path, '%ssh_rzrq_data.xls' % date), sheet_name='明细信息')
        except xlrd.biffh.XLRDError:
            return None
        data['标的证券代码'] = [str(x) + '.SH' for x in data['标的证券代码']]
        data['as_of_date'] = [pd.to_datetime(date).date()] * data.shape[0]
        col = ['security_code', 'security_institution', 'rz_remained_amounts', 'rz_buy_amounts',
               'rz_returned_amounts', 'rq_remained_shares', 'rq_short_shares', 'rq_returned_shares',
               'as_of_date']
        data.columns = col
        data['rq_remained_amounts'] = np.nan * data.shape[0]
        data['rzrq_remained_amounts'] = np.nan * data.shape[0]
        # 转换数据类型
        data['rz_remained_amounts'] = data['rz_remained_amounts'].apply(lambda x: float(x))
        data['rz_buy_amounts'] = data['rz_buy_amounts'].apply(lambda x: float(x))
        data['rz_returned_amounts'] = data['rz_returned_amounts'].apply(lambda x: float(x))
        data['rq_remained_shares'] = data['rq_remained_shares'].apply(lambda x: float(x))
        data['rq_short_shares'] = data['rq_short_shares'].apply(lambda x: float(x))
        data['rq_returned_shares'] = data['rq_returned_shares'].apply(lambda x: float(x))
        return data


def save_daily_rzrq_to_csv(data, save_path):
    data.to_csv(save_path, encoding='utf-8')


if __name__ == '__main__':
    from WindPy import w
    from chinese_calendar import is_holiday

    # w.start()

    # date = w.tdays(datetime.today()-timedelta(7),datetime.today()).Data[0][-2].date()

    spring_holiday = pd.period_range('2021-02-11', '2021-02-17', freq='1D')
    spring_holiday = [x.to_timestamp().date() for x in spring_holiday]
    date=datetime(2021,4,2).date()
    #
    # while date<(datetime.today()).date():
    #     if date.isoweekday() in [6,7]:
    #         print('%s无数据!' % date.strftime('%Y-%m-%d'))
    #         date += timedelta(1)
    #         continue
    #     sz_data = CHINAEXCHANGERZRQ.get_data_from_exchange(date=date, ex_name='sz')
    #     sh_data=CHINAEXCHANGERZRQ.get_data_from_exchange(date=date, ex_name='sh')
    #     if sz_data is None or sh_data is None:
    #         print('%s无数据!' % date.strftime('%Y-%m-%d'))
    #         date += timedelta(1)
    #         continue
    #     save_data_to_sql(sz_data, CHINAEXCHANGE)
    #     save_data_to_sql(sh_data, CHINAEXCHANGE)
    #     print('%s输入完成!' % date.strftime('%Y-%m-%d'))
    #     date = date + timedelta(1)
    # while date<datetime(2021,1,1).date():
    while date < (datetime.today()).date() and date not in spring_holiday:
        if date.isoweekday() == 6 or date.isoweekday() == 7:
            print('%s是周末,跳过' % date.strftime('%Y-%m-%d'))
            date = date + timedelta(1)
            continue
        while True:
            sz_data = CHINAEXCHANGERZRQ.get_data_from_exchange(date=date, ex_name='sz')
            if sz_data is None or sz_data.shape[0]==0:
                print('深圳%s数据抓取完成' % date.strftime('%Y-%m-%d'))
                break
            else:
                save_daily_rzrq_to_csv(sz_data, 'H:\\RZRQ_result_csv\\sz\\%s_sz_rzrq.csv' % date.strftime('%Y-%m-%d'))
                break
        # 再爬上海交易所两融数据
        while True:
            sh_data = CHINAEXCHANGERZRQ.get_data_from_exchange(date=date, ex_name='sh')
            if sh_data is None :
                print('上海%s数据抓取完成' % date.strftime('%Y-%m-%d'))
                break
            else:
                save_daily_rzrq_to_csv(sh_data,'H:\\RZRQ_result_csv\\sh\\%s_sh_rzrq.csv' % date.strftime('%Y-%m-%d'))
                break
        date = date + timedelta(1)
