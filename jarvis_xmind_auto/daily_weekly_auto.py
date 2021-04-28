import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import mpl_finance as finance_mpl
import yfinance as yf
import os

import excel2img

import time
import datetime

import requests
import warnings

# from omk.core.vendor.RQData import RQData
from omk.interface import AbstractJob
from omk.events import Event
from omk.utils.const import FolderName, TODAY, ProcessDocs, ProcessType, EVENT, TIME

from jarvis.utils import FOLDER, mkdir, copy_to_jarvisoutput, XlsxSaver
import matplotlib.ticker as ticker

from matplotlib import colors as mcolors

from datetime import datetime, timedelta
from WindPy import w

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False
sns.set(font_scale=1.5, font='SimHei')
warnings.filterwarnings("ignore")

# based_file = FOLDER.Syn_save
based_file = 'H:\\'
my_fred_api_key = '7e41a8c76559d6eed6b02fbeccd136fd'


class MorningAfterMetting:
    @staticmethod
    def get_main_global_contracts(date=None, init=True):
        if init:
            w.start()
        # temp = w.wset("sectorconstituent",
        #               "date=%s;sectorid=1000015511000000" % pd.to_datetime(date).strftime('%Y-%m-%d'))
        # data = pd.DataFrame(np.transpose(temp.Data), columns=temp.Fields)
        # # 过滤合约
        # filter_contracts1 = {'SP500小型': 'CME', '纳指100小型': 'CME',
        #                      '黄金': 'CMX', '轻质原油': 'NYM'}
        # filter_contracts2 = 'MGCM|QOM|SGCZ|SGUM|QMK'
        # final_data = pd.DataFrame()
        # for contract_name, contract_tail in filter_contracts1.items():
        #     final_data = pd.concat([final_data, data[data.sec_name.str.contains(contract_name) &
        #                                              data.wind_code.str.contains(contract_tail)
        #                                              ]],
        #                            axis=0)
        # final_data = final_data[final_data.wind_code.str.contains(filter_contracts2) == False]
        # for index, code in zip(final_data.wind_code.index.to_list(),
        #                        final_data.wind_code.to_list()):
        #     if code.split('.')[0].rfind('E') == len(code.split('.')[0]) - 1:
        #         final_data.wind_code[index] = '.'.join([code.split('.')[0][:-1], code.split('.')[-1]])
        #     else:
        #         final_data.wind_code[index] = '.'.join([code.split('.')[0].replace('E', ''), code.split('.')[-1]])
        #
        # returned_dict = {x: y for x, y in zip(final_data.wind_code.to_list(), final_data.sec_name.to_list())}
        returned_dict={}
        # returned_dict.pop('QMM21.NYM')
        returned_dict.update({'GCM21.CMX': '黄金'})
        returned_dict.update({'CLM21.NYM': '轻质原油'})
        returned_dict.update({'^VIX': 'VIX'})
        # returned_dict.update({'^TNX': '10年期美债收益率'})
        returned_dict.update({'ZT=F': '美债2年期货'})
        returned_dict.update({'ZN=F': '美债10年期货'})
        returned_dict.update({'TIP': '美国TIPS通涨保值债券ETF'})
        returned_dict.update({'^DJI': 'Dow指数'})
        returned_dict.update({'^IXIC': 'NASDAQ指数'})
        returned_dict.update({'^GSPC': 'SP500指数'})
        returned_dict.update({'^FTSE': '英国富时100'})
        returned_dict.update({'^GDAXI': 'GDAXI(德国DAX)'})
        returned_dict.update({'^RUT': '罗素2000指数'})
        returned_dict.update({'^NDX': 'NASDAQ100指数'})
        returned_dict.update({'ES=F': '标普期货'}),
        returned_dict.update({'YM=F': '道琼斯期货'}),
        returned_dict.update({'NQ=F': '纳斯达克期货'}),
        returned_dict.update({'^N225': '日经指数'}),
        returned_dict.update({'^STOXX50E': '斯托克50'}),
        returned_dict.update({'CNH=X': '美元兑换离岸人民币'})

        return returned_dict

    @staticmethod
    def get_past_one_month_trendency(wind_code_name_dict, date=None):
        if len(wind_code_name_dict) == 0:
            raise ValueError('wind code list is empty!')
        elif isinstance(wind_code_name_dict, dict):
            wind_code_list = ','.join(list(wind_code_name_dict.keys()))

        date = pd.to_datetime(date).strftime('%Y-%m-%d')
        start_date = (pd.to_datetime(date).date() - timedelta(30)).strftime('%Y-%m-%d')
        # 获取wind数据
        temp1 = w.wsd("%s" % wind_code_list, "close", "%s" % start_date, "%s" % date, "")
        trendency_df = pd.DataFrame(np.transpose(temp1.Data), columns=temp1.Codes, index=temp1.Times).fillna(0)
        # pct from previous:16:00 to today 16:00
        temp2 = w.wsd("%s" % wind_code_list, "pct_chg", "%s" % start_date, "%s" % date, "")
        pct_df = pd.DataFrame(np.transpose(temp2.Data), columns=temp2.Codes, index=temp2.Times).fillna(0)

        # 更换列名
        pct_df.columns = [wind_code_name_dict[key] for key in pct_df.columns]
        trendency_df.columns = [wind_code_name_dict[key] for key in trendency_df.columns]

        return trendency_df, pct_df

    @staticmethod
    def plot_candle(wind_code, contract_name, start_date, end_date,
                    use_wind=True, df=None, save_path=based_file, file_name=None,
                    use_chinese_time_zone=True):
        # get data
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        if use_wind:
            temp = w.wsd("%s" % wind_code, "open,high,low,close", "%s" % start_date.strftime('%Y-%m-%d'),
                         "%s" % end_date.strftime('%Y-%m-%d'), "")
            data = pd.DataFrame(np.transpose(temp.Data), index=temp.Times, columns=temp.Fields).dropna()
            data.index = pd.to_datetime(data.index)
            plot_data = data.loc[(end_date - timedelta(120)):]
        else:
            data = df[['Open', 'High', 'Low', 'Close']]
            data.columns = data.columns.str.upper()
            if use_chinese_time_zone:
                data.index = [x.tz_localize('Asia/Shanghai') for x in data.index]
            plot_data = data.loc[start_date:end_date]

        # 3 ma line

        ma20 = data['CLOSE'].rolling(20).mean()
        ma60 = data['CLOSE'].rolling(60).mean()
        ma120 = data['CLOSE'].rolling(120).mean()

        fig = plt.figure(figsize=(45, 20), edgecolor='black', facecolor='black')
        ax = fig.add_subplot()
        finance_mpl.candlestick2_ohlc(ax, opens=plot_data.OPEN, highs=plot_data.HIGH, lows=plot_data.LOW,
                                      closes=plot_data.CLOSE, width=0.9, colorup='r', colordown='cyan',
                                      alpha=0.9)
        ax.plot(ma20.loc[plot_data.index].values, label='20MA', color='fuchsia')
        ax.plot(ma60.loc[plot_data.index].values, label='60MA', color='lime')
        ax.plot(ma120.loc[plot_data.index].values, label='120MA', color='aqua')
        # finance_mpl.candlestick_ohlc(ax,data,width=0.35, colorup='r', colordown='green')
        ax.set_title(
            '报告日:%s,%s合约K线图,绝对值:%0.4f,涨跌%0.4f,涨跌幅:%0.2f' % (
                plot_data.index[-1].strftime('%Y-%m-%d'), wind_code,
                plot_data.CLOSE[-1],
                plot_data.CLOSE[-1] -
                plot_data.CLOSE[-2],
                round((plot_data.CLOSE[-1] -
                       plot_data.CLOSE[-2]) / plot_data.CLOSE[-2] * 100,
                      2)
            ) + '%',
            fontsize=35, color='white'
        )
        if use_chinese_time_zone:
            ax.set_xlabel('时间(亚洲/上海时区)', fontsize=35, color='white')
        else:
            ax.set_xlabel('时间(当地时区)', fontsize=35, color='white')
        ax.set_ylabel('价格点数', fontsize=35, color='white')
        plt.grid(True, color='white', alpha=0.3)
        plt.xticks(range(0, len(plot_data.index), 16), [x.strftime('%Y-%m-%d') for x in
                                                        plot_data.index[list(range(0, len(plot_data.index),
                                                                                   16))]],
                   rotation=45, color='white',
                   fontsize=35)
        ax.set_yticklabels([round(x, 2) for x in ax.get_yticks()], fontsize=35,
                           color='white')
        ax.patch.set_facecolor("black")  # 设置 ax1 区域背景颜色
        ax.patch.set_alpha(0.5)  # 设置 ax1 区域背景颜色透明度

        # ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        fig.subplots_adjust(bottom=0.2)  #
        plt.legend(loc='best')
        if save_path is None:
            plt.show()
        else:
            if file_name is None:
                if datetime.now().hour >= 6 and datetime.now().hour < 16:
                    file_name = 'morning_temp'
                else:
                    file_name = 'afternoon_temp'
            mkdir(os.path.join(based_file, file_name, datetime.today().strftime('%Y-%m-%d')))
            temp_path = os.path.join(based_file, file_name, datetime.today().strftime('%Y-%m-%d'))
            save_path = os.path.join(temp_path, '%s.png' % (contract_name))
            plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def plot_curve2(data, picture_name,
                    ylabel_name, xlabel_name, title_value,
                    pct_value, pct_change,
                    save_path=based_file, plot_col=None, lag_days=0,
                    file_name=None):

        def format_date(x, pos=None):
            if x < 0 or x > len(data.index) - 1:
                return ''
            return data.index[int(x)]

        data.index = pd.to_datetime(data.index)
        # data.index=[x.strftime('%Y-%m-%d') for x in data.index]
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
        x = range(len(data.index))
        if plot_col is None:
            ax.plot(x, data.values, label='Close')
        else:
            ax.plot(x, data[plot_col])
        plt.title('报告日:%s,%s日图(滞后了T-%d日),绝对数是:%0.2f,涨跌是:%0.2f,涨跌幅是:%0.2f' % (
            data.index[-1].strftime('%Y-%m-%d'),
            picture_name, lag_days,
            title_value, pct_value,
            pct_change) + '%')
        ax.set_ylabel(ylabel_name)
        ax.set_xlabel(xlabel_name)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(48))
        if len(data.index)>100:
            plt.xticks(range(0, len(data.index), 30), [x.strftime('%Y-%m-%d') for x in
                                                       data.index[list(range(0, len(data.index),
                                                                             30))]],
                       rotation=45)
        else:
            plt.xticks(range(0, len(data.index), 5), [x.strftime('%Y-%m-%d') for x in
                                                       data.index[list(range(0, len(data.index),
                                                                             5))]],
                       rotation=45)
        fig.autofmt_xdate()
        if save_path is None:
            plt.show()
        else:
            if file_name is None:
                if datetime.now().hour >= 6 and datetime.now().hour < 16:
                    file_name = 'morning_temp'
                else:
                    file_name = 'afternoon_temp'
            mkdir(os.path.join(based_file, file_name, datetime.today().strftime('%Y-%m-%d')))
            temp_path = os.path.join(based_file, file_name, datetime.today().strftime('%Y-%m-%d'))
            save_path = os.path.join(temp_path, '%s.png' % (picture_name))
            plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def plot_curve(data, picture_name,
                   ylabel_name, xlabel_name, title_value,
                   pct_value, pct_change,
                   save_path=based_file, plot_col=None, file_name=None):

        def format_date(x, pos=None):
            if x < 0 or x > len(data.index) - 1:
                return ''
            return data.index[int(x)]

        data.index = pd.to_datetime(data.index)
        # data.index=[x.strftime('%Y-%m-%d') for x in data.index]
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
        x = range(len(data.index))
        if plot_col is None:
            ax.plot(x, data.values, label='Close')
        else:
            ax.plot(x, data[plot_col])
        plt.title('报告日:%s,%s分时图,绝对数是:%0.2f,涨跌是:%0.4f,涨跌幅是:%0.2f' % (
            data.index[-1],
            picture_name,
            title_value, pct_value,
            pct_change) + '%')
        ax.set_ylabel(ylabel_name)
        ax.set_xlabel(xlabel_name)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(48))
        plt.xticks(range(0, len(data.index), 30), [x.strftime('%Y-%m-%d %H:%M:%S') for x in
                                                   data.index[list(range(0, len(data.index),
                                                                         30))]],
                   rotation=45)
        fig.autofmt_xdate()
        if save_path is None:
            plt.show()
        else:
            if file_name is None:
                if datetime.now().hour >= 6 and datetime.now().hour < 16:
                    file_name = 'morning_temp'
                else:
                    file_name = 'afternoon_temp'
            mkdir(os.path.join(based_file, file_name, datetime.today().strftime('%Y-%m-%d')))
            temp_path = os.path.join(based_file, file_name, datetime.today().strftime('%Y-%m-%d'))
            save_path = os.path.join(temp_path, '%s.png' % (picture_name))
            plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def get_contract_data(wind_code, end_date, period='3d', interval='1m'):
        end_date = pd.to_datetime(end_date)
        if end_date.isoweekday() not in [1, 6, 7]:
            start_date = end_date - timedelta(1)
        elif end_date.isoweekday() == 7:
            start_date = end_date - timedelta(3)
            period = '4d'
        elif end_date.isoweekday() == 6:
            start_date = end_date - timedelta(2)
            period = '4d'
        elif end_date.isoweekday() == 1:
            start_date = end_date - timedelta(3)
        else:
            print('Weekend!')
        if isinstance(wind_code, dict) or isinstance(wind_code, list):
            data = {}
            daily_data = {}
            for code in wind_code:
                data_model = yf.Ticker(code)
                if code != '^VIX':
                    try:
                        temp_data = data_model.history(period=period, interval=interval,
                                                       start=start_date.strftime('%Y-%m-%d'),
                                                       end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                        data[code] = temp_data
                        daily_data[code] = data_model.history(end=(end_date + timedelta(2)).strftime('%Y-%m-%d'),
                                                              start=pd.to_datetime(datetime(end_date.year - 1,
                                                                                            end_date.month,
                                                                                            end_date.day)).strftime(
                                                                  '%Y-%m-%d'))
                    except requests.exceptions.ConnectionError:
                        counter = 0
                        while True:
                            temp_data = pd.DataFrame()
                            try:
                                temp_data = data_model.history(period=period, interval=interval,
                                                               start=start_date.strftime('%Y-%m-%d'),
                                                               end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                            except requests.exceptions.ConnectionError:
                                counter += 1
                            if counter >= 30:
                                requests.exceptions.ConnectionError('无法请求到雅虎金融的数据!')
                            if temp_data.shape[0] > 0:
                                break
                        data[code] = temp_data
                        daily_data[code] = data_model.history(end=(end_date + timedelta(2)).strftime('%Y-%m-%d'),
                                                              start=pd.to_datetime(datetime(end_date.year - 1,
                                                                                            end_date.month,
                                                                                            end_date.day)).strftime(
                                                                  '%Y-%m-%d'))
                else:
                    try:
                        temp_data = data_model.history(period=period, interval=interval,
                                                       start=start_date.strftime('%Y-%m-%d'),
                                                       end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                        data[code] = temp_data
                        daily_data[code] = data_model.history(end=(end_date + timedelta(2)).strftime('%Y-%m-%d'),
                                                              start=pd.to_datetime(datetime(end_date.year - 1,
                                                                                            end_date.month,
                                                                                            end_date.day)).strftime(
                                                                  '%Y-%m-%d'))
                    except requests.exceptions.ConnectionError:
                        counter = 0
                        while True:
                            time.sleep(10)
                            temp_data = pd.DataFrame()
                            try:
                                temp_data = data_model.history(period=period, interval=interval,
                                                               start=start_date.strftime('%Y-%m-%d'),
                                                               end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                            except requests.exceptions.ConnectionError:
                                counter += 1
                            if counter >= 30:
                                requests.exceptions.ConnectionError('无法请求到雅虎金融的数据!')
                            if temp_data.shape[0] > 0:
                                break
                        data[code] = temp_data
                        daily_data[code] = data_model.history(end=(end_date + timedelta(2)).strftime('%Y-%m-%d'),
                                                              start=pd.to_datetime(datetime(end_date.year - 1,
                                                                                            end_date.month,
                                                                                            end_date.day)).strftime(
                                                                  '%Y-%m-%d'))
        else:
            data_model = yf.Ticker(wind_code)
            data = {}
            daily_data = {}
            if wind_code != '^VIX':
                try:
                    temp_data = data_model.history(period=period, interval=interval,
                                                   start=start_date.strftime('%Y-%m-%d'),
                                                   end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                except requests.exceptions.ConnectionError:
                    counter = 0
                    while True:
                        time.sleep(10)
                        temp_data = pd.DataFrame()
                        try:
                            temp_data = data_model.history(period=period, interval=interval,
                                                           start=start_date.strftime('%Y-%m-%d'),
                                                           end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                        except requests.exceptions.ConnectionError:
                            counter += 1
                        if counter >= 30:
                            requests.exceptions.ConnectionError('无法请求到雅虎金融的数据!')
                        if temp_data.shape[0] > 0:
                            break
                data[wind_code] = temp_data
                daily_data[wind_code] = data_model.history(end=(end_date + timedelta(2)).strftime('%Y-%m-%d'),
                                                           start=pd.to_datetime(datetime(end_date.year - 1,
                                                                                         end_date.month,
                                                                                         end_date.day)).strftime(
                                                               '%Y-%m-%d'))
            else:
                try:
                    temp_data = data_model.history(period=period, interval=interval,
                                                   start=start_date.strftime('%Y-%m-%d'),
                                                   end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                except requests.exceptions.ConnectionError:
                    counter = 0
                    while True:
                        time.sleep(10)
                        temp_data = pd.DataFrame()
                        try:
                            temp_data = data_model.history(period=period, interval=interval,
                                                           start=start_date.strftime('%Y-%m-%d'),
                                                           end=(end_date + timedelta(2)).strftime('%Y-%m-%d'))
                        except requests.exceptions.ConnectionError:
                            counter += 1
                        if counter >= 30:
                            raise requests.exceptions.ConnectionError('无法请求到雅虎金融的数据!')
                        if temp_data.shape[0] > 0:
                            break
                data[wind_code] = temp_data
                daily_data[wind_code] = data_model.history(end=(end_date + timedelta(2)).strftime('%Y-%m-%d'),
                                                           start=pd.to_datetime(datetime(end_date.year - 1,
                                                                                         end_date.month,
                                                                                         end_date.day)).strftime(
                                                               '%Y-%m-%d'))
        # we will cut date range from T-1 at 8a.m to T at 8a.m from data according to time zone different
        if len(data.keys()) > 0:
            returned_dict = {}
            for key in data:
                print(key)

                if key not in ['^VIX','^TNX','TIP', '^DJI', '^IXIC', '^GSPC', '^NDX', '^RUT', '^GDAXI', '^FTSE',
                           '^STOXX50E', 'CNH=X']:

                    minutes_data = data[key].loc[end_date.strftime('%Y-%m-%d')]
                    minutes_data.index = [x.tz_convert('Asia/Shanghai') for x in minutes_data.index]
                else:
                    if datetime.now().isoweekday()!=1:
                        minutes_data = data[key].loc[end_date.strftime('%Y-%m-%d')]
                        minutes_data.index = [x.tz_convert('Asia/Shanghai') for x in minutes_data.index]

                data[key].index = [x.tz_convert('Asia/Shanghai') for x in data[key].index]

                if key in ['TIP', '^DJI', '^IXIC', '^GSPC', '^NDX', '^RUT', '^GDAXI', '^FTSE',
                           '^STOXX50E', 'CNH=X']:
                    if end_date.isoweekday() not in  [1,5]:
                        if datetime.now().hour>=6 and datetime.now().hour<16:
                            returned_dict[key + '_晨会'] = {
                                '涨跌': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                      data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                              data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                             data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                '收盘绝对数': minutes_data.Close[-1],
                                '分时图数据': minutes_data
                            }
                    else:
                        if key not in ['^FTSE', '^STOXX50E', '^GDAXI', '^RUT']:
                            returned_dict[key + '_晨会'] = {
                                '涨跌': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] - \
                                      data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] - \
                                              data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                             data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                '收盘绝对数': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1],
                                '分时图数据': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')]
                            }
                        else:
                            returned_dict[key + '_晨会'] = {
                                '涨跌': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                      data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                              data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                             data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                '收盘绝对数': minutes_data.Close[-1],
                                '分时图数据': minutes_data
                            }
                elif key in ['ZT=F', 'ZN=F', 'ESM21.CME', 'NQM21.CME', 'CLM21.NYM']:
                    if datetime.now().hour >= 16:
                        temp_data = data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d 08:00:00'):(
                                end_date + timedelta(1)).strftime('%Y-%m-%d 16:00:00')]
                        returned_dict[key + '_复盘'] = {
                            '涨跌': temp_data.Close[-1] - temp_data.Close[0],
                            '涨跌幅': round(
                                (temp_data.Close[-1] - temp_data.Close[0]) / temp_data.Close[0] * 100, 2),
                            '收盘绝对数': temp_data.Close[-1],
                            '分时图数据': temp_data
                        }
                    else:
                        temp_data = data[key].loc[
                                    end_date.strftime('%Y-%m-%d 15:00:00'):(end_date + timedelta(1)).strftime(
                                        '%Y-%m-%d 08:00:00')]
                        returned_dict[key + '_晨会'] = {
                            '涨跌': temp_data.Close[-1] - temp_data.Close[0],
                            '涨跌幅': round((temp_data.Close[-1] - temp_data.Close[0]) / temp_data.Close[0] * 100, 2),
                            '收盘绝对数': temp_data.Close[-1],
                            '分时图数据': temp_data
                        }
                elif key in ['GCM21.CMX', 'CLK21.NYM', '^VIX', 'ES=F', 'YM=F', 'NQ=F']:
                    if key != '^VIX':
                        if datetime.now().hour >= 16:
                            temp_data = data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d 08:00:00'):(
                                    end_date + timedelta(1)).strftime('%Y-%m-%d 16:00:00')]
                            returned_dict[key + '_复盘'] = {
                                '涨跌': temp_data.Close[-1] - temp_data.Close[0],
                                '涨跌幅': round(
                                    (temp_data.Close[-1] - temp_data.Close[0]) / temp_data.Close[0] * 100, 2),
                                '收盘绝对数': temp_data.Close[-1],
                                '分时图数据': temp_data
                            }
                        else:
                            temp_data = data[key].loc[
                                        end_date.strftime('%Y-%m-%d 15:00:00'):(end_date + timedelta(1)).strftime(
                                            '%Y-%m-%d 08:00:00')]
                            returned_dict[key + '_晨会'] = {
                                '涨跌': temp_data.Close[-1] - temp_data.Close[0],
                                '涨跌幅': round((temp_data.Close[-1] - temp_data.Close[0]) / temp_data.Close[0] * 100, 2),
                                '收盘绝对数': temp_data.Close[-1],
                                '分时图数据': temp_data
                            }
                    else:
                        if datetime.now().hour >= 16:
                            if datetime.now().isoweekday()!=1:
                                returned_dict[key + '_复盘'] = {
                                    '涨跌': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] -
                                          data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1],
                                    '涨跌幅': round((data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] -
                                                  data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1]) /
                                                 data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1], 2),
                                    '收盘绝对数': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1],
                                    '分时图数据': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')]
                                }
                            else:
                                returned_dict[key + '_复盘'] = {
                                    '涨跌': data[key].loc[(end_date - timedelta(2)).strftime('%Y-%m-%d')].Close[-1] -
                                          data[key].loc[(end_date-timedelta(3)).strftime('%Y-%m-%d')].Close[-1],
                                    '涨跌幅': round((data[key].loc[(end_date  - timedelta(2)).strftime('%Y-%m-%d')].Close[-1] -
                                                  data[key].loc[(end_date-timedelta(3)).strftime('%Y-%m-%d')].Close[-1]) /
                                                 data[key].loc[(end_date - timedelta(3)).strftime('%Y-%m-%d')].Close[-1], 2),
                                    '收盘绝对数': data[key].loc[(end_date  - timedelta(2)).strftime('%Y-%m-%d')].Close[-1],
                                    '分时图数据': data[key].loc[(end_date  - timedelta(2)).strftime('%Y-%m-%d')]
                                }
                        else:
                            if datetime.now().isoweekday()!=1:
                                returned_dict[key + '_晨会'] = {
                                    '涨跌': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                          data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1],
                                    '涨跌幅': round((data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                                  data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                                 data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                    '收盘绝对数': minutes_data.Close[-1],
                                    '分时图数据': minutes_data
                                }
                            else:
                                returned_dict[key + '_晨会'] = {
                                    '涨跌': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                          data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1],
                                    '涨跌幅': round((data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                                  data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                                 data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                    '收盘绝对数': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1],
                                    '分时图数据': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1]
                                }
                elif key == '^N225':
                    if datetime.now().hour >= 16:
                        if datetime.now().isoweekday()!=1:
                            returned_dict[key + '_复盘'] = {
                                '涨跌': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] -
                                      data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] -
                                              data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1]) /
                                             data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1], 2),
                                '收盘绝对数': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1],
                                '分时图数据': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')]
                            }
                        else:
                            returned_dict[key + '_复盘'] = {
                                '涨跌': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] -
                                      data[key].loc[(end_date - timedelta(3)).strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] -
                                              data[key].loc[(end_date - timedelta(3)).strftime('%Y-%m-%d')].Close[-1]) /
                                             data[key].loc[(end_date - timedelta(3)).strftime('%Y-%m-%d')].Close[-1], 2),
                                '收盘绝对数': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1],
                                '分时图数据': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')]
                            }
                    else:
                        returned_dict[key + '_晨会'] = {
                            '涨跌': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                  data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1],
                            '涨跌幅': round((data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                          data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                         data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                            '收盘绝对数': minutes_data.Close[-1],
                            '分时图数据': minutes_data
                        }
                elif key == '^TNX':
                    if end_date.isoweekday() not in [1, 5]:
                        if datetime.now().hour>=6 and datetime.now().hour<16:
                            returned_dict[key + '_晨会'] = {
                                '涨跌': data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                      data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] - \
                                              data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                             data[key].loc[start_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                '收盘绝对数': minutes_data.Close[-1],
                                '分时图数据': minutes_data
                            }
                    else:
                        if datetime.now().hour >= 6 and datetime.now().hour < 16:
                            returned_dict[key + '_晨会'] = {
                                '涨跌': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] - \
                                      data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1],
                                '涨跌幅': round((data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1] - \
                                              data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1]) / \
                                             data[key].loc[end_date.strftime('%Y-%m-%d')].Close[-1] * 100, 2),
                                '收盘绝对数': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')].Close[-1],
                                '分时图数据': data[key].loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')]
                            }
                else:
                    print('%s不需要提取数据' % code)
        else:
            raise RuntimeError('没有抓到任何数据!')

        return returned_dict, daily_data

    @staticmethod
    def get_wind_k_plot_code(end_date, code_list=None):
        end_date = pd.to_datetime(end_date)
        if code_list is None:
            code_list = {'USDX.FX': '美元指数', 'USDCNH.FX': '美元兑换离岸人名币', 'SX5E.GI': '斯托克50'}

        temp_data = w.wset("sectorconstituent", "date=%s;sectorid=1000015510000000" % end_date.strftime('%Y-%m-%d'))
        filter_data = pd.DataFrame(np.transpose(temp_data.Data), columns=temp_data.Fields)
        filter_list = ['T%s' % str(end_date.year)[-2:], 'TS%s' % str(end_date.year)[-2:],
                       'CNJ%s' % str(end_date.year)[-2:]]
        filter_contract = filter_data[
            filter_data.sec_name.str.contains('|'.join([x for x in filter_list]))][['wind_code', 'sec_name']]
        for code, name in zip(filter_contract.wind_code, filter_contract.sec_name):
            code_list[code] = name

        # taking data from wind
        # final_data = {}
        # for contract in code_list:
        #     temp_data=w.wsd("%s" % contract, "open,high,low,close,pct_chg,chg", "%s" % start_date.strftime('%Y-%m-%d'),
        #           "%s" % end_date.strftime('%Y-%m-%d'), "")
        #     data=pd.DataFrame(np.transpose(temp_data.Data),columns=temp_data.Fields,index=temp_data.Times)
        #     final_data[contract]=data

        return code_list

    @staticmethod
    def get_fixed_data_from_wind(wind_code, data_col, start_date, end_date):
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        if isinstance(wind_code, list):
            wind_code = ','.join([x for x in wind_code])
        else:
            wind_code = wind_code
        if isinstance(data_col, list):
            data_col = ','.join([x for x in data_col])
        else:
            data_col = data_col
        temp_data = w.wsd("%s" % wind_code, data_col,
                          "%s" % start_date.strftime('%Y-%m-%d'),
                          "%s" % end_date.strftime('%Y-%m-%d'), "")
        data = pd.DataFrame(np.transpose(temp_data.Data),
                            index=temp_data.Times, columns=temp_data.Fields)
        return data

    @staticmethod
    def get_fred_data(fred_code, end_date, init=True):
        end_date = pd.to_datetime(end_date)
        if init:
            import fred
            fred.key(my_fred_api_key)
        original_data = pd.DataFrame(fred.observations(fred_code)['observations']).set_index('date')
        original_data.index = pd.to_datetime(original_data.index)
        start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
        return_data = original_data.loc[start_date:end_date]
        return_data = return_data.replace('.', np.nan).dropna()
        return return_data.value.apply(float)

    @staticmethod
    def get_treasury_bond_YTM(fred_code, end_date, init=True):
        end_date = pd.to_datetime(end_date)
        if init:
            import fred
            fred.key(my_fred_api_key)

        original_data = pd.DataFrame(fred.observations(fred_code)['observations']).set_index('date')
        original_data.index = pd.to_datetime(original_data.index)
        start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
        return_data = original_data.loc[start_date:end_date]
        return_data = return_data.replace('.', np.nan).dropna()
        return return_data.value.apply(float)

    @staticmethod
    def get_currency_output(end_date=None):
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = pd.to_datetime(end_date)
        start_date = end_date - timedelta(60)
        # code is fixed
        temp_data = w.edb("M0041372,M0062600,M6217188,M0329542,M5528819",
                          "%s" % start_date.strftime('%Y-%m-%d'), "%s" % end_date.strftime('%Y-%m-%d'),
                          "Fill=Previous")
        data = pd.DataFrame(np.transpose(temp_data.Data), index=temp_data.Times,
                            columns=temp_data.Codes)
        data.columns = ['逆回购7天', '逆回购到期', 'TMLF投放',
                        'MLF投放', 'MLF收回']

        return data

    @staticmethod
    def get_bond_ytm_from_wind(end_date=None):
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = pd.to_datetime(end_date)
        start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
        temp_data = w.edb("M1001654,M1001647", "%s" % start_date.strftime('%Y-%m-%d'),
                          "%s" % end_date.strftime('%Y-%m-%d'), "Fill=Previous")
        data = pd.DataFrame(np.transpose(temp_data.Data), index=temp_data.Times,
                            columns=temp_data.Codes)
        return data

    @staticmethod
    def get_wind_tick_data(code, start_date, end_date):
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        temp_data = w.wsi("%s" % code, "close,open", "%s 09:00:00" % start_date.strftime('%Y-%m-%d'),
                          "%s 15:59:00" % end_date.strftime('%Y-%m-%d'), "")
        data = pd.DataFrame(np.transpose(temp_data.Data), index=temp_data.Times, columns=temp_data.Fields)
        return data.fillna(method='ffill')

    @staticmethod
    def get_yahoo_daily_data(code, start_date, end_date):
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        model = yf.Ticker(code)
        his_data = model.history(start=start_date, end=end_date)
        his_data.index = [x.tz_localize('Asia/Shanghai') for x in his_data.index]
        return his_data

    @staticmethod
    def get_north_south_data_from_wind(start_date, end_date):
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        # top line
        temp_data = w.wset("shhkactivitystock", "startdate=%s;enddate=%s;direction=south" %
                           (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        south = pd.DataFrame((np.transpose(temp_data.Data)), columns=temp_data.Fields)
        # north
        temp_data = w.wset("shhkactivitystock", "startdate=%s;enddate=%s;direction=north" %
                           (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        north = pd.DataFrame((np.transpose(temp_data.Data)), columns=temp_data.Fields)

        cols = ['代码', '证券名', '日期', '类型', '排名', '买卖总额', '买入', '卖出', '成交净价', '收盘价',
                '货币', '涨跌', '涨跌幅', '换手率', '市盈率', '市净率', '交易所行业', '行业']
        # to excel
        south.columns = cols
        north.columns = cols

        south = south.set_index('日期')
        north = north.set_index('日期')

        south.index = [x.strftime('%Y%m%d') for x in south.index]
        north.index = [x.strftime('%Y%m%d') for x in north.index]

        south.index.name = '日期'
        north.index.name = '日期'

        file_path = os.path.join(based_file, 'morning_temp', datetime.today().strftime('%Y-%m-%d'),
                                 'south_north_data.xlsx')
        xlsx = XlsxSaver(north.reset_index(), file_path, 'north')
        for col in cols:
            if col == '交易所行业':
                xlsx.set_width(col, 25)
            else:
                xlsx.set_width(col, 15)
        xlsx.save()

        xlsx = XlsxSaver(south.reset_index(), file_path, 'south')
        for col in cols:
            if col == '交易所行业':
                xlsx.set_width(col, 25)
            else:
                xlsx.set_width(col, 15)
        xlsx.save()

        excel2img.export_img(os.path.join(based_file, 'morning_temp', datetime.today().strftime('%Y-%m-%d'),
                                          'south_north_data.xlsx'),
                             os.path.join(based_file, 'morning_temp', datetime.today().strftime('%Y-%m-%d'),
                                          'south.png'),
                             '', 'south!A1:R41')

        excel2img.export_img(os.path.join(based_file, 'morning_temp', datetime.today().strftime('%Y-%m-%d'),
                                          'south_north_data.xlsx'),
                             os.path.join(based_file, 'morning_temp', datetime.today().strftime('%Y-%m-%d'),
                                          'north.png'),
                             '', 'north!A1:R41')

    @staticmethod
    def get_activaty_contract_for_SIG(code, pre_format=None, pos_format=None,
                                      end_date=None, use_format=False):
        if isinstance(pre_format, list) and pre_format is not None:
            raise TypeError('code_fromat仅接受单个合约前缀!')
        if end_date is None:
            end_date = pd.to_datetime(datetime.today())
        else:
            end_date = pd.to_datetime(end_date)

        if use_format:
            year = str(end_date.year)
            code = [pre_format + chr(num) + year[-2:] + pos_format for num in range(65, 91)]

            max_vol = 0
            activative_contract = None
            for contract in code:
                model = yf.Ticker(contract)
                try:
                    temp_data = model.history()
                    if temp_data.iloc[-1]['Volume'] > max_vol:
                        max_vol = temp_data.Volume.values
                        activative_contract = contract
                except Exception as e:
                    print('%s合约存在错误:%s' % (contract, e))
        return activative_contract


    @staticmethod
    def domestic_insurance_index(end_date):
        wind_code=['601318.SH','601601.SH','000016.SH']
        end_date=pd.to_datetime(end_date)
        start_date=pd.to_datetime(datetime(end_date.year-1,end_date.month,end_date.day))
        temp_data=w.wsd("%s" % ','.join([x for x in wind_code]),
                        "pct_chg", "%s" % start_date.strftime('%Y-%m-%d'),
                        "%s" % end_date.strftime('%Y-%m-%d'), "")

        data=pd.DataFrame(np.transpose(temp_data.Data),columns=temp_data.Codes,
                          index=temp_data.Times)/100
        data['index_value']=(data['601318.SH']*0.7+data['601601.SH']*0.3)-data['000016.SH']
        data['Close']=(data['index_value']+1).cumprod()

        # 做出模拟K线的
        data['Open']=data['Close'].shift()
        data['High']=np.maximum(data['Open'],data['Close'])
        data['Low']=np.minimum(data['Open'],data['Close'])
        return data[['Open','High','Low','Close']]

    @staticmethod
    def main(end_date=None, period='4d', interval='1m'):
        if end_date is None:
            wind_code_name_dict = MorningAfterMetting.get_main_global_contracts(end_date=datetime.today())
            returned_df, daily_data = MorningAfterMetting.get_contract_data(wind_code_name_dict, datetime.now(),
                                                                            period, interval)
        else:
            wind_code_name_dict = MorningAfterMetting.get_main_global_contracts(date=pd.to_datetime(end_date).date(),
                                                                                )
            returned_df, daily_data = MorningAfterMetting.get_contract_data(wind_code_name_dict,
                                                                            pd.to_datetime(end_date).to_pydatetime(),
                                                                            period, interval)

        # plot candle plots1
        # code_list = MorningAfterMetting.get_wind_k_plot_code(end_date)
        if datetime.now().hour > 6 and datetime.now().hour < 16:
            # code_list = {'USDX.FX': '美元指数', 'USDCNH.FX': '美元兑换离岸人名币', 'SX5E.GI': '斯托克50欧元'}
            code_list = {'USDX.FX': '美元指数','USDCNH.FX': '美元兑换离岸人民币',
                         'BTC.CME':'比特币期货','CNYJPY.FX':'人民币兑换日元汇率',
                         'CA00.LME':'连续伦敦铜','C.DCE':'玉米'
                         }
            for code in code_list:
                print(code)
                MorningAfterMetting.plot_candle(wind_code=code, contract_name=code_list[code],
                                                start_date=datetime(end_date.year - 1, end_date.month,
                                                                    end_date.day).strftime(
                                                    '%Y-%m-%d'),
                                                end_date=(end_date + timedelta(1)).strftime('%Y-%m-%d'))

        if datetime.now().hour >= 16:
            code_list = {'T2106.CFE': 'T2106', 'TS2106.CFE': 'TS2106'}
            # code_list = [x for x in code_list if 'T%s' % str(end_date.year)[-2:] in x
            #              or 'TS%s' % str(end_date.year)[-2:] in x]
            for code in code_list:
                if ('TS%s' % str(end_date.year)[-2:] not in code) and \
                        ('T%s' % str(end_date.year)[-2:] not in code):
                    MorningAfterMetting.plot_candle(wind_code=code, contract_name=code,
                                                    start_date=datetime(end_date.year - 1, end_date.month,
                                                                        end_date.day),
                                                    end_date=(end_date + timedelta(1)).strftime(
                                                        '%Y-%m-%d'))
                else:
                    if 'TS%s' % str(end_date.year)[-2:] in code:
                        MorningAfterMetting.plot_candle(wind_code=code,
                                                        contract_name='TS_2years',
                                                        start_date=datetime(end_date.year - 1, end_date.month,
                                                                            end_date.day),
                                                        end_date=(end_date + timedelta(1)).strftime(
                                                            '%Y-%m-%d'))
                    else:
                        MorningAfterMetting.plot_candle(wind_code=code,
                                                        contract_name='T_10years',
                                                        start_date=datetime(end_date.year - 1, end_date.month,
                                                                            end_date.day),
                                                        end_date=(end_date + timedelta(1)).strftime(
                                                            '%Y-%m-%d'))

        # chinese bond minuts plot
        chinese_bond_list = [x for x in code_list if 'T' in x]
        if datetime.now().hour >= 16:
            if datetime.now().isoweekday()!=1:
                for bond_code in chinese_bond_list:
                    data = MorningAfterMetting.get_wind_tick_data(bond_code, end_date, end_date + timedelta(1))
                    if 'TS%s' % str(end_date.year)[-2:] in bond_code:
                        MorningAfterMetting.plot_curve(data.loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')], 'TS_2years.CFE',
                                                       '收盘价', '时间(亚洲/上海时区)',
                                                       data.close[-1],
                                                       data.close[-1] -
                                                       data.loc[end_date.strftime('%Y-%m-%d')].close[-1],
                                                       ((data.close[-1] - data.loc[end_date.strftime('%Y-%m-%d')]
                                                         .close[-1]) / data.close[-1]) * 100,
                                                       plot_col='close')
                    else:
                        MorningAfterMetting.plot_curve(data.loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')], 'T_10years.CFE',
                                                       '收盘价', '时间(亚洲/上海时区)',
                                                       data.close[-1],
                                                       data.close[-1] -
                                                       data.loc[end_date.strftime('%Y-%m-%d')].close[-1],
                                                       ((data.close[-1] - data.loc[end_date.strftime('%Y-%m-%d')]
                                                         .close[-1]) / data.close[-1]) * 100,
                                                       plot_col='close')
            else:
                for bond_code in chinese_bond_list:
                    data = MorningAfterMetting.get_wind_tick_data(bond_code, end_date-timedelta(2), end_date + timedelta(1))
                    if 'TS%s' % str(end_date.year)[-2:] in bond_code:
                        MorningAfterMetting.plot_curve(data.loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')],
                                                       'TS_2years.CFE',
                                                       '收盘价', '时间(亚洲/上海时区)',
                                                       data.close[-1],
                                                       data.close[-1] -
                                                       data.loc[(end_date-timedelta(2)).strftime('%Y-%m-%d')].close[-1],
                                                       ((data.close[-1] - data.loc[(end_date-timedelta(2)).strftime('%Y-%m-%d')]
                                                         .close[-1]) / data.loc[(end_date-timedelta(2)).strftime('%Y-%m-%d')] .close[-1]) * 100,
                                                       plot_col='close')
                    else:
                        MorningAfterMetting.plot_curve(data.loc[(end_date + timedelta(1)).strftime('%Y-%m-%d')],
                                                       'T_10years.CFE',
                                                       '收盘价', '时间(亚洲/上海时区)',
                                                       data.close[-1],
                                                       data.close[-1] -
                                                       data.loc[(end_date-timedelta(2)).strftime('%Y-%m-%d')].close[-1],
                                                       ((data.close[-1] - data.loc[(end_date-timedelta(2)).strftime('%Y-%m-%d')]
                                                         .close[-1]) / data.loc[(end_date-timedelta(2)).strftime('%Y-%m-%d')] .close[-1]) * 100,
                                                       plot_col='close')

        # plot candle plots2
        if datetime.now().hour < 16 and datetime.now().hour >= 6:
            for code in daily_data:
                print(code)
                if code in ['ESM21.CME', 'NQM21.CME','CNH=X']:
                    continue
                temp_data = daily_data[code]
                if datetime.now().isoweekday() not in [6, 7]:
                    MorningAfterMetting.plot_candle(wind_code_name_dict[code.split('_')[0]],
                                                    wind_code_name_dict[code.split('_')[0]],
                                                    (end_date - timedelta(120)).strftime('%Y-%m-%d'),
                                                    end_date.strftime('%Y-%m-%d'),
                                                    False, temp_data, based_file)
                else:
                    MorningAfterMetting.plot_candle(wind_code_name_dict[code.split('_')[0]],
                                                    wind_code_name_dict[code.split('_')[0]],
                                                    (end_date - timedelta(120)).strftime('%Y-%m-%d'),
                                                    (end_date + timedelta(1)).strftime('%Y-%m-%d'),
                                                    False, temp_data, based_file)
        elif datetime.now().hour >= 16:
            code_list = ['^N225', 'GCM21.CMX', 'CLM21.NYM', 'ES=F', 'YM=F', 'NQ=F',
                         'ZT=F', 'ZN=F']
            for code in code_list:
                temp_data = daily_data[code]
                MorningAfterMetting.plot_candle(wind_code_name_dict[code.split('_')[0]],
                                                wind_code_name_dict[code.split('_')[0]],
                                                (end_date - timedelta(120)).strftime('%Y-%m-%d'),
                                                (end_date + timedelta(1)).strftime('%Y-%m-%d'),
                                                False, temp_data, based_file)
        # plot minute plots
        # taking morning code
        if datetime.now().hour < 16 and datetime.now().hour >= 6:
            # if datetime.now().hour < 23:
            morning_list = ['^VIX_晨会', 'TIP_晨会', '^DJI_晨会', '^IXIC_晨会', '^GSPC_晨会', '^FTSE_晨会',
                            '^GDAXI_晨会', 'GCM21.CMX_晨会', 'CLM21.NYM_晨会', 'ES=F_晨会', 'YM=F_晨会', 'NQ=F_晨会',
                            'ZT=F_晨会', 'ZN=F_晨会', '^N225_晨会', '^STOXX50E_晨会', 'CNH=X_晨会']
            for code in morning_list:
                print(code,':',returned_df[code]['分时图数据'].index[-1],
                      returned_df[code]['收盘绝对数'],
                      returned_df[code]['涨跌'],
                      returned_df[code]['涨跌幅'])
                MorningAfterMetting.plot_curve(
                    returned_df[code]['分时图数据']['Close'],
                    code.replace('_晨会', ''), '收盘价', '时间(亚洲/上海时区)',
                    returned_df[code]['收盘绝对数'],
                    returned_df[code]['涨跌'],
                    returned_df[code]['涨跌幅']
                )
            #
        elif datetime.now().hour >= 16 and datetime.now().isoweekday() not in [6, 7]:
            afternoon_list = ['^N225_复盘', 'ZT=F_复盘', 'ZN=F_复盘', 'GCM21.CMX_复盘', 'CLM21.NYM_复盘',
                              'ES=F_复盘', 'YM=F_复盘', 'NQ=F_复盘']
            for code in afternoon_list:
                MorningAfterMetting.plot_curve(
                    returned_df[code]['分时图数据']['Close'],
                    code.replace('_复盘', ''), '收盘价', '时间(亚洲/上海时区)',
                    returned_df[code]['收盘绝对数'],
                    returned_df[code]['涨跌'],
                    returned_df[code]['涨跌幅']
                )


        else:
            print('当前时间段非8~16点！')

        # solving factor from wind
        if datetime.now().hour >= 16:
            wind_code = ['DR007.IB', '204007.SH']
            for code in wind_code:
                data = MorningAfterMetting.get_fixed_data_from_wind(wind_code=code, data_col=['open', 'close'],
                                                                    start_date=datetime(end_date.year - 1,
                                                                                        end_date.month,
                                                                                        end_date.day),
                                                                    end_date=(end_date + timedelta(1))
                                                                    )
                title_value = data.CLOSE[-1]
                pct_value = data.CLOSE.diff()[-1]
                pct_change = data.CLOSE.pct_change()[-1]
                MorningAfterMetting.plot_curve2(data, code,
                                                '银行质押7天利率', '时间(亚洲/上海时区)',
                                                title_value, pct_value,
                                                pct_change,
                                                save_path=based_file, plot_col='CLOSE')

        # fred infaltion rate 10 years
        if datetime.now().hour > 6 and datetime.now().hour < 16:
            fred_code = 'T10YIE'
            fred_data = MorningAfterMetting.get_fred_data(fred_code, end_date)
            # title_value = fred_data.iloc[-1] - fred_data.iloc[0]

            MorningAfterMetting.plot_curve(fred_data, 'T10YIE',
                                           '美国十年期通涨盈亏平衡率', '时间',
                                           fred_data.iloc[-1], fred_data.iloc[-1] - fred_data.iloc[-2],
                                           (fred_data.iloc[-1] - fred_data.iloc[-2]) / fred_data.iloc[-2],
                                           save_path=based_file)

        # plot line for bond ytm
        if datetime.now().hour >= 16:
            bond_data = MorningAfterMetting.get_bond_ytm_from_wind(end_date + timedelta(1))
            bond_data_diff = bond_data.iloc[:, 0] - bond_data.iloc[:, -1]
            bond_data_diff.name = '10年-2年收益率差'
            MorningAfterMetting.plot_curve2(
                bond_data_diff, '中国10年国债-中国2年国债收益率差值',
                '10年-2年收益率差',
                '时间(亚洲/上海时区)', bond_data_diff.iloc[-1],
                bond_data_diff.iloc[-1] - bond_data_diff.iloc[-2],
                bond_data_diff.pct_change().iloc[-1]
            )

        # nasdaq100/rusell2000
        if datetime.now().hour > 6 and datetime.now().hour < 16:
            nasdaq100_data = MorningAfterMetting.get_yahoo_daily_data('^NDX', datetime(end_date.year - 1,
                                                                                       end_date.month,
                                                                                       end_date.day),
                                                                      end_date + timedelta(1))
            rusell2000_data = MorningAfterMetting.get_yahoo_daily_data('^RUT', datetime(end_date.year - 1,
                                                                                        end_date.month,
                                                                                        end_date.day),
                                                                       end_date + timedelta(1))
            divided_data = nasdaq100_data.Close / rusell2000_data.Close

            MorningAfterMetting.plot_curve(divided_data, 'NASDAQ100VSRUSSELL2000', '收盘价', '时间(亚洲/上海时区)',
                                           divided_data.iloc[-1],
                                           divided_data.iloc[-1] - divided_data.iloc[-2],
                                           (divided_data.iloc[-1] - divided_data.iloc[-2]) / divided_data.iloc[
                                               -2])

        if datetime.now().hour > 6 and datetime.now().hour < 16:
            MorningAfterMetting.get_north_south_data_from_wind(datetime(end_date.year - 1,
                                                                        end_date.month,
                                                                        end_date.day),
                                                               end_date)

        # 美国10年国债收益率折线图
        if datetime.now().hour >= 6 and datetime.now().hour < 16:
            # 10年美国国债收益率
            fred_code = 'DGS10'
            his_data = MorningAfterMetting.get_treasury_bond_YTM(fred_code, end_date)
            MorningAfterMetting.plot_curve2(his_data, '10_Year_YTM_curve', '利率值',
                                            '时间', his_data.iloc[-1], his_data.iloc[-1] - his_data.iloc[-2],
                                            (his_data.iloc[-1] - his_data.iloc[-2]) / his_data.iloc[-2] * 100,
                                            lag_days=2)

            # 2年美国国债收益率
            fred_code = 'DGS2'
            his_data2 = MorningAfterMetting.get_treasury_bond_YTM(fred_code, end_date)
            MorningAfterMetting.plot_curve2(his_data2, '2_Year_YTM_curve', '利率值',
                                            '时间', his_data2.iloc[-1], his_data2.iloc[-1] - his_data2.iloc[-2],
                                            (his_data2.iloc[-1] - his_data2.iloc[-2]) / his_data2.iloc[-2] * 100,
                                            lag_days=2)

            fred_code='IRLTLT01DEM156N'
            his_data3 = MorningAfterMetting.get_treasury_bond_YTM(fred_code, end_date)
            MorningAfterMetting.plot_curve2(his_data3, '10_Year_YTM_curve_Germany', '利率值',
                                            '时间', his_data3.iloc[-1], his_data3.iloc[-1] - his_data3.iloc[-2],
                                            (his_data3.iloc[-1] - his_data3.iloc[-2]) / his_data3.iloc[-2] * 100,
                                            lag_days=2)

            # 两个利率之差
            his_diff = his_data.loc[his_data2.index.to_list()] - his_data2
            his_diff.fillna(method='ffill', inplace=True)
            MorningAfterMetting.plot_curve2(his_diff, '10_year_substract_2_year_YTM_diff', '利率差值',
                                            '时间', his_diff.iloc[-1], his_diff.iloc[-1] - his_diff.iloc[-2],
                                            (his_diff.iloc[-1] - his_diff.iloc[-2]) / his_diff.iloc[-2] * 100,
                                            lag_days=2)

        if datetime.now().hour >= 6 and datetime.now().hour < 16:
            # 10 YEAR USA NOTE - INFATION RATE
            fred_data2 = fred_data.loc[his_data.index]
            result = his_data - fred_data2
            result.name = '10 Year Bond YTM - 10 Year Inflation Rate'
            MorningAfterMetting.plot_curve(result, '10 Year Bond YTM - 10 Year Inflation Rate',
                                           '利率差值', '时间',
                                           result.iloc[-1],
                                           result.iloc[-1] - result.iloc[-2],
                                           np.nan)

        # 新加坡富士50
        # if datetime.now().hour >= 6 and datetime.now().hour < 16:
        #     activative_contract = MorningAfterMetting.get_activaty_contract_for_SIG(
        #         None, 'CN-', '.SI', end_date, True
        #     )
        #     # 获取分时数据
        #     model = yf.Ticker(activative_contract)
        #     data_mins = model.history(period='1d', interval='1m', start=end_date.strftime('%Y-%m-%d'),
        #                               end=(end_date+timedelta(1)).strftime('%Y-%m-%d'))
        #     MorningAfterMetting.plot_curve(data_mins.Close,'A50','合约收盘价格',
        #                                     '时间/新加坡时区',data_mins.iloc[-1].Close,
        #                                    np.nan,np.nan)
        #
        #     # 日K
        #     if datetime.now().hour>=6 and datetime.now().hour<16:
        #         MorningAfterMetting.plot_candle(activative_contract.split('.')[0]+'.SG',
        #                                         'A50期货合约',datetime(end_date.year-1,end_date.month,end_date.day),
        #                                         end_date,True,None,use_chinese_time_zone=True)
        #     else:
        #         MorningAfterMetting.plot_candle(activative_contract.split('.')[0]+'.SG',
        #                                         'A50期货合约',datetime(end_date.year-1,end_date.month,end_date.day),
        #                                         end_date+timedelta(1),True,None,use_chinese_time_zone=True)
        #

        # (中国平安70%+中国太保30%)-上证50
        if datetime.now().hour>=16:
            index_data=MorningAfterMetting.domestic_insurance_index(end_date)
            index_data.index=pd.to_datetime(index_data.index)
            MorningAfterMetting.plot_candle('(平安-人保)-上证','(平安-人保)-上证',
                                            (end_date - timedelta(120)).strftime('%Y-%m-%d'),
                                            (end_date + timedelta(1)).strftime('%Y-%m-%d'),
                                            False,index_data,use_chinese_time_zone=False
                                            )




class WeekendMeeting:
    @staticmethod
    def index_pct_change(code, start_date=None, end_date=datetime.today().date()):
        if isinstance(code, list):
            code = ','.join([x for x in code])
        else:
            code = code
        if start_date is None:
            start_date = datetime(2021, 1, 1)
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        temp_data = w.wsd("%s" % code, "pct_chg", "%s" % start_date.strftime('%Y-%m-%d'),
                          "%s" % end_date.strftime('%Y-%m-%d'), "")
        data = pd.DataFrame(np.transpose(temp_data.Data), index=temp_data.Times, columns=temp_data.Codes)
        return data

    @staticmethod
    def plot_curve(data, picture_name, ylabel_name, xlabel_name,
                   plot_col=None, save_path=based_file):

        def format_date(x, pos=None):
            if x < 0 or x > len(data.index) - 1:
                return ''
            return data.index[int(x)]

        data.index = pd.to_datetime(data.index).strftime('%Y-%m-%d')
        # data.index=[x.strftime('%Y-%m-%d') for x in data.index]
        if not isinstance(data, pd.DataFrame):
            fig = plt.figure(figsize=(30, 15))
            ax = fig.add_subplot(111)
            x = range(len(data.index))
            if plot_col is None:
                ax.plot(x, data.values, 'o-', label='Close')
            else:
                ax.plot(x, data[plot_col])
            plt.title('报告日:%s,%s' % (data.index[-1], picture_name))
            ax.set_ylabel(ylabel_name)
            ax.set_xlabel(xlabel_name)
            ax.grid(True)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(48))
            # plt.xticks(range(0, len(data.index), 8), [x.strftime('%Y-%m-%d %H:%M:%S') for x in
            #                                           data.index[list(range(0, len(data.index),
            #                                                                 8))]],
            #            rotation=45)
            fig.autofmt_xdate()
        else:
            data[plot_col].plot(figsize=(30, 15), secondary_y=True,
                                fontsize=20, x_compat=True,
                                mark_right=False,
                                legend=False, grid=True
                                )
            plt.ylabel(ylabel_name, fontsize=20)
            plt.xlabel(xlabel_name, fontsize=20)
            plt.title('报告日:%s,%s' % (data.index[-1], picture_name))
            plt.grid(True)
            plt.legend(loc='best')
        if save_path is None:
            plt.show()
        else:
            mkdir(os.path.join(based_file, 'weekend_temp', datetime.today().strftime('%Y-%m-%d')))
            temp_path = os.path.join(based_file, 'weekend_temp', datetime.today().strftime('%Y-%m-%d'))
            save_path = os.path.join(temp_path, '%s.png' % (picture_name))
            plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def copy_funds_plot(key_words, read_path, save_file_name='weekend_temp'):
        whole_files = []
        for root, dir, files in os.walk(read_path):
            for file in files:
                whole_files.append(os.path.join(root, file))
        try:
            copy_to_jarvisoutput(key_words, save_file_name=save_file_name, read_based_folder=read_path)
        except Exception as e:
            raise e

    @staticmethod
    def main(start_date=None, end_date=None,
             key_words=['Fund_issued_analyst.png', '限售解禁VS高层减持规模.png'],
             save_file_name='weekend_temp'):
        if pd.to_datetime(end_date).isoweekday() not in [6, 7]:
            raise ValueError('end date is not weekend!')
        else:
            if datetime.today().isoweekday() == 6:
                end_date = pd.to_datetime(end_date) - timedelta(1)
            elif datetime.today().isoweekday() == 7:
                end_date = pd.to_datetime(end_date) - timedelta(2)
            else:
                end_date = pd.to_datetime(end_date)
        if start_date is None:
            start_date = pd.to_datetime(datetime(2021, 1, 1))
        # main indeies around world
        index_code = ['000001.SH', '399006.SZ', 'HSI.HI', 'N225.GI', 'SPX.GI', 'IXIC.GI', 'SX5E.GI']
        index_pct_change_data = WeekendMeeting.index_pct_change(index_code, start_date, end_date)
        WeekendMeeting.plot_curve(((index_pct_change_data / 100).cumsum() + 1).fillna(method='ffill'),
                                  '全球重要指数今年以来表现', '指数累计收益率(今年以来),起点为1',
                                  '时间(国内时区)', index_pct_change_data.columns.to_list())

        WeekendMeeting.copy_funds_plot(key_words, os.path.join('E:\\Jarvis_Temp', 'Fund_Issue_Reports'),
                                       save_file_name=save_file_name)


if __name__ == '__main__':
    w.start()
    if datetime.now().hour < 24:
        if datetime.today().isoweekday() not in [6, 7]:
            MorningAfterMetting.main(datetime.today() - timedelta(1))
        elif datetime.today().isoweekday() == 6:
            MorningAfterMetting.main(datetime.today() - timedelta(1))
            WeekendMeeting.main(start_date=None, end_date=datetime.today() + timedelta(1))
        elif datetime.today().isoweekday() == 7:
            MorningAfterMetting.main(datetime.today() - timedelta(2))
            WeekendMeeting.main(start_date=None, end_date=datetime.today())
        else:
            pass
