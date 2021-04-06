from WindPy import w
import rqdatac as rq
from rqdatac import query, fundamentals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from pylab import mpl

from datetime import timedelta
from datetime import datetime

import seaborn as sns

from omk.interface import AbstractJob
from omk.core.vendor.RQData import RQData

import os

sns.set(font_scale = 1.5, font = 'SimHei')
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False


def get_stock_code_data_from_rq(date, type = 'CS', market = 'cn', rq_start = False):
    # if rq_start:
    #     RQData.init()
    date = pd.to_datetime(date).date()
    return rq.all_instruments(type = type, date = date, market = market).loc[:, ['order_book_id', 'symbol']]


def get_stock_section_from_rq(source = 'citics_2019', date = None):
    industry = rq.get_industry_mapping(source = source, market = 'cn', date = date)
    industry_dict = {y: str(x) for x, y in zip(industry.first_industry_code, industry.first_industry_name)}
    return industry_dict


def get_stock_under_per_industry(industry_dict, date = None, source = 'citics_2019'):
    if industry_dict is None:
        raise ValueError('industry_dict cannot be empty!')
    whole_stocks_dict = {}
    for code in industry_dict:
        whole_stocks_dict[code] = rq.get_industry(industry = code, source = source, date = date)
    return whole_stocks_dict


# 专门处理非银金融一级行业，拆分成保险和证券行业分类
def solve_nonfinance_industry(whole_stocks_dict, source = 'citics_2019', date = None, level = 2):
    temp = whole_stocks_dict.pop('非银行金融')
    insurance_list = []
    security_list = []
    whole_none_financial = rq.get_instrument_industry(temp, source = source, level = level, date = date)
    for code in whole_none_financial.index.values:
        if '保险' in whole_none_financial.loc[code, 'second_industry_name']:
            insurance_list.append(code)
        elif '证券' in whole_none_financial.loc[code, 'second_industry_name']:
            security_list.append(code)
        else:
            continue
    whole_stocks_dict['保险'] = insurance_list
    whole_stocks_dict['证券'] = security_list
    return whole_stocks_dict


def get_ZX_Index_pct(start_date, end_date):
    temp = w.wsd(
        "CI005001.WI,CI005002.WI,CI005003.WI,CI005004.WI,CI005005.WI,CI005006.WI,CI005007.WI,CI005008.WI,CI005009.WI,CI005010.WI,CI005011.WI,CI005012.WI,CI005013.WI,CI005014.WI,CI005015.WI,CI005016.WI,CI005017.WI,CI005018.WI,CI005019.WI,CI005020.WI,CI005021.WI,CI005023.WI,CI005024.WI,CI005025.WI,CI005026.WI,CI005027.WI,CI005028.WI,CI005029.WI,CI005030.WI",
        "pct_chg", "%s" % start_date, "%s" % end_date, "")

    index_data = pd.DataFrame(np.transpose(temp.Data), index = temp.Times, columns = temp.Codes)
    index_data.index = pd.to_datetime(index_data.index)
    # 获取中信一级行业指数名字-代码映射
    index_name = w.wset("sectorconstituent", "date=%s;sectorid=a39901012e000000" % end_date)
    index_name = {x: y.replace('(中信)', '') for x, y in zip(index_name.Data[1], index_name.Data[-1])}
    index_data.columns = [index_name[x] for x in index_data.columns]
    # 提出中信一级行业中的非银金融，加入二级行业的保险和证券
    temp2 = w.wsd("CI005165.WI,CI005166.WI", "pct_chg", "%s" % start_date, "%s" % end_date, "")
    index_data2 = pd.DataFrame(np.transpose(temp2.Data), index = temp2.Times, columns = temp2.Codes)
    index_data2.columns = ['证券', '保险']
    index_data = pd.concat([index_data, index_data2], axis = 1)
    return index_data


def get_industry_return_median(per_industry_stocks_dict, start_date, end_date):
    industry_change = get_ZX_Index_pct(start_date, end_date)
    industry_dict = dict()
    for industry in per_industry_stocks_dict:
        temp_data = rq.get_price_change_rate(per_industry_stocks_dict[industry], start_date, end_date)
        temp_median = temp_data.iloc[-1].median()
        industry_dict[industry] = {'industry_change': industry_change[industry].iloc[-1],
                                   'industry_median': temp_median * 100}
    return industry_dict


class CitiIndustryAnalysis:
    global start_date, end_date

    @staticmethod
    def get_per_industry_security_codes(source = 'citics_2019', date = datetime.today().strftime('%Y-%m-%d'), level = 2,
                                        data_source = 'excel'):
        whole_industry_dict = get_stock_section_from_rq(source = source, date = date)
        per_industry_stocks_dict = get_stock_under_per_industry(whole_industry_dict, source = source, date = date)
        per_industry_stocks_dict = solve_nonfinance_industry(per_industry_stocks_dict, source = source, date = date,
                                                             level = level)
        # 将rq转换为wind标准代码
        if data_source == 'excel':
            for key in per_industry_stocks_dict:
                per_industry_stocks_dict[key] = [x.replace('XSHG', 'SH') if 'XSHG' in x else x.replace('XSHE', 'SZ') for
                                                 x
                                                 in per_industry_stocks_dict[key]]
        return per_industry_stocks_dict

    def _solve_RZRQ(self, per_industry_stocks_dict, data_source = 'rq',
                    sz_path = 'H:\\RZRQ_result_csv\\sz', sh_path = 'H:\\RZRQ_result_csv\\sh',
                    start_date = None, end_date = None):
        RZRQ_history = pd.DataFrame()
        col_list = []
        for key in per_industry_stocks_dict:
            if data_source == 'rq':
                temp = rq.get_securities_margin(per_industry_stocks_dict[key], start_date = start_date,
                                                end_date = end_date).fillna(0)
                temp1 = temp.margin_balance.sum(axis = 1)
                temp2 = temp.short_balance.sum(axis = 1)
                temp3 = pd.concat([temp1, temp2], axis = 1)
                temp4 = (temp1 + temp2)
                temp3 = pd.concat([temp3, temp4], axis = 1)
                temp3.columns = ['%s_margin_balance' % key, '%s_short_balance' % key, '%s_total_balance' % key]
                RZRQ_history = pd.concat([RZRQ_history, temp3], axis = 1)
            # 此处修改成excel
            elif data_source == 'excel':
                # 分别获取上海和深圳两融最近一次的文件夹文件
                file_list1 = []
                file_list2 = []
                for (root1, dir1, files1), (root2, dir2, files2) in zip(os.walk(sz_path), os.walk(sh_path)):
                    for file1, file2 in zip(files1, files2):
                        file_list1.append(os.path.join(root1, file1))
                        file_list2.append(os.path.join(root2, file2))
                # 提取最大日期下的文件
                file1_date = max([x.split('\\')[-1].split('_')[0] for x in file_list1])
                file2_date = max([x.split('\\')[-1].split('_')[0] for x in file_list2])
                if file1_date != file2_date:
                    raise ValueError('深圳交易数据与上海交易所数据不一致!')
                else:
                    file1_path = [x for x in file_list1 if file1_date in x]
                    file2_path = [x for x in file_list2 if file2_date in x]

                temp = pd.DataFrame()
                for path in file1_path + file2_path:
                    temp_data = pd.read_csv(path).drop(columns = 'Unnamed: 0')
                    temp = pd.concat([temp, temp_data], axis = 0)
                temp.set_index('as_of_date', inplace = True)
                temp.index = pd.to_datetime(temp.index)
                # temp.security_code = [x.replace('SH', 'XSHG') if 'SH' in x else
                #                       x.replace('SZ', 'XSHE') for x in temp.security_code]
                # 提取有效标的物ID
                useful_code = [x for x in per_industry_stocks_dict[key] if x in temp.security_code.values]
                # 根据有效标的物ID直接计算每日总计两融数据
                temp1 = temp[temp.security_code.isin(per_industry_stocks_dict[key]) == True] \
                    [['security_code', 'rz_remained_amounts']].fillna(0).reset_index().groupby('as_of_date').sum()
                # 提取收盘价
                temp2_close = w.wsd(useful_code, "close", start_date.strftime('%Y-%m-%d'),
                                    end_date.strftime('%Y-%m-%d'), "")
                temp2_close = pd.DataFrame(np.transpose(temp2_close.Data), index = pd.to_datetime(temp2_close.Times),
                                           columns = temp2_close.Codes).fillna(0)
                temp2_close.index = pd.to_datetime(temp2_close.index)
                temp2_data = temp[temp.security_code.isin(per_industry_stocks_dict[key]) == True] \
                    [['security_code', 'rq_remained_shares']].fillna(0).groupby('as_of_date')
                # 提取融券剩余份额
                temp2 = pd.DataFrame()
                for index, item in temp2_data:
                    local_temp2 = pd.DataFrame(item.rq_remained_shares.values.transpose(),
                                               index = item.security_code.to_numpy(), columns = [index]).T
                    # 可能存在某一天某个证券不存在于当天的融资融券数据中，则补齐数据为0
                    added_columns = list(set(temp2_close.columns) - set(local_temp2.columns))
                    for col in added_columns:
                        local_temp2[col] = 0
                    temp2 = pd.concat([temp2, local_temp2], axis = 0)
                # 重新排序
                temp2_close = temp2_close[temp2.columns]
                temp2 = pd.DataFrame(np.sum(temp2.values * temp2_close.values, axis = 1),
                                     index = temp2_close.index, columns = ['rq_remained_amounts'])
                temp2.index.name = 'as_of_date'
                temp3 = pd.concat([temp1, temp2], axis = 1)
                temp4 = pd.DataFrame(temp1.to_numpy() + temp2.to_numpy(), index = temp3.index,
                                     columns = ['rzrq_remained_amounts'])
                temp3 = pd.concat([temp3, temp4], axis = 1)
                temp3.columns = ['%s_margin_balance' % key, '%s_short_balance' % key, '%s_total_balance' % key]
                RZRQ_history = pd.concat([RZRQ_history, temp3], axis = 1)
            else:
                raise ValueError('data_source参数仅支持rq/sql!')
        # 计算变化率
        for name in RZRQ_history.columns:
            RZRQ_history['%s_RZRQ_pct' % name] = RZRQ_history[name].pct_change()
            # print(name)

        return RZRQ_history

    @staticmethod
    def plot(RZRQ_history, specify_date, save_path = None, index_data = None):
        RZRQ_history = RZRQ_history.fillna(0)
        # 可视化前期准备
        useful_col1 = [x for x in RZRQ_history.columns if 'total_balance' in x and 'total_balance_RZRQ_pct' not in x]
        useful_col2 = [x for x in RZRQ_history.columns if 'margin_balance' in x and 'margin_balance_RZRQ_pct' not in x]
        useful_col3 = [x for x in RZRQ_history.columns if 'short_balance' in x and 'short_balance_RZRQ_pct' not in x]
        useful_col6 = [x for x in RZRQ_history.columns if 'total_balance_RZRQ_pct' in x]

        # 计算两融每日涨跌幅+提前补充0
        total_history_day_diff = RZRQ_history.loc[:specify_date, useful_col1].diff().iloc[-1]
        total_history_day_diff.index = [x.split('_')[0] for x in total_history_day_diff.index]
        margin_history_day_diff = RZRQ_history.loc[:specify_date, useful_col2].diff().iloc[-1]
        margin_history_day_diff.index = [x.split('_')[0] for x in margin_history_day_diff.index]
        short_history_day_diff = RZRQ_history.loc[:specify_date, useful_col3].diff().iloc[-1]
        short_history_day_diff.index = [x.split('_')[0] for x in short_history_day_diff.index]

        # 计算3个两融数据的最后一天和倒数第二天的差值
        total_history_diff = total_history_day_diff.sum()
        margin_history_diff = margin_history_day_diff.sum()
        short_history_diff = short_history_day_diff.sum()

        total_history_pct = RZRQ_history.loc[pd.to_datetime(specify_date).strftime('%Y-%m-%d'),
                                             useful_col6].sort_values(ascending = False)
        # 按照融资余额重新排序两融余额展示顺序
        print('融资余额变化顺序是: ')
        for i, industry in enumerate([x.split('_')[0] for x in
                                      margin_history_day_diff.sort_values(ascending = False).index], start = 1):
            print(i, ':', industry)
        # 根据融资余额变化重新排序两融余额涨跌幅
        total_history_pct = total_history_pct[
            [x.split('_')[0] + '_total_balance_RZRQ_pct' for x in
             margin_history_day_diff.sort_values(ascending = False).index]]
        short_history_day_diff = short_history_day_diff[
            [x.split('_')[0] for x in
             margin_history_day_diff.sort_values(ascending = False).index]]
        index_data = index_data[[x.split('_')[0] for x in
                                 margin_history_day_diff.sort_values(ascending = False).index]].loc[
                     pd.to_datetime(specify_date).strftime('%Y-%m-%d'), :]

        max_change = round(index_data.max(), 1)
        min_change = round(index_data.min(), 1)
        y_right_lim = [min_change - 0.2, max_change + 0.1]
        y_left_lim = [round(margin_history_day_diff.min() / 1e8 - 0.1, 1),
                      round(margin_history_day_diff.max() / 1e8 + 2, 1)]
        title_size = 20
        text_size = 15

        # 可视化
        sns.set(font_scale = 1.5, font = 'SimHei')
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize = (32, 22))
        ax1 = ax.twinx()
        for i, name in enumerate(total_history_pct.index):
            name1 = name.split('_')[0]
            # height1 = total_history_pct.loc[name] * 100 - 0.5 if total_history_pct.loc[name] < 0 else \
            #     total_history_pct.loc[name] * 100 + 0.3
            height1 = margin_history_day_diff.loc[name1] / 1e8
            # height2 = total_history_pct.loc[name] * 100 - 0.6 if total_history_pct.loc[name] < 0 else \
            #     total_history_pct.loc[name] * 100 + 0.2
            height2 = margin_history_day_diff.loc[name1] / 1e8 - 0.2
            num1 = str((margin_history_day_diff.loc[name1] / 1e8).round(1))
            num2 = str((short_history_day_diff.loc[name1] / 1e8).round(1))
            # print(name, num1, num2)
            ax.text(i - 0.3, height1, num1, size = text_size)
            ax.text(i - 0.3, height2, num2, size = text_size)
        ax.set_ylim(y_left_lim)
        ax.set_ylabel('融资余额变化 单位：亿元\n 柱状图上方数值单位: 亿元 \n 显示顺序：融资余额变化/融券余额变化')

        ax1.bar(index_data.index, index_data.values,
                width = 0.9, color = ['blue' if x > 0 else 'orange' for x in index_data],
                alpha = 0.35, label = '两融余额变化')
        ax.set_xticklabels([x.split('_')[0] for x in index_data.index], rotation = 90, fontsize = 18,
                           color = 'navy')
        ax1.spines['bottom'].set_position(('data', 0))

        ax1.set_ylim(y_right_lim)
        ax1.set_ylabel('行业涨跌幅 单位：%')

        total_history_diff = \
        RZRQ_history.loc[: pd.to_datetime(specify_date).strftime('%Y-%m-%d'), useful_col1].iloc[-2:, :].diff().iloc[
            -1].sum()
        plt.title(str('中信一级行业%s' % pd.to_datetime(specify_date).strftime('%Y-%m-%d')) +

                  str('两融余额变化%.1f' % (total_history_diff / 1e8)) + ' ' +
                  str('融资余额变化%.1f' % (margin_history_diff / 1e8)) + ' ' +
                  str('融券余额变化%.1f' % (short_history_diff / 1e8)) + ' 单位(亿元)',
                  size = title_size, loc = 'center')
        if save_path is not None:
            plt.savefig(
                os.path.join(save_path, '%s_两融余额行业涨跌幅柱状图.png' % pd.to_datetime(specify_date).strftime('%Y-%m-%d')),
                bbox_inches = 'tight')
        else:
            plt.show()

        sns.set(font_scale = 1.5, font = 'SimHei')
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax6 = plt.subplots(figsize = (32, 22))
        index_data = index_data[margin_history_day_diff.index]
        ax6.scatter(margin_history_day_diff / 1e8, index_data, s = 200)

        industry = margin_history_day_diff.index.values

        for i, txt in enumerate(industry):
            ax6.annotate(txt, (margin_history_day_diff[txt] / 1e8 + 0.05, index_data[txt]))

        plt.axhline(y = 0, ls = "-", c = "red", lw = 5)  # 添加水平直线
        plt.axvline(x = 0, ls = "-", c = "red", lw = 5)  # 添加垂直直线

        # 添加四个区域文字区分
        ax6.annotate('止盈', (margin_history_day_diff.min() / 1e8 + 0.5, 0.2),
                     fontsize = 40, color = 'red')
        ax6.annotate('加仓', (margin_history_day_diff.max() / 1e8 - 1.5, 0.2),
                     fontsize = 40, color = 'red')
        ax6.annotate('抄底', (margin_history_day_diff.max() / 1e8 - 1.5, -0.2),
                     fontsize = 40, color = 'red')
        ax6.annotate('减仓', (margin_history_day_diff.min() / 1e8 + 0.5, -0.2),
                     fontsize = 40, color = 'red')

        ax6.grid(True)
        ax6.set_xlabel('融资余额变化 单位:亿元')
        ax6.set_ylabel('行业涨跌幅 单位: %')
        ax6.set_title(str('融资余额变化 VS 行业涨跌幅 散点图\n' + '中信一级行业%s' % pd.to_datetime(specify_date).strftime('%Y-%m-%d')) +

                      str('两融余额变化%.1f' % (total_history_diff / 1e8)) + ' ' +
                      str('融资余额变化%.1f' % (margin_history_diff / 1e8)) + ' ' +
                      str('融券余额变化%.1f' % (short_history_diff / 1e8)) + ' 单位(亿元)',
                      size = title_size, loc = 'center')
        if save_path is not None:
            plt.savefig(
                os.path.join(save_path, '%s_融资余额变化VS行业涨跌幅散点图.png' % pd.to_datetime(specify_date).strftime('%Y-%m-%d')),
                bbox_inches = 'tight')
        else:
            plt.show()

        sns.set(font_scale = 1.5, font = 'SimHei')
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax7 = plt.subplots(figsize = (32, 22))
        short_history_day_diff = short_history_day_diff[margin_history_day_diff.index]
        ax7.scatter(margin_history_day_diff / 1e8, short_history_day_diff / 1e8, s = 200)
        industry = margin_history_day_diff.index.values

        for i, txt in enumerate(industry):
            ax7.annotate(txt, (margin_history_day_diff[txt] / 1e8, short_history_day_diff[txt] / 1e8))

        # 再x=0 y=0的地方加入两条直线
        plt.axhline(y = 0, ls = "-", c = "red", lw = 5)  # 添加水平直线
        plt.axvline(x = 0, ls = "-", c = "red", lw = 5)  # 添加垂直直线

        # 添加四个区域文字区分
        ax7.annotate('看空', (margin_history_day_diff.min() / 1e8 + 0.5, 0.2),
                     fontsize = 40, color = 'red')
        ax7.annotate('流动性上升', (margin_history_day_diff.max() / 1e8 - 1.5, 0.2),
                     fontsize = 40, color = 'red')
        ax7.annotate('看多', (margin_history_day_diff.max() / 1e8 - 1.5, 0 - 0.5),
                     fontsize = 40, color = 'red')
        ax7.annotate('流动性下降', (margin_history_day_diff.min() / 1e8 + 0.5, 0 - 0.5),
                     fontsize = 40, color = 'red')
        ax7.add_line(Line2D(margin_history_day_diff / 1e8, margin_history_day_diff / 1e8, linewidth = 1, color = 'red'))

        ax7.set_xlabel('融资余额变化 单位: 亿元')
        ax7.set_ylabel('融券余额变化 单位: 亿元')
        ax7.grid(True)
        ax7.set_title(str('融资余额变化 VS 融券余额变化 散点图\n' + '中信一级行业%s' % pd.to_datetime(specify_date).strftime('%Y-%m-%d')) +

                      str('两融余额变化%.1f' % (total_history_diff / 1e8)) + ' ' +
                      str('融资余额变化%.1f' % (margin_history_diff / 1e8)) + ' ' +
                      str('融券余额变化%.1f' % (short_history_diff / 1e8)) + ' 单位(亿元)',
                      size = title_size, loc = 'center')
        if save_path is not None:
            plt.savefig(
                os.path.join(save_path, '%s_融资余额变化VS融券余额变化散点图.png' % pd.to_datetime(specify_date).strftime('%Y-%m-%d')),
                bbox_inches = 'tight')
        else:
            plt.show()

    @staticmethod
    def industry_return_median_plot(industry_dict, date, save_path):
        # 首先分别提取行业中值和行业涨幅
        median_list = []
        change_list = []
        for industry in industry_dict:
            median_list.append(industry_dict[industry]['industry_median'])
            change_list.append(industry_dict[industry]['industry_change'])

        sns.set(font_scale = 1.5, font = 'SimHei')
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize = (32, 22))
        ax.scatter(change_list, median_list, s = 200)

        for i, txt in enumerate(industry_dict.keys()):
            ax.annotate(txt, (change_list[i] + 0.05, median_list[i]), fontsize = 15)

        ax.grid(True)
        ax.set_xlabel('行业涨跌幅 单位: %')
        ax.set_ylabel('行业涨跌幅中位数 单位: %')
        ax.set_title('%s_中信行业涨跌幅 行业中位数 散点图' % pd.to_datetime(date).strftime('%Y-%m-%d'), size = 18, loc = 'center')

        # 添加直线
        ax.add_line(Line2D(change_list, change_list, linewidth = 1, color = 'red'))
        # 再x=0 y=0的地方加入两条直线
        plt.axhline(y = 0, ls = "-", c = "red", lw = 5)  # 添加水平直线
        plt.axvline(x = 0, ls = "-", c = "red", lw = 5)  # 添加垂直直线
        # 添加文字
        ax.annotate('大盘', (max(change_list) - 0.2, min(median_list) + 0.3),
                    fontsize = 40, color = 'red')
        ax.annotate('小盘', (min(change_list) + 0.2, max(median_list) - 0.3),
                    fontsize = 40, color = 'red')

        if save_path is not None:
            plt.savefig(
                os.path.join(save_path, '%s_中信行业涨跌幅 行业中位数 散点图.png' % pd.to_datetime(date).strftime('%Y-%m-%d')),
                bbox_inches = 'tight')
        else:
            plt.show()

    @staticmethod
    def take_stock_rzrq_to_sql(code_list, start_date, end_date, engine = None):
        security_rzrq_data = rq.get_securities_margin(code_list, start_date = start_date,
                                                      end_date = end_date)
        if security_rzrq_data is not None:
            security_rzrq_data = security_rzrq_data.fillna(0)
        else:
            return
        # 循环处理每一个字段
        columns_list = ['as_of_date', 'security_code',
                        'margin_balance', 'buy_on_margin',
                        'margin_repayment', 'short_balance',
                        'short_balance_quantity', 'short_sell_quantity',
                        'short_repayment_quantity', 'total_balance']
        # 切割数据分类
        local_margin_balance = security_rzrq_data['margin_balance']
        local_buy_on_margin_value = security_rzrq_data['buy_on_margin_value']
        local_short_sell_quantity = security_rzrq_data['short_sell_quantity']
        local_margin_repayment = security_rzrq_data['margin_repayment']
        local_short_balance_quantity = security_rzrq_data['short_balance_quantity']
        local_short_repayment_quantity = security_rzrq_data['short_repayment_quantity']
        local_short_balance = security_rzrq_data['short_balance']
        local_total_balance = security_rzrq_data['total_balance']

        for date in local_margin_balance.index:
            temp_dateframe = pd.DataFrame(columns = columns_list)
            temp_index = local_margin_balance.loc[date].name
            temp_dateframe['margin_balance'] = local_margin_balance.loc[date].to_list()
            temp_dateframe['buy_on_margin'] = local_buy_on_margin_value.loc[date].to_list()
            temp_dateframe['short_sell_quantity'] = local_short_sell_quantity.loc[date].to_list()
            temp_dateframe['margin_repayment'] = local_margin_repayment.loc[date].to_list()
            temp_dateframe['short_balance_quantity'] = local_short_balance_quantity.loc[date].to_list()
            temp_dateframe['short_repayment_quantity'] = local_short_repayment_quantity.loc[date].to_list()
            temp_dateframe['short_balance'] = local_short_balance.loc[date].to_list()
            temp_dateframe['total_balance'] = local_total_balance.loc[date].to_list()
            temp_dateframe['as_of_date'] = [temp_index] * len(temp_dateframe['margin_balance'])
            temp_dateframe['security_code'] = local_margin_balance.loc[date].index.to_list()

            temp_dateframe = temp_dateframe.replace(0, np.nan).dropna(subset = ['margin_balance', 'buy_on_margin'])
            # print(temp_dateframe)
            # 插入到数据库
            # try:
            #     temp_dateframe.to_sql(
            #         RZRQSecurity.table_name(), engine, schema=RZRQSecurity.schema(),
            #         index=False, if_exists='append')
            # except Exception as e:
            #     print(e)

    def main(self, start_date, end_date = None, source = 'citics_2019', data_source = 'excel'):
        if end_date is None:
            if datetime.today().isoweekday() == 1:
                end_date = (datetime.today() - timedelta(3)).strftime('%Y-%m-%d')
            else:
                end_date = (datetime.today() - timedelta(1)).strftime('%Y-%m-%d')
            # end_date='20210106'
            last_day = (datetime.today() - timedelta(10)).strftime('%Y-%m-%d')
            # end_date = RQData.get_trading_dates(last_day, end_date)[-1]
            end_date = datetime.today()
            if end_date.isoweekday() == 7:
                end_date -= timedelta(3)
                # end_date = RQData.get_trading_dates(last_day, end_date)[-2]
        if start_date is None:
            start_date = pd.to_datetime('2021-01-04')
        # else:
        #     start_date = RQData.get_trading_dates(last_day, end_date)[0]
        index_data = get_ZX_Index_pct(start_date, end_date)
        per_industry_stocks_dict = CitiIndustryAnalysis.get_per_industry_security_codes(date = end_date,
                                                                                        data_source = data_source)
        RZRQ_history = self._solve_RZRQ(per_industry_stocks_dict, data_source = data_source,
                                        start_date = pd.to_datetime(start_date),
                                        end_date = end_date)
        CitiIndustryAnalysis.plot(RZRQ_history, end_date, save_path = 'H:\\RZRQ_plot',
                                  index_data = index_data)


if __name__ == '__main__':
    RQData.init()
    w.start()

    # 单独执行
    citi_model = CitiIndustryAnalysis()
    citi_model.main(start_date = '2021-03-30')
    # citi_manager = JobManager()
    # citi_model.register_event(event_bus=citi_manager.event_bus, job_uuid=None, debug=True)
    # citi_manager.event_bus.event_queue_reload(EVENT.AM0930)
    # citi_manager.event_bus.sequential_publish()
