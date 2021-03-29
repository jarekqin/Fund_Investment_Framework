# %%
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

from omk.interface import AbstractJob
from omk.core.vendor.RQData import RQData
from omk.utils.const import FolderName, TODAY, ProcessDocs, ProcessType, EVENT, TIME
from jarvis.utils import FOLDER,mkdir

import rqdatac as rq
from WindPy import w
from omk.events import Event
from omk.toolkit.job_tool import JobManager



class DynamicAndPessive(AbstractJob):
    def __init__(self):
        w.start()
        RQData.init()

    def register_event(self, event_bus, job_uuid, debug):
        if RQData.check_for_trading_date():
            event_bus.add_listener(Event(
                event_type=EVENT.PM0430,
                func=self.main,
                alert=True,
                gap=None,
                p_type=ProcessType.Jarvis,
                des=ProcessDocs.Jarvis_DynamicPessive,
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
                based_file=os.path.join(FOLDER.Syn_save,'dynamicandpessive'),
                date_end=TIME.today(),
                omaka_file_path=os.path.join(FOLDER.Syn_read,'omaka行业.csv'),
                omaka_based_file=os.path.join(FOLDER.hd5_file,'%s_omaka_industry_weights.h5' \
                    % (datetime.today().strftime('%Y-%m-%d')))
            ))

    def main(self, date_end=TIME.yesterday(),
             based_file=os.path.join(FOLDER.Syn_save,'dynamicandpessive'),
             omaka_file_path=None, omaka_based_file=None):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # %%
        date_last = rq.get_previous_trading_date(date_end)
        date_start = date_end
        trading_dates = rq.get_trading_dates(date_start, date_end, market='cn')
        all_instruments = rq.all_instruments(type='CS', market='cn', date=date_end)
        wind_code = all_instruments['order_book_id'].str.replace('XSHG', 'SH').str.replace('XSHE', 'SZ')
        # %%
        example_all = rq.get_price(all_instruments['order_book_id'],
                                   start_date=date_start,
                                   end_date=date_end,
                                   fields=['open', 'close', 'volume', 'total_turnover'],
                                   frequency='5m',
                                   expect_df=True)

        citics = rq.get_instrument_industry(all_instruments['order_book_id'], source='citics_2019', level=1,
                                            date=date_end)
        citics_ii = rq.get_instrument_industry(all_instruments['order_book_id'], source='citics_2019', level=2,
                                               date=date_end)

        nonbank_id = citics[citics['first_industry_name'] == '非银行金融'].index
        citics.loc[nonbank_id, 'first_industry_name'] = citics_ii.loc[nonbank_id, 'second_industry_name'].values

        citics_trans_first = citics['first_industry_name'].to_dict()
        # %%
        # 自由流通市值，rq无数据，使用wind数据
        temp = w.wss(','.join(wind_code), "mkt_freeshares", f"unit=1;tradeDate={date_last:%Y%m%d}")
        free_mv = pd.Series(temp.Data[0], index=temp.Codes)
        free_mv.index = free_mv.index.str.replace('SH', 'XSHG').str.replace('SZ', 'XSHE')
        free_mv = free_mv.to_frame('free_mv')
        free_mv['citics'] = free_mv.index.map(citics_trans_first)
        citics_free_mv = free_mv.groupby('citics').sum()
        free_mv['citics_free_mv'] = free_mv['citics'].map(citics_free_mv['free_mv'])
        free_mv['weight'] = (free_mv['free_mv'] / free_mv['citics_free_mv']).fillna(0)
        # %%
        example_all = example_all.reset_index()
        example_all['pct_change'] = example_all['close'] / example_all['open'] - 1
        example_all['label'] = example_all['pct_change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        example_all['pct_change_abs'] = example_all['pct_change'].abs()
        example_all['date'] = example_all['datetime'].dt.date
        # %%
        example_all_sum = example_all.groupby(['order_book_id', 'date', 'label']).sum()
        example_all_sum = example_all_sum.reset_index()
        example_all_sum['citics'] = example_all_sum['order_book_id'].map(citics_trans_first)
        example_all_sum_up = example_all_sum.query('label == 1').set_index(['order_book_id', 'date'])
        example_all_sum_down = example_all_sum.query('label == -1').set_index(['order_book_id', 'date'])

        # %%
        ret = pd.DataFrame()
        ret['vol_ratio'] = example_all_sum_up['volume'] / example_all_sum_down['volume'] - 1
        ret['liq_ratio'] = -example_all_sum_up['pct_change'] / example_all_sum_up['volume'] / (
                example_all_sum_down['pct_change'] / example_all_sum_down['volume']) - 1
        ret['citics'] = example_all_sum_up.loc[ret.index, 'citics']
        ret = ret.reset_index()
        ret['weight'] = free_mv.loc[ret['order_book_id'], 'weight'].values
        ret = ret.fillna(0)
        ret['w_vol_ratio'] = ret['vol_ratio'] * ret['weight']
        ret['w_liq_ratio'] = ret['liq_ratio'] * ret['weight']
        ret_sum = ret.groupby('citics').sum().drop(0)
        # %%
        # 画图
        sns.set(font_scale=1.5, font='SimHei')
        ax = ret_sum.plot.scatter(x='w_vol_ratio', y='w_liq_ratio', s=100, figsize=(30, 15))
        for s, row in ret_sum.iterrows():
            ax.text(row['w_vol_ratio'], row['w_liq_ratio'], s, size='large',fontsize=35)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        plt.hlines(0, x0, x1)
        plt.vlines(0, y0, y1)
        ax.text(x0, y1, '主动卖\n被动买', color='r', ha='center', va='center', fontsize=35)
        ax.text(x1, y1, '主动买\n被动买', color='r', ha='center', va='center', fontsize=35)
        ax.text(x0, y0, '主动卖\n被动卖', color='r', ha='center', va='center', fontsize=35)
        ax.text(x1, y0, '主动买\n被动卖', color='r', ha='center', va='center', fontsize=35)
        plt.xlabel('主动盘面',fontsize=35)
        plt.ylabel('被动盘面',fontsize=35)
        plt.title(f'中信一级行业主动vs被动盘面, {date_end:%Y-%m-%d}',fontsize=35)
        mkdir(os.path.join(based_file, date_end.strftime('%Y-%m-%d')))

        plt.savefig(os.path.join(based_file, date_end.strftime('%Y-%m-%d'),
                                 '中信一级行业主动vs被动盘面象限图_%s.png' % date_end.strftime('%Y-%m-%d')), bbox_inches='tight')

        citics_iii = rq.get_instrument_industry(all_instruments['order_book_id'], source='citics_2019', level=3,
                                                date=date_end)
        # 映射到OMAKA三级行业
        omaka_three_industry = pd.read_csv(omaka_file_path)[['third_industry_name', 'omaka_industry']]
        for industry in citics_iii.third_industry_name.unique():
            if industry in omaka_three_industry.third_industry_name.to_list():
                citics_iii.loc[citics_iii[citics_iii.third_industry_name == industry].index, 'third_industry_name'] = \
                    omaka_three_industry[omaka_three_industry.third_industry_name == industry].omaka_industry.values[0]
            else:
                if industry == '水泥':
                    industry = '水泥III'
                    citics_iii.loc[
                        citics_iii[citics_iii.third_industry_name == industry].index, 'third_industry_name'] = \
                        omaka_three_industry[
                            omaka_three_industry.third_industry_name == industry].omaka_industry.values[0]
                else:
                    print('%s不存在于OMAKA三级分类中!' % industry)
                continue

        # 提取omaka当日已经有的个股权重
        omaka_trans_third = citics_iii['third_industry_name'].to_dict()
        name_list = ['传媒_其他', '传媒_游戏', '电力及公用事业', '电气设备', '新能源', '电子_半导体', '电子_其他',
                     '电子_消费电子', '电子_元器件', '房地产', '纺织服装', '保险', '多元金融', '证券', '钢铁',
                     '国防军工', '机械', '基础化工', '计算机_其他', '计算机_云服务', '家电', '建材', '建筑',
                     '交通运输_其他', '交通运输_物流', '煤炭', '农林牧渔_畜牧业', '农林牧渔_其他', '汽车_其他',
                     '汽车_汽车零部件', '轻工制造', '商贸零售', '石油石化', '食品饮料_酒类', '食品饮料_其他', '通信',
                     '消费者服务', '医药_化学制药', '医药_其他医药医疗', '医药_生物医药Ⅱ', '医药_中药生产', '银行',
                     '有色金属', '综合']
        # 读取已经计算好的omaka个股权重
        omaka_weights_df = pd.DataFrame()
        for code in name_list:
            temp_df = pd.read_hdf(omaka_based_file, key=code)
            # temp_df.security_code = temp_df.security_code.str.replace('SH', 'XSHG').str.replace('SZ', 'XSHE')
            omaka_weights_df = pd.concat([omaka_weights_df, temp_df], axis=1)
        # 自由流通市值，rq无数据，使用wind数据
        # temp = w.wss(','.join(wind_code), "mkt_freeshares", f"unit=1;tradeDate={date_last:%Y%m%d}")
        # free_mv = pd.Series(temp.Data[0], index=temp.Codes)
        # free_mv.index = free_mv.index.str.replace('SH', 'XSHG').str.replace('SZ', 'XSHE')
        # free_mv = free_mv.to_frame('free_mv')
        # free_mv['omaka'] = free_mv.index.map(omaka_trans_third)
        # omaka_free_mv = free_mv.groupby('omaka').sum()
        # free_mv['omaka_free_mv'] = free_mv['omaka'].map(omaka_free_mv['free_mv'])
        # free_mv['weight'] = (free_mv['free_mv'] / free_mv['omaka_free_mv']).fillna(0)
        # %%
        example_all = example_all.reset_index()
        example_all['pct_change'] = example_all['close'] / example_all['open'] - 1
        example_all['label'] = example_all['pct_change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        example_all['pct_change_abs'] = example_all['pct_change'].abs()
        example_all['date'] = example_all['datetime'].dt.date
        # %%
        example_all_sum = example_all.groupby(['order_book_id', 'date', 'label']).sum()
        example_all_sum = example_all_sum.reset_index()
        example_all_sum['omaka'] = example_all_sum['order_book_id'].map(omaka_trans_third)
        example_all_sum_up = example_all_sum.query('label == 1').set_index(['order_book_id', 'date'])
        example_all_sum_down = example_all_sum.query('label == -1').set_index(['order_book_id', 'date'])

        # %%
        ret = pd.DataFrame()
        ret['vol_ratio'] = example_all_sum_up['volume'] / example_all_sum_down['volume'] - 1
        ret['liq_ratio'] = -example_all_sum_up['pct_change'] / example_all_sum_up['volume'] / (
                example_all_sum_down['pct_change'] / example_all_sum_down['volume']) - 1
        ret['omaka'] = example_all_sum_up.loc[ret.index, 'omaka']
        ret = ret.reset_index()
        ret['weight'] = omaka_weights_df.loc[:, ret['order_book_id']].values[0]
        ret = ret.fillna(0)
        ret['w_vol_ratio'] = ret['vol_ratio'] * ret['weight']
        ret['w_liq_ratio'] = ret['liq_ratio'] * ret['weight']
        ret_sum = ret.groupby('omaka').sum().drop(0)
        # %%
        # 画图
        sns.set(font_scale=1.5, font='SimHei')
        ax = ret_sum.plot.scatter(x='w_vol_ratio', y='w_liq_ratio', s=100, figsize=(30, 15))
        for s, row in ret_sum.iterrows():
            ax.text(row['w_vol_ratio'], row['w_liq_ratio'], s, size='large',fontsize=35)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        plt.hlines(0, x0, x1)
        plt.vlines(0, y0, y1)
        ax.text(x0, y1, '主动卖\n被动买', color='r', ha='center', va='center', fontsize=35)
        ax.text(x1, y1, '主动买\n被动买', color='r', ha='center', va='center', fontsize=35)
        ax.text(x0, y0, '主动卖\n被动卖', color='r', ha='center', va='center', fontsize=35)
        ax.text(x1, y0, '主动买\n被动卖', color='r', ha='center', va='center', fontsize=35)
        plt.xlabel('主动盘面',fontsize=35)
        plt.ylabel('被动盘面',fontsize=35)
        plt.title(f'OMAKA三级行业主动vs被动盘面, {date_end:%Y-%m-%d}',fontsize=35)
        mkdir(os.path.join(based_file, date_end.strftime('%Y-%m-%d')))

        plt.savefig(os.path.join(based_file, date_end.strftime('%Y-%m-%d'),
                                 'OMAKA三级行业主动vs被动盘面象限图_%s.png' % date_end.strftime('%Y-%m-%d')), bbox_inches='tight')


if __name__ == "__main__":
    model = DynamicAndPessive()
    manager = JobManager()
    model.register_event(event_bus=manager.event_bus, job_uuid=None, debug=True)
    manager.event_bus.event_queue_reload(EVENT.PM0430)
    manager.event_bus.sequential_publish()
