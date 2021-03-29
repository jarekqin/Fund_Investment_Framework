import pandas as pd
import numpy as np
from WindPy import w
from datetime import datetime, timedelta

from omk.toolkit.cache import CacheKit
from jarvis.jobs.free_shares import FreeSharesToSql

name_list = ['传媒_其他', '传媒_游戏', '电力及公用事业', '电气设备', '新能源', '电子_半导体', '电子_其他',
             '电子_消费电子', '电子_元器件', '房地产', '纺织服装', '保险', '多元金融', '证券', '钢铁',
             '国防军工', '机械', '基础化工', '计算机_其他', '计算机_云服务', '家电', '建材', '建筑',
             '交通运输_其他', '交通运输_物流', '煤炭', '农林牧渔_畜牧业', '农林牧渔_其他', '汽车_其他',
             '汽车_汽车零部件', '轻工制造', '商贸零售', '石油石化', '食品饮料_酒类', '食品饮料_其他', '通信',
             '消费者服务', '医药_化学制药', '医药_其他医药医疗', '医药_生物医药Ⅱ', '医药_中药生产', '银行',
             '有色金属', '综合']


class OMAKAIndustryMapping:
    _based_file = 'D:\\SynologyDrive\\SynologyDrive\\Jarvis_Temp\\omaka_industry_weights'

    @staticmethod
    # 读取omaka三级行业映射表
    def get_omaka_three_level(path_file='D:\\SynologyDrive\\SynologyDrive\\Jarvis_Temp\\omaka行业.xlsx'):
        data = pd.read_excel(path_file, encoding='gb18030')
        omaka_industry_dict = {x: y for x, y in zip(data.third_industry_name, data.omaka_industry)}
        return omaka_industry_dict

    @staticmethod
    # 将rq代码转换未wind
    def convert_id_to_wind(codes):
        if isinstance(codes, dict):
            return_codes = {}
            for ind in codes:
                temp_code = [code.replace('XSHG', 'SH') if 'XSHG' in code
                             else code.replace('XSHE', 'SZ') for code in codes[ind]]
                return_codes[ind] = temp_code
        elif isinstance(codes, list):
            return_codes = [code.replace('XSHG', 'SH') if 'XSHG' in code
                            else code.replace('XSHE', 'SZ') for code in codes]
        elif isinstance(codes, str):
            if 'XSHG' in codes:
                return_codes = codes.replace('XSHG', 'SH')
            elif 'XSHE' in codes:
                return_codes = codes.replace('XSHE', 'SZ')
            else:
                print('%s无法转换由于错误的代码!' % codes)
        else:
            raise NotImplementedError
        return return_codes

    @staticmethod
    def get_mkt_shares_from_wind(codes, start_date, end_date):
        saved_df = pd.DataFrame()
        if isinstance(codes, list):
            temp = w.wsd(codes, "mkt_freeshares",
                         "%s" % pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                         "%s" % pd.to_datetime(end_date).strftime('%Y-%m-%d'),
                         "unit=1")
            return pd.DataFrame(temp.Data, index=temp.Times, columns=temp.Codes)
        elif isinstance(codes, dict):
            new_codes_dict = {}
            for ind in codes:
                temp = w.wsd(codes[ind], "mkt_freeshares",
                             "%s" % pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                             "%s" % pd.to_datetime(end_date).strftime('%Y-%m-%d'),
                             "unit=1")
                try:
                    saved_df = pd.concat([saved_df, pd.DataFrame(temp.Data, index=temp.Times, columns=temp.Codes)],
                                         axis=1)
                    new_codes_dict[ind] = pd.DataFrame(temp.Data, index=temp.Times, columns=temp.Codes)
                except Exception as e:
                    saved_df = pd.concat(
                        [saved_df, pd.DataFrame(np.transpose(temp.Data), index=temp.Times, columns=temp.Codes)],
                        axis=1)
                    new_codes_dict[ind] = pd.DataFrame(np.transpose(temp.Data), index=temp.Times, columns=temp.Codes)
            # FreeSharesToSql.save_fixed_free_shares(data=saved_df)
            return new_codes_dict
        elif isinstance(codes, str):
            temp = w.wsd(codes, "mkt_freeshares",
                         "%s" % pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                         "%s" % pd.to_datetime(end_date).strftime('%Y-%m-%d'),
                         "unit=1")
            return pd.DataFrame(temp.Data, index=temp.Times, columns=temp.Codes)
        else:
            raise NotImplementedError

    @staticmethod
    def cal_per_industry_weight(codes, start_date, end_date, codes_source='rq', mapping_omaka_industry=True):
        if codes_source == 'rq':
            return_codes = OMAKAIndustryMapping.convert_id_to_wind(codes)
            ind_mkt_shares = OMAKAIndustryMapping.get_mkt_shares_from_wind(return_codes, start_date, end_date)
            omaka_industry = OMAKAIndustryMapping.get_omaka_three_level()
            # 先映射到omaka行业
            omaka_ind_mkt_shares = {}
            for ind in ind_mkt_shares:
                if ind not in omaka_industry:
                    print('%s不存在于omaka自定义行业!请注意!' % ind)
                    continue
                if omaka_industry[ind] not in omaka_ind_mkt_shares:
                    # ind_mkt_shares[ind].index=[pd.to_datetime(x.date()) for x in ind_mkt_shares[ind].index.to_list()]
                    omaka_ind_mkt_shares[omaka_industry[ind]] = ind_mkt_shares[ind]
                else:
                    # ind_mkt_shares[ind].index = [pd.to_datetime(x.date()) for x in ind_mkt_shares[ind].index.to_list()]
                    omaka_ind_mkt_shares[omaka_industry[ind]] = pd.concat([omaka_ind_mkt_shares[omaka_industry[ind]],
                                                                           ind_mkt_shares[ind]], axis=1)
            # 按照个股-》行业计算权重
            return_weights = pd.DataFrame()
            for ind in omaka_ind_mkt_shares:
                temp_data = omaka_ind_mkt_shares[ind]
                # print(temp_data.columns.str.contains('XXSHGE'))
                # print('XXSHGE' in temp_data.columns)
                # print('XXSHGE' in temp_data.columns)
                for date in temp_data.index.to_numpy():
                    # 计算当前行业权重
                    temp_weights = temp_data.loc[date] / temp_data.loc[date].sum()
                    temp_weights = pd.DataFrame(temp_weights)
                    temp_weights = temp_weights.reset_index().rename(columns={date: 'industry_weights',
                                                                              'index': 'security_code'})
                    temp_weights['industry'] = ind
                    temp_weights['date'] = date
                    return_weights = pd.concat([return_weights, temp_weights], axis=0)
                del temp_data
            return_weights = return_weights.set_index('date')
            return_weights.index = pd.to_datetime(return_weights.index)
            return_weights.security_code = [x.replace('.SZ', '.XSHE') if 'SZ' in x else x.replace('SH', 'XSHG') for x in
                                            return_weights.security_code]
            # df=pd.pivot_table(return_weights, values=['industry_weights'], columns=['security_code'],
            #                index=return_weights.index).fillna(0)
            return return_weights

        elif codes_source == 'wind':
            ind_mkt_shares = OMAKAIndustryMapping.get_mkt_shares_from_wind(codes)
            # FreeSharesToSql.save_fixed_free_shares(data=ind_mkt_shares)
            # 按照个股-》行业计算权重
            omaka_industry = OMAKAIndustryMapping.get_omaka_three_level()
            omaka_ind_mkt_shares = {}
            for ind in ind_mkt_shares:
                if omaka_industry[ind] not in omaka_ind_mkt_shares:
                    # ind_mkt_shares[ind].index=[pd.to_datetime(x.date()) for x in ind_mkt_shares[ind].index.to_list()]
                    omaka_ind_mkt_shares[omaka_industry[ind]] = ind_mkt_shares[ind]
                else:
                    # ind_mkt_shares[ind].index = [pd.to_datetime(x.date()) for x in ind_mkt_shares[ind].index.to_list()]
                    omaka_ind_mkt_shares[omaka_industry[ind]] = pd.concat([omaka_ind_mkt_shares[omaka_industry[ind]],
                                                                           ind_mkt_shares[ind]], axis=1)
            # 按照个股-》行业计算权重
            return_weights = pd.DataFrame()
            for ind in omaka_ind_mkt_shares:
                temp_data = omaka_ind_mkt_shares[ind]
                # print(temp_data.columns.str.contains('XXSHGE'))
                # print('XXSHGE' in temp_data.columns)
                # print('XXSHGE' in temp_data.columns)
                for date in temp_data.index.to_numpy():
                    # 计算当前行业权重
                    temp_weights = temp_data.loc[date] / temp_data.loc[date].sum()
                    temp_weights = pd.DataFrame(temp_weights)
                    temp_weights = temp_weights.reset_index().rename(columns={date: 'industry_weights',
                                                                              'index': 'security_code'})
                    temp_weights['industry'] = ind
                    temp_weights['date'] = date
                    return_weights = pd.concat([return_weights, temp_weights], axis=0)
                del temp_data
            return_weights = return_weights.set_index('date')
            return_weights.index = pd.to_datetime(return_weights.index)
            return_weights.security_code = [x.replace('.SZ', '.XSHE') if 'SZ' in x else x.replace('SH', 'XSHG') for x in
                                            return_weights.security_code]
            # df=pd.pivot_table(return_weights, values=['industry_weights'], columns=['security_code'],
            #                       index=return_weights.index).fillna(0)
            return return_weights
        else:
            raise ValueError('codes_source仅支持rq/wind!')

    @staticmethod
    def out_func(return_weights, start_date, end_date):
        name_list = ['传媒_其他', '传媒_游戏', '电力及公用事业', '电气设备', '新能源', '电子_半导体', '电子_其他',
                     '电子_消费电子', '电子_元器件', '房地产', '纺织服装', '保险', '多元金融', '证券', '钢铁',
                     '国防军工', '机械', '基础化工', '计算机_其他', '计算机_云服务', '家电', '建材', '建筑',
                     '交通运输_其他', '交通运输_物流', '煤炭', '农林牧渔_畜牧业', '农林牧渔_其他', '汽车_其他',
                     '汽车_汽车零部件', '轻工制造', '商贸零售', '石油石化', '食品饮料_酒类', '食品饮料_其他', '通信',
                     '消费者服务', '医药_化学制药', '医药_其他医药医疗', '医药_生物医药Ⅱ', '医药_中药生产', '银行',
                     '有色金属', '综合']

        @CacheKit.tseries_df_cache(
            '%s_omaka_industry_weights.h5' % datetime.today().strftime('%Y-%m-%d'), start_ref='start_date',
            end_ref='end_date', key_ref='name_list', renew=False,
            truncate=False, verbose=True, folder=OMAKAIndustryMapping._based_file
        )
        def divide_into_cache(return_weights, start_date, end_date, name_list):
            # 分别生成44个omaka三级行业dataframe
            return_weights = return_weights[
                return_weights['date'].apply(lambda dt: pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date))]
            return_weights_group = return_weights.groupby('industry')
            results = [
                return_weights_group.get_group(ind).pivot('date', 'security_code', 'industry_weights') for ind in
                name_list
            ]
            return tuple(results)

        return divide_into_cache(return_weights, start_date, end_date, name_list)


if __name__ == "__main__":
    w.start()
    # name_list = ['传媒_其他', '传媒_游戏', '电力及公用事业', '电气设备', '新能源', '电子_半导体', '电子_其他',
    #              '电子_消费电子', '电子_元器件', '房地产', '纺织服装', '保险', '多元金融', '证券', '钢铁',
    #              '国防军工', '机械', '基础化工', '计算机_其他', '计算机_云服务', '家电', '建材', '建筑',
    #              '交通运输_其他', '交通运输_物流', '煤炭', '农林牧渔_畜牧业', '农林牧渔_其他', '汽车_其他',
    #              '汽车_汽车零部件', '轻工制造', '商贸零售', '石油石化', '食品饮料_酒类', '食品饮料_其他', '通信',
    #              '消费者服务', '医药_化学制药', '医药_其他医药医疗', '医药_生物医药Ⅱ', '医药_中药生产', '银行',
    #              '有色金属', '综合']
    # name_mapping_code = list(range(1, len(name_list) + 1))
    # return_name_list = {y: x for x, y in zip(name_list, name_mapping_code)}
    test = OMAKAIndustryMapping.cal_per_industry_weight(['000001.SZ', '000002.SZ'], '2021-02-18', '2021-02-19')
    # # print(test)
    # test2 = pd.read_hdf(OMAKAIndustryMapping._based_file, key='ZX_third_industry_weights')

    # data = pd.read_csv('D:\\备份\\Temp\\test.csv', encoding='gb18030')
    # data = data.set_index('date')
    # data.index = pd.to_datetime(data.index)
    # _ = OMAKAIndustryMapping.save_file(data, '2021-01-15', '2021-01-22')
