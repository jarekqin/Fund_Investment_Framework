import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import rqdatac as rq

from omk.utils.const import TIME
from omk.core.vendor.RQData import RQData

from Factors.PerformanceFactor.Performance_Factor import PerformanceFacotr
from WindPy import w


class SingalFactorBackTesting:
    def __init__(self, factor_name, rolling_window=20, init=True, fund_type='stock'):
        if init:
            w.start()
            RQData.init()
        self.__factor_name = factor_name
        self.__rolling_window = rolling_window
        self.__fund_type = fund_type

    def _get_data(self, data_source='wind', end_date=None, backward_days=30):
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=backward_days)).strftime('%Y-%m-%d')
        if data_source == 'wind':
            if self.__fund_type == 'stock':
                temp = w.wset("sectorconstituent", "date=%s;sectorid=1000002533000000" % end_date)
            elif self.__fund_type == 'mix':
                temp = w.wset("sectorconstituent", "date=%s;sectorid=1000002535000000" % end_date)
            elif self.__fund_type == 'bond':
                temp = w.wset("sectorconstituent", "date=%s;sectorid=1000002534000000" % end_date)
            else:
                raise TypeError('fund type wrongly input!')
            fund_name = pd.DataFrame(np.transpose(temp.Data), columns=temp.Fields).wind_code.to_list()
            # taking net value pct
            fund_name = ','.join([str(x) for x in fund_name])
            temp = w.wsd("%s" % fund_name, "nav", "%s" % start_date, "%s" % end_date, "")
            data = pd.DataFrame(np.transpose(temp.Data), index=temp.Times, columns=temp.Codes).dropna(axis=1)
            data.index = pd.to_datetime(data.index)
        elif data_source == 'rq':
            data = None
        else:
            raise ValueError('Wrong data_source type!')
        return data.pct_change()

    def _back_testing(self, original_data, taking_date=None, pre_order=0.1):
        if taking_date is None:
            # if taking_date is None, then we will take today as basic variable
            taking_date =datetime.today().strftime('%Y-%m-%d')
        else:
            taking_date = pd.to_datetime(taking_date).strftime('%Y-%m-%')
        fund_taken_data = original_data.loc[taking_date].sort_values(ascending=False)
        pre_fund_id=fund_taken_data.iloc[:int(fund_taken_data.shape[0]*pre_order)]

    def main(self, data_source, end_date=None, backward_days=30):
        original_data = self._get_data(data_source, end_date, backward_days)


if __name__ == '__main__':
    pass
