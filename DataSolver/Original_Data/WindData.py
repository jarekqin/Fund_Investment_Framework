import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from WindPy import w
from BasedClass.FactorBasedClass import FactorBase


class WindData(FactorBase):
    def __init__(self, start_date=None, end_date=None, time_step=10, init_engine=True):
        if init_engine:
            w.start()
        else:
            print('默认不初始化WIND引擎')
        if start_date is None:
            self.__start_date = pd.to_datetime(datetime.today() - timedelta(time_step))
        else:
            self.__start_date = pd.to_datetime(start_date)
        if end_date is None:
            self.__start_date = pd.to_datetime(datetime.today())
        else:
            self.__start_date = pd.to_datetime(end_date)
        self.__data=None

    def __repr__(self):
        return 'getting %s length data from wind' % self.__data.shape[0]

    def get_data(self,file_type='fund_basic_info'):
        pass

    def save_factor(self,save_file_path,encoding='utf-8'):
        self.__data.to_csv(save_file_path,encoding=encoding)


if __name__=='__main__':
    test=WindData('2020-01-01','2021-01-01')