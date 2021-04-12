import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omk.utils.const import TIME
from omk.core.vendor.RQData import RQData

from Factors.PerformanceFactor.Performance_Factor import PerformanceFacotr
from WindPy import w


class SingalFactorBackTesting:
    def __init__(self, factor_name,rolling_window=20, init=True):
        if init:
            w.start()
            RQData.init()
        self.__factor_name = factor_name
        self.__rolling_window=rolling_window

    def _get_data(self, data_source='wind'):
        if data_source == 'wind':
            data = None
        elif data_source == 'rq':
            data = None
            raise ValueError('Wrong data_source type!')
        return data


    def _back_testing(self):
        pass

    def main(self, data_source):
        original_data = self._get_data(data_source)


if __name__ == '__main__':
    pass
