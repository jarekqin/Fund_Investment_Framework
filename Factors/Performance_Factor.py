import pandas as pd
import numpy as np
from BasedClass import FactorBasedClass

class PerformanceFacotr(FactorBasedClass):
    @staticmethod
    def rolling_return_daily(return_data, rolling_window):
        annual_factor = 250 / rolling_window
        log_return = np.log(return_data + 1)
        total_log_return = log_return.rolling(rolling_window).sum()
        cum_return = np.exp(total_log_return)
        annual_return = cum_return ** annual_factor - 1
        return annual_return

    @staticmethod
    def rolling_volatility_daily(return_data, rolling_window):
        annual_factor = 250 / rolling_window
        volatility = return_data.rolling(rolling_window, 0).std(ddof=1) * (annual_factor ** 0.5)
        return volatility

    @staticmethod
    def rolling_sharpe_daily(return_data, rolling_window):
        annual_return = PerformanceFacotr.rolling_return_daily(return_data, rolling_window)
        annual_volatility = PerformanceFacotr.rolling_volatility_daily(return_data, rolling_window)
        sharpe = annual_return / annual_volatility
        return sharpe

    @staticmethod
    def rolling_downside_deviation_daily(return_data, rolling_window, mar=0):
        annual_factor = 250 / rolling_window
        downside = return_data.clip(np.NINF, mar)
        downside = downside ** 2
        downside_deviation = downside.rolling(rolling_window).mean() ** 0.5
        annual_downside_deviation = downside_deviation * (annual_factor ** 0.5)
        return annual_downside_deviation

    @staticmethod
    def rolling_sortino_daily(return_data, rolling_window):
        annual_factor = 250 / rolling_window
        annual_downside_deviation = PerformanceFacotr.rolling_downside_deviation_daily(return_data, rolling_window)
        annual_return = return_data.rolling(rolling_window).mean() ** annual_factor
        sortino = annual_return / annual_downside_deviation
        return sortino

    @staticmethod
    def calculate_maxDrawDown(return_data):
        cum_return = (return_data + 1).cumprod()
        max_return = np.fmax.accumulate(cum_return)
        DD = ((cum_return - max_return) / max_return)
        maxDrawDown = DD.min()
        return maxDrawDown

    @staticmethod
    def rolling_maxDrawDown_daily(return_data, rolling_window):
        maxDD = return_data.rolling(rolling_window).apply(PerformanceFacotr.calculate_maxDrawDown)
        return maxDD

    @staticmethod
    def rolling_calmar_daily(return_data, rolling_window):
        maxDD = PerformanceFacotr.rolling_maxDrawDown_daily(return_data, rolling_window)
        annual_return = PerformanceFacotr.rolling_return_daily(return_data, rolling_window)
        calmar = annual_return / maxDD
        return calmar

    @staticmethod
    def save_factor(saved_data):
        pass

