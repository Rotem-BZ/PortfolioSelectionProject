import pandas as pd
import yfinance as yf
from portfolio_new import Portfolio
from portfolio_tf import PortfolioTF

import numpy as np
import os
import pickle
import utils

# START_DATE = '2017-08-01'
# END_TRAIN_DATE = '2022-08-31'
# END_TEST_DATE = '2022-09-31'

START_DATE = utils.START_DATE
END_TRAIN_DATE = utils.END_TRAIN_DATE
END_TEST_DATE = utils.END_TEST_DATE


def _test_portfolio(strategy=None):
    if strategy is None:
        strategy = Portfolio().get_portfolio
    # full_train = get_data()
    full_train = utils.get_data_with_cache()
    returns = []
    # strategy = Portfolio()
    for test_date in pd.date_range(END_TRAIN_DATE, END_TEST_DATE):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        if np.isnan(test_data).any():
            print("THERE IS NAN!!!")
        cur_return = cur_portfolio @ test_data
        returns.append({'date': test_date, 'return': cur_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    # print(sharpe)
    return sharpe


def multiple_test_portfolio(*funcs):
    cls = Portfolio()
    func_dict = {'baseline': cls.get_portfolio_baseline,
                'baseline_no_nan': cls.get_portfolio_baseline_no_nans,
                'minvar': cls.get_portfolio_minvariance,
                'minvar_no_nan': cls.get_portfolio_minvariance_no_nans,
                'transformer': cls.get_portfolio_transformer,
                'average': cls.get_portfolio_average,
                'average_g': cls.get_portfolio_greedy_average,
                'average_g_n': cls.get_portfolio_greedy_negative_average,
                 'average_g_n_pct': cls.get_portfolio_greedy_negative_average_pct,
                 'ensemble': cls.get_portfolio_ensemble}
    for func_name in funcs:
        func = func_dict[func_name]
        sharpe_value = _test_portfolio(func)
        print(f"{func_name=}, {sharpe_value= }")


if __name__ == '__main__':
    multiple_test_portfolio('ensemble', 'baseline', 'average', 'average_g', 'average_g_n', 'average_g_n_pct', 'transformer')
    # multiple_test_portfolio('baseline', 'baseline_no_nan', 'minvar', 'minvar_no_nan', 'transformer')
    # _test_portfolio()
