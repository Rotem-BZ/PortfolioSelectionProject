import pandas as pd
import yfinance as yf
from portfolio import Portfolio
import numpy as np
import os
import pickle


# START_DATE = '2017-08-01'
# END_TRAIN_DATE = '2022-08-31'
# END_TEST_DATE = '2022-09-31'

START_DATE = '2017-08-01'
END_TRAIN_DATE = '2022-03-31'
END_TEST_DATE = '2022-06-30'


def get_wiki_table():
    filename = 'wiki_table_save.pkl'
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            wiki_table = pickle.load(file)
    else:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        with open(filename, 'wb') as file:
            pickle.dump(wiki_table, file)
    return wiki_table


def get_data_with_cache():
    filename = 'data_save.pkl'
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    else:
        data = get_data()
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    return data


def get_data():
    # wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    wiki_table = get_wiki_table()
    sp_tickers = wiki_table[0]
    tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
    data = yf.download(tickers, START_DATE, END_TEST_DATE)
    return data


def test_portfolio(strategy=None):
    if strategy is None:
        strategy = Portfolio()
    # full_train = get_data()
    full_train = get_data_with_cache()
    returns = []
    # strategy = Portfolio()
    for test_date in pd.date_range(END_TRAIN_DATE, END_TEST_DATE):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        cur_return = cur_portfolio @ test_data
        returns.append({'date': test_date, 'return': cur_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    # print(sharpe)
    return sharpe


def multiple_test_portfolio(*funcs):
    for func_name in funcs:
        cls = Portfolio()
        # func = {'baseline': cls.get_portfolio_baseline,
        #         'baseline_no_nan': cls.get_portfolio_baseline_no_nans,
        #         'SAA': cls.get_portfolio_SAA}[func_name]
        func = {'baseline': cls.get_portfolio_baseline,
                'baseline_no_nan': cls.get_portfolio_baseline_no_nans}[func_name]
        cls.get_portfolio = func
        sharpe_value = test_portfolio(cls)
        print(f"{func_name=}, {sharpe_value=}")


if __name__ == '__main__':
    # test_portfolio()
    # multiple_test_portfolio('baseline', 'baseline_no_nan', 'SAA')
    multiple_test_portfolio('baseline', 'baseline_no_nan')
