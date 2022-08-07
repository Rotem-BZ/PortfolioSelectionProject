import numpy as np
import os
from copy import deepcopy

save_model_weights = 'tf_weights.pkl'
# setting the seed allows for reproducible results
np.random.seed(123)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd

import utils
import portfolio_new
import dataset as ds


class Model(nn.Module):
    def __init__(self, input_shape=None, outputs=None):
        super(Model, self).__init__()
        if input_shape is None or outputs is None:
            data = utils.get_data_with_cache()
            data = portfolio_new.Portfolio.process_traindata(data)
            data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)
            input_shape = data_w_ret.shape
            outputs = len(data.columns)

        self.model = nn.Sequential(
            nn.LSTM(64, input_size=input_shape, batch_first=True),
            nn.Flatten(),
            nn.Linear(64, outputs),
            nn.Softmax(dim=-1)
        )

    def sharpe_loss(self, y_pred):
        # make all time-series start at 1
        data = torch.div(self.data, self.data[0])

        # value of the portfolio after allocations applied
        portfolio_values = torch.sum(torch.mul(data, y_pred), dim=1)

        portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[
                                                                             :-1]  # % change formula

        sharpe = torch.mean(portfolio_returns) / torch.std(portfolio_returns)

        # since we want to maximize Sharpe, while gradient descent minimizes the loss,
        #   we can negate Sharpe (the min of a negated function is its max)
        return -sharpe

    def get_allocations(self, data: pd.DataFrame):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data

        input: data - DataFrame of historical closing prices of various assets

        return: the allocations ratios for each of the given assets
        '''

        # data with returns
        data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)

        data = data.iloc[1:]

        fit_predict_data = data_w_ret[np.newaxis, :]
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=20, shuffle=False)
        return self.model.predict(fit_predict_data)[0]


class PortfolioTF:

    def __init__(self):
        if os.path.isfile(save_model_weights):
            self.model_class = Model()
            self.model_class.load_state_dict(torch.load(save_model_weights))
        else:
            raise ValueError("no saved weights for PortfolioTF")


    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        # train_data = train_data['Adj Close']
        train_data = portfolio_new.Portfolio.process_traindata(train_data)
        portfolio = self.model_class.get_allocations(train_data)
        return portfolio_new.Portfolio.finalize_portfolio(portfolio)


if __name__ == '__main__':
    p = PortfolioTF()
    data = utils.get_data_with_cache()
    x = p.get_portfolio(data)
