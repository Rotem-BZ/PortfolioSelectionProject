import numpy as np
import pandas as pd
# import cvxpy as cp


NAN_ROWS_RATIO_THRESHOLD = 0.1


class Portfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.nan_columns = []
        self.nan_column_indices = []

    def clear(self):
        self.nan_columns = []
        self.nan_column_indices = []

    def process_traindata(self, train_data: pd.DataFrame):
        train_data = train_data['Adj Close'].dropna(axis=0, how='all')
        N_rows = train_data.shape[0]
        columns_with_nan = [x for x,v in train_data.isna().any().items() if v]
        first_date_indexes = []
        for col in columns_with_nan:
            first_date_index = train_data[col].isna().argmin()
            if first_date_index > N_rows * NAN_ROWS_RATIO_THRESHOLD:
                # columns with ratio of nans higher than NAN_ROWS_RATIO_THRESHOLD are deleted
                self.nan_columns.append(col)
                self.nan_column_indices.append(train_data.columns.tolist().index(col))
            else:
                first_date_indexes.append(first_date_index)
        last_start_date = max(first_date_indexes)
        train_data = train_data.iloc[last_start_date:]

        # remove nan columns from train set. will be given 0 in the final portfolio
        train_data = train_data[[col for col in train_data.columns if col not in self.nan_columns]]
        return train_data

    def finalize_portfolio(self, portfolio: np.ndarray):
        final_portfolio = np.zeros(len(portfolio) + len(self.nan_columns))
        final_portfolio[[i for i in range(len(final_portfolio)) if i not in self.nan_column_indices]] = portfolio
        return final_portfolio

    def get_portfolio_baseline(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        self.clear()
        train_data = train_data['Adj Close']
        portfolio = np.ones(len(train_data.columns)) / len(train_data.columns)
        return portfolio

    def get_portfolio_baseline_no_nans(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        self.clear()
        train_data = self.process_traindata(train_data)
        portfolio = np.ones(len(train_data.columns)) / len(train_data.columns)
        return self.finalize_portfolio(portfolio)

    # def get_portfolio_SAA(self, train_data: pd.DataFrame) -> np.ndarray:
    #     """
    #     The function used to get the model's portfolio for the next day
    #     :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
    #     with all the training data. The following day (the first that does not appear in the index) is the test day
    #     :return: a numpy array of shape num_stocks with the portfolio for the test day
    #     """
    #     self.clear()
    #     train_data = self.process_traindata(train_data)
    #     portfolio = self.offline_SAA_solver_func(train_data.to_numpy(), rho=0.1, alpha=0.5)
    #     return self.finalize_portfolio(np.array(portfolio))
    #
    # @staticmethod
    # def offline_SAA_solver_func(xi_mat: np.ndarray, rho: float, alpha: float):
    #     portfolio = cp.Variable(xi_mat.shape[1])
    #     tau = cp.Variable()
    #     inner_products = xi_mat @ portfolio  # shape (n)
    #     vec1 = -inner_products + rho * tau
    #     vec2 = inner_products * (-1 - rho / alpha) + rho * tau * (1 - 1 / alpha)
    #     losses = cp.maximum(vec1, vec2)
    #     objective = cp.Minimize(cp.sum(losses) / xi_mat.shape[0])
    #     constraints = [portfolio >= 0, cp.sum(portfolio) == 1, tau >= 0]
    #     prob = cp.Problem(objective, constraints)
    #     prob.solve()
    #     return portfolio.value
