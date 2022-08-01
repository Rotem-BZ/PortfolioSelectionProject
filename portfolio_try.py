import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

import utils
import dataset as ds


path_to_weights = 'transformer_weights_try2.pkl'
NAN_COLUMNS = ['CARR', 'CDAY', 'CEG', 'CTVA', 'DOW', 'FOX', 'FOXA', 'MRNA', 'OGN', 'OTIS']
LAST_START_DAY = 107

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Read data
# data = utils.make_training_data(LAST_START_DAY, NAN_COLUMNS)
data = utils.get_data_with_cache()

# Hyperparams
test_size = 0.1
batch_size = 16
# EPOCHS = 40
EPOCHS = 20

# Params
# dim_val = 24
n_heads = 5
total_input_size = 510
n_decoder_layers = 3
n_encoder_layers = 3
dec_seq_len = 92  # length of input given to decoder. UNUSED!
enc_seq_len = 70  # length of input given to encoder
output_sequence_length = 1  # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length  # used to slice data into sub-sequences
step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step
dim_feedforward = 1024
dropout = 0.1
# in_features_encoder_linear_layer = 1024
# in_features_decoder_linear_layer = 1024
max_seq_len = enc_seq_len

# Define input variables
# input_variables = target_cols
target_idx = 0 # index position of target in batched trg_y

# input_size = len(input_variables)
input_size = data['Adj Close'].shape[1] - len(NAN_COLUMNS)


def train_and_save_weights():
    model = nn.Transformer(d_model=total_input_size, nhead=n_heads,
                           num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout, device=device, batch_first=True)
    # training_data = Portfolio.process_traindata_2(data)

    train_data = data['Adj Close'].dropna(axis=0, how='all')
    # remove some rows at the beginning with nans
    train_data = train_data.iloc[LAST_START_DAY:]
    columns_with_nan = [x for x, v in train_data.isna().any().items() if v]
    diff = set(columns_with_nan) - set(NAN_COLUMNS)
    if len(diff) != 0:
        print(f"true NAN COLUMNS: \n {columns_with_nan}")
        raise(ValueError(f"unknown columns with nan: \n {', '.join(list(diff))} \n"))

    # remove nan columns from train set. will be given 0 in the final portfolio
    train_data = train_data[[col for col in train_data.columns if col not in NAN_COLUMNS]]

    train_data = train_data.pct_change(1)[1:]  # relative change in return
    train_data = np.pad(train_data.to_numpy(), [(0, 0), (0, total_input_size - input_size)])

    training_data = train_data[:-(round(len(data) * test_size))]
    test_data = train_data[-(round(len(data) * test_size)):]

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
    # Should be training data indices only
    training_indices = utils.get_indices_entire_sequence(
        data=training_data,
        window_size=window_size,
        step_size=step_size)

    # Making instance of custom dataset class
    training_data = ds.TransformerDataset(
        data=torch.tensor(training_data).float(),
        indices=training_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=output_sequence_length
    )

    training_data = DataLoader(training_data, batch_size)

    test_indices = utils.get_indices_entire_sequence(
        data=test_data,
        window_size=window_size,
        step_size=step_size)

    test_data = ds.TransformerDataset(
        data=torch.tensor(test_data).float(),
        indices=test_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=output_sequence_length
    )

    test_data = DataLoader(test_data, len(test_data))

    src_mask = model.generate_square_subsequent_mask(enc_seq_len).to(device)
    tgt_mask = model.generate_square_subsequent_mask(output_sequence_length).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()
    model.train()
    print("length of training data:", len(training_data))
    train_losses = []
    test_losses = []
    for epoch in range(EPOCHS):
        print(f"epoch {epoch + 1} / {EPOCHS}")
        for i, (src, trg, trg_y) in enumerate(training_data):
            src = src.to(device)
            trg = trg.to(device)
            trg_y = trg_y.to(device)
            output = model(
                src=src,
                tgt=trg,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            trg_y = trg_y[:, target_idx, :]
            loss = loss_func(output.squeeze(1), trg_y)
            loss.backward()
            # print(f"loss {i} ={loss.item()}")
            optimizer.step()
            train_losses.append(loss.item())
        with torch.no_grad():
            src, trg, trg_y = next(iter(test_data))
            src = src.to(device)
            trg = trg.to(device)
            trg_y = trg_y.to(device)
            output = model(
                src=src,
                tgt=trg,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            trg_y = trg_y[:, target_idx, :]
            loss = loss_func(output.squeeze(1), trg_y)
            test_losses.append(loss.item())

    torch.save(model.state_dict(), path_to_weights)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(train_losses)
    ax1.set_title("train losses")
    ax2.plot(test_losses)
    ax2.set_title("test losses")
    plt.show()
    return model


class Portfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.nan_columns = ['CARR', 'CDAY', 'CEG', 'CTVA', 'DOW', 'FOX', 'FOXA', 'MRNA', 'OGN', 'OTIS']
        self.nan_column_indices = []

        if os.path.isfile(path_to_weights):
            self.model = nn.Transformer(d_model=total_input_size, nhead=n_heads,
                           num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout, device=device, batch_first=True)
            self.model.load_state_dict(torch.load(path_to_weights))
        else:
            self.model = train_and_save_weights()

        self.src_mask = self.model.generate_square_subsequent_mask(enc_seq_len)
        self.tgt_mask = self.model.generate_square_subsequent_mask(output_sequence_length)

    def clear(self):
        self.nan_columns = NAN_COLUMNS
        self.nan_column_indices = []

    # def process_traindata(self, train_data: pd.DataFrame):
    #     train_data = train_data['Adj Close'].dropna(axis=0, how='all')
    #     N_rows = train_data.shape[0]
    #     columns_with_nan = [x for x,v in train_data.isna().any().items() if v]
    #     first_date_indexes = []
    #     for col in columns_with_nan:
    #         first_date_index = train_data[col].isna().argmin()
    #         if first_date_index > N_rows * NAN_ROWS_RATIO_THRESHOLD:
    #             # columns with ratio of nans higher than NAN_ROWS_RATIO_THRESHOLD are deleted
    #             self.nan_columns.append(col)
    #             self.nan_column_indices.append(train_data.columns.tolist().index(col))
    #         else:
    #             first_date_indexes.append(first_date_index)
    #     last_start_date = max(first_date_indexes)
    #     print(self.nan_columns)
    #     train_data = train_data.iloc[last_start_date:]
    #
    #     # remove nan columns from train set. will be given 0 in the final portfolio
    #     train_data = train_data[[col for col in train_data.columns if col not in self.nan_columns]]
    #     return train_data


    def process_traindata(self, train_data: pd.DataFrame):
        train_data = train_data['Adj Close'].dropna(axis=0, how='all')
        # remove some rows at the beginning with nans
        train_data = train_data.iloc[LAST_START_DAY:]
        columns_with_nan = [x for x, v in train_data.isna().any().items() if v]
        diff = set(columns_with_nan) - set(self.nan_columns)
        assert len(diff) == 0, f"unknown columns with nan: \n {', '.join(list(diff))} \n"
        N_rows = train_data.shape[0]
        columns_list = train_data.columns.tolist()
        for col in self.nan_columns:
            self.nan_column_indices.append(columns_list.index(col))

        # remove nan columns from train set. will be given 0 in the final portfolio
        train_data = train_data[[col for col in train_data.columns if col not in self.nan_columns]]
        return train_data

    def process_traindata_2(self, train_data: pd.DataFrame):
        # train_data = train_data['Adj Close'].dropna(axis=0, how='all')
        # # remove some rows at the beginning with nans
        # train_data = train_data.iloc[LAST_START_DAY:]
        # columns_with_nan = [x for x, v in train_data.isna().any().items() if v]
        # diff = set(columns_with_nan) - set(self.nan_columns)
        # assert len(diff) == 0, f"unknown columns with nan: \n {', '.join(list(diff))} \n"
        # columns_list = train_data.columns.tolist()
        # for col in self.nan_columns:
        #     self.nan_column_indices.append(columns_list.index(col))
        #
        # # remove nan columns from train set. will be given 0 in the final portfolio
        # train_data = train_data[[col for col in train_data.columns if col not in self.nan_columns]]
        # # all_weekdays = pd.date_range(start=train_data.index[0], end=train_data.index[-1], freq='B')
        # # close = close.reindex(all_weekdays)
        # # close = close.fillna(method='ffill')
        train_data = self.process_traindata(train_data)
        returns = train_data.pct_change(1)  # relative change in return
        returns = np.pad(returns.to_numpy(), [(0,0), (0, total_input_size - input_size)])
        return returns


    def finalize_portfolio(self, portfolio: np.ndarray):
        final_portfolio = np.zeros(len(portfolio) + len(self.nan_columns))
        final_portfolio[[i for i in range(len(final_portfolio)) if i not in self.nan_column_indices]] = portfolio
        return final_portfolio / final_portfolio.sum()


    def finalize_portfolio_2(self, portfolio: np.ndarray):
        final_portfolio = portfolio[:input_size]
        final_portfolio = self.finalize_portfolio(final_portfolio)
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

    # def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
    #     """
    #     The function used to get the model's portfolio for the next day
    #     :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
    #     with all the training data. The following day (the first that does not appear in the index) is the test day
    #     :return: a numpy array of shape num_stocks with the portfolio for the test day
    #     """
    #     self.clear()
    #     train_data = self.process_traindata(train_data)
    #     train_data = train_data.to_numpy()
    #     train_data = train_data[1:] - train_data[:-1] - 1
    #     src = torch.from_numpy(train_data[-enc_seq_len:]).unsqueeze(0).float()
    #     trg = src[:, -1:, :]
    #     self.model.eval()
    #     with torch.inference_mode():
    #         output = self.model(
    #             src=src.to(device),
    #             tgt=trg.to(device),
    #             src_mask=self.src_mask.to(device),
    #             tgt_mask=self.tgt_mask.to(device)
    #         )
    #     prediction = output.squeeze(0).squeeze(0).to('cpu').numpy()
    #     return self.finalize_portfolio(prediction)

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = self.process_traindata_2(train_data)[1:]
        src = torch.from_numpy(train_data[-enc_seq_len:]).unsqueeze(0).float()
        trg = src[:, -1:, :]
        self.model.eval()
        with torch.inference_mode():
            output = self.model(
                src=src.to(device),
                tgt=trg.to(device),
                src_mask=self.src_mask.to(device),
                tgt_mask=self.tgt_mask.to(device)
            )
        prediction = output.squeeze(0).squeeze(0).to('cpu').numpy()
        return self.finalize_portfolio_2(prediction)

    def get_portfolio_minvariance(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        close = train_data['Adj Close']
        all_weekdays = pd.date_range(start=train_data.index[0], end=train_data.index[-1], freq='B')
        close = close.reindex(all_weekdays)
        close = close.fillna(method='ffill')

        returns = close.pct_change(1)  # relative change in return
        # returns = returns.reindex(sorted(returns.columns), axis=1)
        cov_mat = returns.cov()
        cov_mat = cov_mat.to_numpy()

        inverse_cov_mat = np.linalg.inv(cov_mat)
        row_sum = np.sum(inverse_cov_mat, axis=1)
        min_var_portfolio = row_sum / np.sum(row_sum)
        return min_var_portfolio

    def get_portfolio_minvariance_no_nans(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        self.clear()
        close = self.process_traindata(train_data)
        all_weekdays = pd.date_range(start=train_data.index[0], end=train_data.index[-1], freq='B')
        close = close.reindex(all_weekdays)
        close = close.fillna(method='ffill')

        returns = close.pct_change(1)  # relative change in return
        # returns = returns.reindex(sorted(returns.columns), axis=1)
        cov_mat = returns.cov()
        cov_mat = cov_mat.to_numpy()

        inverse_cov_mat = np.linalg.inv(cov_mat)
        row_sum = np.sum(inverse_cov_mat, axis=1)
        min_var_portfolio = row_sum / np.sum(row_sum)
        return self.finalize_portfolio(min_var_portfolio)


if __name__ == '__main__':
    train_and_save_weights()
