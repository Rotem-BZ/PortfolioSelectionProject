import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
import dataset as ds
import transformer_timeseries as tst


path_to_weights = 'transformer_weights.pkl'
NAN_ROWS_RATIO_THRESHOLD = 0.1
NAN_COLUMNS = ['CARR', 'CDAY', 'CEG', 'CTVA', 'DOW', 'FOX', 'FOXA', 'MRNA', 'OGN', 'OTIS']
LAST_START_DAY = 107

# Read data
# data = utils.get_data_with_cache()['Adj Close'][LAST_START_DAY:]
data = utils.make_training_data(LAST_START_DAY, NAN_COLUMNS)

# Hyperparams
test_size = 0.1
# target_cols = [col for col in data.columns if col not in NAN_COLUMNS]
batch_size = 32

# Params
dim_val = 24
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92  # length of input given to decoder
enc_seq_len = 153  # length of input given to encoder
output_sequence_length = 1  # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length  # used to slice data into sub-sequences
step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len

# Define input variables
# input_variables = target_cols
target_idx = 0 # index position of target in batched trg_y

# input_size = len(input_variables)
input_size = data.shape[1]


def train_and_save_weights():
    training_data = data[:-(round(len(data) * test_size))]

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

    model = tst.TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size,
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length,
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads)

    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, enc_seq_len]
    src_mask = utils.generate_square_subsequent_mask(
        dim1=batch_size * n_heads,
        dim2=output_sequence_length,
        dim3=enc_seq_len
    )

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.generate_square_subsequent_mask(
        dim1=batch_size * n_heads,
        dim2=output_sequence_length,
        dim3=output_sequence_length
    )

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()
    print("length of training data:", len(training_data))
    for i, (src, trg, trg_y) in enumerate(training_data):
        output = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        trg_y = trg_y[:, target_idx, :]
        loss = loss_func(output, trg_y)
        loss.backward()
        print(f"loss {i} ={loss.item()}")
        optimizer.step()

    torch.save(model.state_dict(), path_to_weights)
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
            self.model = tst.TimeSeriesTransformer(
                dim_val=dim_val,
                input_size=input_size,
                dec_seq_len=dec_seq_len,
                max_seq_len=max_seq_len,
                out_seq_len=output_sequence_length,
                n_decoder_layers=n_decoder_layers,
                n_encoder_layers=n_encoder_layers,
                n_heads=n_heads)
            self.model.load_state_dict(torch.load(path_to_weights))
        else:
            self.model = train_and_save_weights()

        self.src_mask = utils.generate_square_subsequent_mask(
            dim1=batch_size * n_heads,
            dim2=output_sequence_length,
            dim3=enc_seq_len
        )

        # Make tgt mask for decoder with size:
        # [batch_size*n_heads, output_sequence_length, output_sequence_length]
        self.tgt_mask = utils.generate_square_subsequent_mask(
            dim1=batch_size * n_heads,
            dim2=output_sequence_length,
            dim3=output_sequence_length
        )

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
        print(self.nan_columns)
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

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        self.clear()
        train_data = self.process_traindata(train_data)
        output = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        return output.detach().numpy()


if __name__ == '__main__':
    train_and_save_weights()
