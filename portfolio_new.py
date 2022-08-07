import datetime
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

# torch.manual_seed(42)
# import wandb
# wandb.init(project='mlps_project', entity='idozuck', config={'batch_size': 24, 'n_decoder_layers': 3,
#                                                              'n_encoder_layers': 3, 'enc_seq_len': 50,
#                                                              'dim_feedforward': 1024, 'dropout': 0.1,
#                                                              'num_heads': 5, 'CONCAT_STOCKS': True})
# config = wandb.config
# batch_size = config['batch_size']
# n_decoder_layers = config['n_decoder_layers']
# n_encoder_layers = config['n_encoder_layers']
# enc_seq_len = config['enc_seq_len']
# dim_feedforward = config['dim_feedforward']
# dropout = config['dropout']
# num_heads = config['num_heads']
# CONCAT_STOCKS = True

batch_size = 10
n_decoder_layers = 1
n_encoder_layers = 1
enc_seq_len =10
dim_feedforward = 156
dropout = 0.05574
num_heads = 1
CONCAT_STOCKS = False
FEATURES = ['diffs', 'diffs+stocks', 'diffs+volume'][0]

import utils
import dataset as ds


path_to_weights = 'transformer_weights_try2.pkl'
path_to_ensamble_weights = 'ensable_weights.pkl'
# NAN_COLUMNS = ['CARR', 'CDAY', 'CEG', 'CTVA', 'DOW', 'FOX', 'FOXA', 'MRNA', 'OGN', 'OTIS']
NAN_COLUMNS = ['CARR', 'CDAY', 'CEG', 'CTVA', 'DOW', 'FOX', 'FOXA', 'MRNA', 'OGN', 'OTIS', 'VICI']
# NAN_COLUMNS = ['ABBV', 'ALLE', 'AMCR', 'ANET', 'APTV', 'AVGO', 'AWK', 'CARR', 'CBOE',
#                'CDAY', 'CDW', 'CEG', 'CFG', 'CHTR', 'CTLT', 'CTVA', 'CZR', 'DG', 'DOW',
#                'ENPH', 'EPAM', 'ETSY', 'FANG', 'FBHS', 'FLT', 'FOX', 'FOXA', 'FRC', 'FTNT',
#                'FTV', 'GM', 'GNRC', 'HCA', 'HII', 'HLT', 'HPE', 'HWM', 'IQV', 'IR', 'KDP',
#                'KEYS', 'KHC', 'KMI', 'LW', 'LYB', 'META', 'MPC', 'MRNA', 'NCLH', 'NLSN',
#                'NOW', 'NWS', 'NWSA', 'NXPI', 'OGN', 'OTIS', 'PAYC', 'PM', 'PSX', 'PYPL',
#                'QRVO', 'SEDG', 'SYF', 'TSLA', 'TWTR', 'V', 'VICI', 'VRSK', 'WRK', 'XYL', 'ZTS']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

data = utils.get_data_with_cache()
NAN_COLUMN_INDICES = [data['Adj Close'].columns.to_list().index(col) for col in NAN_COLUMNS]

# Hyperparams
test_size = 0.1
# batch_size = 16
EPOCHS = 80

# Params
# dim_val = 24
# n_heads = 5
total_input_size = {'diffs': 510}.get(FEATURES, 1000)
# n_decoder_layers = 3
# n_encoder_layers = 3
dec_seq_len = 15  # length of input given to decoder. UNUSED!
# enc_seq_len = 50  # length of input given to encoder
output_sequence_length = 1  # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length  # used to slice data into sub-sequences
step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step
# dim_feedforward = 1024
# dropout = 0.1
# in_features_encoder_linear_layer = 1024
# in_features_decoder_linear_layer = 1024
max_seq_len = enc_seq_len

# Define input variables
# input_variables = target_cols
target_idx = 0 # index position of target in batched trg_y

# input_size = len(input_variables)
raw_input_size = data['Adj Close'].shape[1] # 503
logistic_regression_seq_len = 24
input_size = raw_input_size - len(NAN_COLUMNS)


def train_and_save_weights():
    model = nn.Transformer(d_model=total_input_size, nhead=num_heads,
                           num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout, device=device, batch_first=True)

    # train_data = Portfolio.process_traindata_3(data) if CONCAT_STOCKS else Portfolio.process_traindata_2(data)
    train_data = {'diffs': Portfolio.process_traindata_2, 'diffs+stocks': Portfolio.process_traindata_3,
                             'diffs+volume': Portfolio.process_traindata_4}[FEATURES](data)

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
    model.train()
    print("length of training data:", len(training_data))
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model_weights = deepcopy(model.state_dict())
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
            # if epoch == 15:
            #     raise ValueError(f"{output.max()=}, {output.min()=}")
            # loss = loss_func(output.squeeze(1), trg_y)
            # loss = -(output.squeeze(1) @ trg_y.T).diag().mean()
            inner_products = (output.squeeze(1) @ trg_y.T).diag()
            loss = -inner_products.mean() / inner_products.std()
            # loss -= 1 * output.std()
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
            # loss = loss_func(output.squeeze(1), trg_y).item()
            # loss = -(output.squeeze(1) @ trg_y.T).diag().mean().cpu().item()
            inner_products = (output.squeeze(1) @ trg_y.T).diag()
            loss = inner_products.mean() / inner_products.std()
            loss = loss.cpu().item()
            # wandb.log({'validation_loss': loss})
            test_losses.append(loss)
            if loss < best_test_loss and epoch > 4:
                best_model_weights = deepcopy(model.state_dict())

    # torch.save(model.state_dict(), path_to_weights)
    torch.save(best_model_weights, path_to_weights)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    test_optimal_loss, best_epoch = min([(loss, epoch) for epoch, loss in enumerate(test_losses)])
    print("best epoch:", best_epoch + 1)
    ax1.plot(train_losses)
    ax1.set_title("train losses")
    ax2.plot(test_losses)
    ax2.scatter(best_epoch, test_optimal_loss, label="saved checkpoint")
    ax2.set_title("test losses")
    ax2.legend()
    plt.show()
    return model


class LogisticRegressionPerStock(nn.Module):
    def __init__(self):
        super(LogisticRegressionPerStock, self).__init__()
        self.linear_layers = [nn.Linear(logistic_regression_seq_len, 1) for _ in range(raw_input_size)]
        self.final_transform = nn.Linear(raw_input_size, raw_input_size)

    def forward(self, x: torch.Tensor):
        # x shape (Batch, seq, input_dim)
        assert x.shape[1] == logistic_regression_seq_len and x.shape[2] == raw_input_size
        linear_transformation = [torch.sigmoid(self.linear_layers[i](x[:, :, i])) for i in range(raw_input_size)]
        concat = torch.cat(linear_transformation)
        return self.final_transform(concat)

    def train_logistic_regressions(self, data: np.ndarray):
        pass

    def train_whole_network(self, data: np.ndarray):
        pass


class Portfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        if os.path.isfile(path_to_weights):
            self.model = nn.Transformer(d_model=total_input_size, nhead=num_heads,
                           num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout, device=device, batch_first=True)
            self.model.load_state_dict(torch.load(path_to_weights))
        # else:
        #     self.model = train_and_save_weights()

        self.src_mask = self.model.generate_square_subsequent_mask(enc_seq_len)
        self.tgt_mask = self.model.generate_square_subsequent_mask(output_sequence_length)

        self.process_func = {'diffs': self.process_traindata_2, 'diffs+stocks': self.process_traindata_3,
                             'diffs+volume': self.process_traindata_4}[FEATURES]

    @staticmethod
    def process_traindata(train_data: pd.DataFrame, col_name: str = 'Adj Close'):
        """
        Remove pre-decided columns (which don't exist for the entire timeline). After finding a portfolio, call
        `finalize_portfolio` to return those columns with the value 0.
        :param train_data: data as dataframe
        :param col_name: column of data to take
        :return:
        """
        train_data = train_data[col_name].dropna(axis=0, how='all')
        # remove some rows at the beginning with nans
        # train_data = train_data.iloc[LAST_START_DAY:]
        columns_with_nan = [x for x, v in train_data.isna().any().items() if v]
        diff1 = set(columns_with_nan) - set(NAN_COLUMNS)
        diff2 = set(NAN_COLUMNS) - set(columns_with_nan)
        if diff1 or diff2:
            print(f"true NAN COLUMNS: \n {columns_with_nan}")
        if len(diff1) != 0:
            raise (ValueError(f"unknown columns with nan: \n {', '.join(list(diff1))} \n "))
        if len(diff2) != 0:
            print(f"warning: there are nanned columns without nans:\n {', '.join(list(diff2))}")

        # remove nan columns from train set. will be given 0 in the final portfolio
        train_data = train_data[[col for col in train_data.columns if col not in NAN_COLUMNS]]
        return train_data

    @staticmethod
    def process_traindata_2(train_data: pd.DataFrame):
        """
        Remove columns as in `process_traindata`, but also call pct_change do take data diffs instead of original stock
        values, and pads the data to get dimension "total_input_size". use `finalize_portfolio_2` at the end.
        :param train_data:
        :return:
        """
        train_data = Portfolio.process_traindata(train_data)
        returns = train_data.pct_change(1)[1:]  # relative change in return
        returns = np.pad(returns.to_numpy(), [(0,0), (0, total_input_size - input_size)])
        return returns

    @staticmethod
    def process_traindata_3(train_data: pd.DataFrame):
        """
        Same as `process_traindata_2`, but concatenates the data with the original stock prices (bigger data dimension)
        :param train_data:
        :return:
        """
        train_data = Portfolio.process_traindata(train_data)
        returns = train_data.pct_change(1)[1:]  # relative change in return
        returns = np.hstack([returns.to_numpy(), train_data[1:].values])
        returns = np.pad(returns, [(0, 0), (0, total_input_size - 2*input_size)])
        return returns

    @staticmethod
    def process_traindata_4(train_data: pd.DataFrame):
        """
        Same as `process_traindata_2`, but concatenates the data with the Volume values (bigger data dimension)
        :param train_data:
        :return:
        """
        # vol_data = train_data['Volume']
        vol_data = Portfolio.process_traindata(train_data, 'Volume')
        train_data = Portfolio.process_traindata(train_data)
        returns = train_data.pct_change(1)[1:]  # relative change in return
        returns = np.hstack([returns.to_numpy(), vol_data[1:].values])
        returns = np.pad(returns, [(0, 0), (0, total_input_size - 2 * input_size)])
        return returns


    @staticmethod
    def finalize_portfolio(portfolio: np.ndarray):
        final_portfolio = np.zeros(len(portfolio) + len(NAN_COLUMNS))
        final_portfolio[[i for i in range(len(final_portfolio)) if i not in NAN_COLUMN_INDICES]] = portfolio
        return final_portfolio / final_portfolio.sum()


    @staticmethod
    def finalize_portfolio_2(portfolio: np.ndarray):
        final_portfolio = portfolio[:input_size]
        final_portfolio = Portfolio.finalize_portfolio(final_portfolio)
        return final_portfolio

    def get_portfolio_baseline(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
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
        train_data = self.process_traindata(train_data)
        portfolio = np.ones(len(train_data.columns)) / len(train_data.columns)
        return self.finalize_portfolio(portfolio)

    def get_portfolio_transformer(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        # train_data = self.process_traindata_3(train_data) if CONCAT_STOCKS else self.process_traindata_2(train_data)
        train_data = self.process_func(train_data)
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
        print(1)
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

    def get_portfolio_average(self, train_data: pd.DataFrame) -> np.ndarray:
        # train_data = self.process_traindata_2(train_data)
        train_data = train_data['Adj Close'].dropna(axis=0, how='all')
        k = 100  # hyper parameter
        train_data = train_data[-k:]
        portfolio = train_data.mean(axis=0)
        return self.finalize_portfolio_2(portfolio)

    def get_portfolio_greedy_average(self, train_data: pd.DataFrame) -> np.ndarray:
        portfolio = self.get_portfolio_average(train_data)
        result = np.zeros(len(portfolio))
        result[np.argmax(portfolio)] = 1
        return result

    def get_portfolio_greedy_negative_average(self, train_data: pd.DataFrame) -> np.ndarray:
        portfolio = self.get_portfolio_average(train_data)
        result = np.zeros(len(portfolio))
        result[np.argmin(portfolio)] = -10_000
        result[np.argmax(portfolio)] = 10_001
        return self.finalize_portfolio_2(result)

    def get_portfolio_greedy_negative_average_pct(self, train_data: pd.DataFrame) -> np.ndarray:
        # portfolio = self.get_portfolio_average(train_data)
        data = self.process_traindata_2(train_data)
        k = 3  # hyper parameter
        data = data[-k:]
        portfolio = data.mean(axis=0)
        result = np.zeros(len(portfolio))
        result[np.argmin(portfolio)] = -10_000
        result[np.argmax(portfolio)] = 10_001
        return self.finalize_portfolio_2(result)


if __name__ == '__main__':
    train_and_save_weights()
