import torch
import torch.nn as nn
from skorch import NeuralNetRegressor

from bsdtq import DataFrameDataset # Reusing the dataset from your bsdtq module

class LSTMModule(nn.Module):
    """
    A PyTorch module for a simple LSTM-based regressor.
    """
    def __init__(
        self,
        input_size,
        hidden_size=50,
        num_layers=1,
        output_size=1,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Important: expects input shape (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # The input X from mlforecast/skorch is (batch, features).
        # LSTM expects (batch, seq_len, features).
        # We treat the sequence length as 1 for this type of model.
        X = X.unsqueeze(1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)

        # We only need the output of the last time step
        lstm_out, _ = self.lstm(X, (h0, c0))
        
        # Get the output from the last sequence step
        last_time_step_out = lstm_out[:, -1, :]
        
        out = self.fc(last_time_step_out)
        # Squeeze the output to be (batch_size,) instead of (batch_size, 1)
        # to comply with mlforecast's expectations.
        return out.squeeze(-1)


class SkorchLSTM(NeuralNetRegressor):
    """
    A skorch wrapper for the LSTMModule that dynamically infers the
    'module__input_size' parameter from the training data X.
    """
    def __init__(self, *args, **kwargs):
        # Set default criterion if not provided
        if 'criterion' not in kwargs:
            kwargs['criterion'] = nn.MSELoss
        
        # Set the dataset for DataFrame compatibility
        kwargs['dataset'] = DataFrameDataset
        
        # Disable shuffling by default for time series
        kwargs.setdefault('iterator_train__shuffle', False)
        kwargs.setdefault('iterator_valid__shuffle', False)

        # Set verbose to False
        kwargs.setdefault('verbose', False)

        super().__init__(*args, **kwargs)

    def fit(self, X, y=None, **fit_params):
        if not self.initialized_:
            # Infer 'input_size' from the shape of X before initialization
            input_size = X.shape[1]
            self.set_params(module__input_size=input_size)
        return super().fit(X, y, **fit_params)
