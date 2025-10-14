import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from skorch import NeuralNetRegressor
from single_sdt import SDTR

class DataFrameDataset(Dataset):
    """
    Custom PyTorch Dataset to handle Pandas DataFrames, converting rows
    to tensors on the fly.
    """
    def __init__(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = X
            
        if y is not None:
             if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                self.y = y.values
             else:
                self.y = y
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, y_tensor
        # Return a dummy y tensor to satisfy skorch's requirement during inference
        return x_tensor, torch.tensor(0.0)

class BoostedSDTR(nn.Module):
    """
    An ensemble of SDTR models with boosting-like behavior.
    'in_features' is determined at initialization.
    """
    def __init__(
        self,
        in_features,
        num_boosting_layers=3,
        transform_type='identity',
        transform_hidden_dim=64,
        **sdt_params
    ):
        super().__init__()
        self.in_features = in_features
        self.num_boosting_layers = num_boosting_layers
        self.transform_type = transform_type
        self.transform_hidden_dim = transform_hidden_dim
        self.sdt_params = sdt_params
        
        self.transformers = nn.ModuleList()
        self.sdtrs = nn.ModuleList()
        
        self._initialize_layers()

    def _initialize_layers(self):
        """Build the transformer and SDTR layers."""
        for _ in range(self.num_boosting_layers):
            if self.transform_type == 'identity':
                transformer = nn.Identity()
                transformer_out_features = self.in_features
            elif self.transform_type == 'linear':
                transformer = nn.Linear(self.in_features, self.in_features, bias=False)
                transformer_out_features = self.in_features
            elif self.transform_type == 'mlp':
                transformer = nn.Sequential(
                    nn.Linear(self.in_features, self.transform_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.transform_hidden_dim, self.in_features)
                )
                transformer_out_features = self.in_features
            else:
                raise ValueError(f"Unknown transform_type: {self.transform_type}")
            
            self.transformers.append(transformer)

            sdtr_model = SDTR(in_features=transformer_out_features, **self.sdt_params)
            self.sdtrs.append(sdtr_model)

    def forward(self, X):
        total_prediction = None
        total_reg_loss = 0.
        total_l1_loss = 0.

        for transformer, sdtr in zip(self.transformers, self.sdtrs):
            transformed_X = transformer(X)
            prediction, reg_loss, l1_loss = sdtr(transformed_X)
            
            if total_prediction is None:
                total_prediction = prediction
            else:
                total_prediction += prediction

            if reg_loss is not None:
                total_reg_loss += reg_loss
            if l1_loss is not None:
                total_l1_loss += l1_loss

        return total_prediction, total_reg_loss, total_l1_loss

class _CustomLoss(nn.Module):
    """
    Custom loss to handle the tuple output (response, reg_loss, l1_loss)
    from the BoostedSDTR model.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        response, reg_loss, l1_loss = y_pred
        y_true = y_true.reshape_as(response)
        
        loss = self.mse(response, y_true)
        reg_loss = torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0)
        l1_loss = torch.nan_to_num(l1_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss + reg_loss + l1_loss
        

class SkorchWrapper(NeuralNetRegressor):
    """
    A NeuralNetRegressor wrapper that dynamically infers the
    'module__in_features' parameter from the training data X before
    the module is initialized.
    """
    def __init__(self, *args, **kwargs):
        # disable any shuffling unless user overwrites
        kwargs.setdefault('iterator_train__shuffle', False)
        kwargs.setdefault('iterator_valid__shuffle', False)
        super().__init__(*args, **kwargs)

    def fit(self, X, y=None, **fit_params):
        if not self.initialized_:
            # Infer 'in_features' from the shape of X. This must be done
            # before super().fit() calls .initialize(), which in turn
            # instantiates the module and optimizer.
            in_features = X.shape[1]
            self.set_params(module__in_features=in_features)
        return super().fit(X, y, **fit_params)

# ------------------------------------------------------------------
# helper for deterministic time-based split
# ------------------------------------------------------------------

def get_time_split(valid_fraction: float = 0.1):
    """Return a callable acceptable by skorch.train_split.

    The last ``valid_fraction`` portion of the data is used for validation.
    ``valid_fraction`` can also be an int meaning the last N rows.
    """
    from torch.utils.data import Subset

    def _split(dataset, y):
        """dataset is already a DataFrameDataset; build deterministic index split."""
        n_total = len(dataset)
        n_valid = int(valid_fraction * n_total) if isinstance(valid_fraction, float) else valid_fraction
        idx_train = list(range(0, n_total - n_valid))
        idx_valid = list(range(n_total - n_valid, n_total))
        return Subset(dataset, idx_train), Subset(dataset, idx_valid)

    return _split

