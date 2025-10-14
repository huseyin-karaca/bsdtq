import torch
from utilsforecast.losses import mse

def my_bsdtr_config(trial):

    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }


    """Optuna search space for BoostedSDTR."""
    # transform_type = trial.suggest_categorical("module__transform_type", ["identity", "linear", "mlp"])
    transform_type = "linear"
    optimizer_key = trial.suggest_categorical("Optimizer", list(OPTIMIZERS))
    
    params = {
        "module__num_boosting_layers": trial.suggest_categorical("Boosting Layers", [1, 2, 4]),
        "module__transform_type": transform_type,
        "module__depth": trial.suggest_categorical("Depth", [2, 4, 6]),
        "module__num_trees": trial.suggest_categorical("Trees per Layer", [1, 4, 16]),
        "module__hidden_dim": trial.suggest_categorical("Hidden Dimension", [None, 16, 32]),
        "module__lmbda": trial.suggest_categorical("L1 Regularization", [0.01, 0.1, 0.2]),
        "module__lmbda2": trial.suggest_categorical("L2 Regularization", [0.01, 0.1, 0.2]),
        "lr": trial.suggest_categorical("Learning Rate", [0.005, 0.01, 0.05]),
        "max_epochs": trial.suggest_categorical("Epochs", [10, 20]),
        "optimizer": OPTIMIZERS[optimizer_key],
        "batch_size": trial.suggest_categorical("Batch Size", [64, 128]),
    }

    if transform_type == "mlp":
        params["module__transform_hidden_dim"] = trial.suggest_categorical("MLP Hidden Dimension", [32, 64, 128])
        
    return params

def my_bsdtr_mlp_config(trial):

    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }


    """Optuna search space for BoostedSDTR."""
    # transform_type = trial.suggest_categorical("module__transform_type", ["identity", "linear", "mlp"])
    transform_type = "mlp"
    optimizer_key = trial.suggest_categorical("Optimizer", list(OPTIMIZERS))
    
    params = {
        "module__num_boosting_layers": trial.suggest_categorical("Boosting Layers", [1, 2, 4]),
        "module__transform_type": transform_type,
        "module__depth": trial.suggest_categorical("Depth", [2, 4, 6]),
        "module__num_trees": trial.suggest_categorical("Trees per Layer", [1, 4, 16]),
        "module__hidden_dim": trial.suggest_categorical("Hidden Dimension", [None, 16, 32]),
        "module__lmbda": trial.suggest_categorical("L1 Regularization", [0.01, 0.1, 0.2]),
        "module__lmbda2": trial.suggest_categorical("L2 Regularization", [0.01, 0.1, 0.2]),
        "lr": trial.suggest_categorical("Learning Rate", [0.005, 0.01, 0.05]),
        "max_epochs": trial.suggest_categorical("Epochs", [10, 20]),
        "optimizer": OPTIMIZERS[optimizer_key],
        "batch_size": trial.suggest_categorical("Batch Size", [64, 128]),
    }

    if transform_type == "mlp":
        params["module__transform_hidden_dim"] = trial.suggest_categorical("MLP Hidden Dimension", [32, 64, 128])
        
    return params

def my_sdtr_config(trial):

    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }


    """Optuna search space for BoostedSDTR."""
    # transform_type = trial.suggest_categorical("module__transform_type", ["identity", "linear", "mlp"])
    transform_type = "identity"
    optimizer_key = trial.suggest_categorical("Optimizer", list(OPTIMIZERS))
    
    params = {
        "module__num_boosting_layers": trial.suggest_categorical("Boosting Layers", [1]),
        "module__transform_type": transform_type,
        "module__depth": trial.suggest_categorical("Depth", [2, 4, 6]),
        "module__num_trees": trial.suggest_categorical("Trees per Layer", [1]),
        "module__hidden_dim": trial.suggest_categorical("Hidden Dimension", [None, 16, 32]),
        "module__lmbda": trial.suggest_categorical("L1 Regularization", [0.01, 0.1, 0.2]),
        "module__lmbda2": trial.suggest_categorical("L2 Regularization", [0.01, 0.1, 0.2]),
        "lr": trial.suggest_categorical("Learning Rate", [0.005, 0.01, 0.05]),
        "max_epochs": trial.suggest_categorical("Epochs", [10, 20]),
        "optimizer": OPTIMIZERS[optimizer_key],
        "batch_size": trial.suggest_categorical("Batch Size", [64, 128]),
    }

    if transform_type == "mlp":
        params["module__transform_hidden_dim"] = trial.suggest_categorical("MLP Hidden Dimension", [32, 64, 128])
        
    return params



def my_lgbm_config(trial):
    return {
        "n_estimators": trial.suggest_categorical("Estimators", [50, 100, 200]),
        "learning_rate": trial.suggest_categorical("Learning Rate", [0.1, 0.3, 0.5]),
        "num_leaves": trial.suggest_categorical("Leaves", [127, 255, 511]),
        "max_depth": trial.suggest_categorical("Depth", [-1, 15, 20]),
        "min_child_samples": trial.suggest_categorical("Min. Child Samples", [1, 5, 10]),
        "lambda_l1": trial.suggest_categorical("L1 Regularization", [0, 0.001]),
        "lambda_l2": trial.suggest_categorical("L2 Regularization", [0, 0.001]),
        "verbose": -1,
    }

def my_xgb_config(trial):
    return {
        "n_estimators": trial.suggest_categorical("Estimators", [50, 100, 200]),
        "learning_rate": trial.suggest_categorical("Learning Rate", [0.1, 0.3, 0.5]),
        "max_depth": trial.suggest_categorical("Depth", [10, 15, 20]),
        "gamma": trial.suggest_categorical("Min. Split Loss", [0, 0.1]),
        "reg_alpha": trial.suggest_categorical("L1 Regularization", [0, 0.001]),
        "reg_lambda": trial.suggest_categorical("L2 Regularization", [1, 0.1, 0]),
        "min_child_weight": trial.suggest_categorical("Min. Child Samples", [1, 2, 5]),
    }

def my_catboost_config(trial):
    return {
        "iterations": trial.suggest_categorical("Estimators", [50, 100, 200]),
        "learning_rate": trial.suggest_categorical("Learning Rate", [0.1, 0.3]),
        "depth": trial.suggest_categorical("Depth", [8, 10, 12]),
        "l2_leaf_reg": trial.suggest_categorical("L2 Regularization", [1, 0.1, 0.01]),
        "min_data_in_leaf": trial.suggest_categorical("Min. Child Samples", [1, 2]),
        "verbose": False,
    }

def my_rf_config(trial):
    return {
        "n_estimators": trial.suggest_categorical("Estimators", [10, 50, 100]),
        "max_depth": trial.suggest_categorical("Depth", [None, 20, 30]),
        "min_samples_leaf": trial.suggest_categorical("Min. Child Samples", [1, 2, 5]),
        "min_samples_split": trial.suggest_categorical("Min. Samples Split", [2, 5, 10]),
        "max_features": trial.suggest_categorical("Max. Features", ["sqrt", "log2", 1.0]),
    }

def my_lasso_config(trial):
    return {
        "alpha": trial.suggest_categorical("Alpha", [0.0001, 0.001, 100.0, 1000.0]),
        "max_iter": trial.suggest_categorical("Max. Iterations", [100, 200]),
    }

def my_lstm_config(trial):
    """Optuna search space for SkorchLSTM."""
    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    optimizer_key = trial.suggest_categorical("Optimizer", list(OPTIMIZERS))
    
    params = {
        "lr": trial.suggest_categorical("Learning Rate", [0.001, 0.005, 0.01]),
        "max_epochs": trial.suggest_categorical("Epochs", [10, 20, 30]),
        "batch_size": trial.suggest_categorical("Batch Size", [64, 128]),
        "optimizer": OPTIMIZERS[optimizer_key],
        "module__hidden_size": trial.suggest_categorical("Hidden Size", [50, 100, 150]),
        "module__num_layers": trial.suggest_categorical("Num Layers", [1, 2, 3]),
        "module__dropout": trial.suggest_categorical("Dropout", [0.1, 0.2, 0.3]),
    }
    return params


def my_init_config(trial):
    return {
        "lags": [],
        "lag_transforms": None,
        "date_features": [],
        "target_transforms": [],
    }


def my_fit_config(trial):
    return {"static_features": []}



def loss_fn(df, train_df):
    return mse(df, models=["model"])["model"].mean()
