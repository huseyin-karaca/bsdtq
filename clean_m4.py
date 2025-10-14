"""
clean_m4.py
M4 dataset experiment runner.
Performs AutoMLForecast training with inner-Optuna model search on M4 dataset.
Compares AutoModel performance against other baseline models.
"""

# -------------------------------------------------------------------
# 0. IMPORTS AND REPRODUCIBILITY SETTINGS
# -------------------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import torch
import optuna

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingStd, RollingMean
from mlforecast.auto import (
    AutoMLForecast,
    AutoModel,
    AutoLightGBM,
    AutoXGBoost,
    AutoCatboost,
    AutoRandomForest,
    AutoLasso,
)
from utilsforecast.losses import mse
from utilsforecast.evaluation import evaluate
from datasetsforecast.m4 import M4, M4Info

from lstm_model import SkorchLSTM, LSTMModule
from bsdtq import (
    SkorchWrapper,
    BoostedSDTR,
    DataFrameDataset,
    _CustomLoss,
    get_time_split,
)
from clean_configs import *

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = 123
    set_seed(seed)

    # -------------------------------------------------------------------
    # 1. PREPARE DATA
    # -------------------------------------------------------------------

    group = "Quarterly"
    horizon = M4Info[group].horizon
    freq = 1  # Quarterly frequency as integer
    train_length = 200
    num_series = 5

    lags = [1, 2, 4, 8, 12, 20]
    mean_windows = [4, 8, 12, 20]
    std_windows = [4, 8, 12, 20]
    shifts = [1]

    # Load M4 data
    y_df, X_df, S_df = M4.load(
        directory="/Users/huseyinkaraca/Documents/datasets",
        group=group
    )

    # Normalize target
    y_df["y"] = y_df.groupby("unique_id", observed=False)["y"] \
        .transform(lambda x: (x - x.mean()) / x.std())

    y_df["ds"] = y_df["ds"].astype(int)

    # Prepare lag and rolling features
    my_init = {
        "lags": lags,
        "lag_transforms": {
            shift: [
                f(window_size=w)
                for w in (mean_windows + std_windows) for f in (RollingMean, RollingStd)
            ]
            for shift in shifts
        }
    }

    lag_rol_features = MLForecast(
        models=[],
        freq=freq,
        **my_init
    ).preprocess(y_df)

    # Subsample to simulate low data regime
    unique_ids = lag_rol_features["unique_id"].unique()
    selected_ids = np.random.choice(unique_ids, size=num_series, replace=False)
    lag_rol_features = lag_rol_features[lag_rol_features["unique_id"].isin(selected_ids)]

    # Sort and split train/test
    df = lag_rol_features.sort_values(["unique_id", "ds"])
    test_df = df.groupby("unique_id", group_keys=False).tail(horizon)
    train_df = df.drop(index=test_df.index).groupby("unique_id", group_keys=False).tail(train_length)


    # Report dataset statistics 
    dataset_stats = pd.DataFrame(
        data = {
            "Train Length": train_df.shape[0],
            "Test Length": test_df.shape[0],
            "Features": train_df.shape[1],
            "Target Mean": train_df["y"].mean(),
            "Target Std": train_df["y"].std(),
        }, 
        index = ['M4']).T.rename_axis("Statistic", axis = 1)

    
    # -------------------------------------------------------------------
    # 2. TRAIN MODEL
    # -------------------------------------------------------------------

    amlf = AutoMLForecast(
        models={
            "BSDT-Q": AutoModel(
                model=SkorchWrapper(
                    verbose=0,
                    module=BoostedSDTR,
                    criterion=_CustomLoss,
                    dataset=DataFrameDataset,
                    train_split=get_time_split(0.1),
                ),
                config=my_bsdtr_config,
            ),
            "LightGBM": AutoLightGBM(config=my_lgbm_config),
            "XGBoost": AutoXGBoost(config=my_xgb_config),
            "Catboost": AutoCatboost(config=my_catboost_config),
            "RandomForest": AutoRandomForest(config=my_rf_config),
            "Lasso": AutoLasso(config=my_lasso_config),
            "LSTM": AutoModel(
                model=SkorchLSTM(
                    module=LSTMModule,
                    # -- skorch parameters --
                    max_epochs=10,
                    lr=0.01,
                    # -- module parameters --
                    module__hidden_size=100,
                    module__num_layers=2,
                    ),
                config = my_lstm_config),
            "BSDT-MLP": AutoModel(
                model=SkorchWrapper(
                    verbose=0,
                    module=BoostedSDTR,
                    criterion=_CustomLoss,
                    dataset=DataFrameDataset,
                    train_split=get_time_split(0.1),
                ),
                config=my_bsdtr_mlp_config,
            ),
            "SDT": AutoModel(
                model=SkorchWrapper(
                    verbose=0,
                    module=BoostedSDTR,
                    criterion=_CustomLoss,
                    dataset=DataFrameDataset,
                    train_split=get_time_split(0.1),
                ),
                config=my_sdtr_config,
            ),
        },
        freq=freq,
        init_config=my_init_config,
        fit_config=my_fit_config,
    )

    amlf.fit(
        df=train_df,
        n_windows=1,
        h=horizon,
        num_samples=10,
        loss=loss_fn,
        refit=True,
    )

    # -------------------------------------------------------------------
    # 3. PREDICT and EVALUATE
    # -------------------------------------------------------------------

    preds = amlf.predict(
        X_df=test_df.drop(columns=["y"]),
        h=horizon,
    )

    merged = test_df[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"])

    eval_df = evaluate(merged, metrics=[mse], agg_fn="mean") \
        .set_index("metric") \
        .T \
        .rename(columns={"mse": "MSE"}) \
        .rename_axis("Model", axis=1)

    model_hyperparams = []
    for model_name, result in amlf.results_.items():
        best_trial = result.best_trial

        # training time
        duration = best_trial.datetime_complete - best_trial.datetime_start
        eval_df.loc[model_name, "Training Time (s)"] = duration.total_seconds()

        # hyperparams
        dist = best_trial.distributions
        model_hyperparams.append(pd.DataFrame({k: str(v.choices) for k, v in dist.items()}, index = [model_name]).T.unstack().to_frame())   

    hyperparams = pd.concat(model_hyperparams, axis = 0)
    hyperparams.rename(columns = {0:"Search Space"}, inplace = True)
    hyperparams.index.set_names(["Model","Parameter"], inplace=True)
    hyperparams = hyperparams.sort_index()

    return eval_df, dataset_stats, hyperparams

if __name__ == "__main__":
    eval_df, dataset_stats, hyperparams = main()
    print(eval_df)
    print(dataset_stats)
