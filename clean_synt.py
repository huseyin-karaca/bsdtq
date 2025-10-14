"""
clean_synt.py
Synthetic dataset experiment runner.
Performs AutoMLForecast training with inner-Optuna model search on synthetic dataset.
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
from sklearn.datasets import make_regression
from statsmodels.tsa.arima_process import arma_generate_sample

from lstm_model import SkorchLSTM, LSTMModule
from bsdtq import (
    SkorchWrapper,
    BoostedSDTR,
    DataFrameDataset,
    _CustomLoss,
    get_time_split,
)
from clean_configs import *

optuna.logging.set_verbosity(optuna.logging.WARNING)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    seed = 0
    set_seed(seed)

    # -------------------------------------------------------------------
    # 1. PREPARE DATA
    # -------------------------------------------------------------------

    horizon = 12
    freq = 1
    train_length = 200
    num_series = 5
    n_features = 50
    n_informative = 10
    noise = 0.3
    ar_params = 0.85
    ma_params = 0.95
    scale = 200.0

    all_series_dfs = []
    total_length = train_length + horizon

    for i in range(num_series):
        # 1. Generate features and the static regression component using sklearn
        X, y_reg = make_regression(
            n_samples=total_length,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=seed + i,  # Different seed for each series for variety
        )

        # 2. Generate the dynamic time series component using an ARMA process
        # The coefficients need to be prepended with 1 for the statsmodels function
        ar = np.r_[1, -np.array(ar_params)]
        ma = np.r_[1, np.array(ma_params)]
        y_arma = arma_generate_sample(ar=ar, ma=ma, nsample=total_length, scale=0.5)

        # 3. Combine the two components
        # y is the sum of the value predictable from features and the time-dependent noise
        y = y_reg + scale * y_arma

        # 4. Assemble into a DataFrame
        feature_names = [f'feature_{j}' for j in range(n_features)]
        series_df = pd.DataFrame(X, columns=feature_names)
        series_df['y'] = y
        series_df['ds'] = np.arange(1, total_length + 1)
        series_df['unique_id'] = f'series_{i + 1}'

        all_series_dfs.append(series_df)

    # Concatenate all individual series into a single DataFrame
    full_df = pd.concat(all_series_dfs, ignore_index=True)

    # Reorder columns for clarity
    cols = ['unique_id', 'ds', 'y'] + [col for col in full_df.columns if col not in ['unique_id', 'ds', 'y']]
    full_df = full_df[cols]

    # Normalize each series (min-max)
    full_df["y"] = full_df.groupby("unique_id",observed=False)["y"] \
                .transform(lambda x: (x - x.mean()) / x.std())

    # 5. Create train and test splits
    test_df = full_df.groupby('unique_id', group_keys=False).tail(horizon)
    train_df = full_df.drop(test_df.index)

    # Report dataset statistics
    dataset_stats = pd.DataFrame(
        data={
            "Train Length": train_df.shape[0],
            "Test Length": test_df.shape[0],
            "Features": train_df.shape[1],
            "Target Mean": train_df["y"].mean(),
            "Target Std": train_df["y"].std(),
        },
        index=['Synthetic']).T.rename_axis("Statistic", axis=1)

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
        .rename_axis("Models", axis=1)

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
