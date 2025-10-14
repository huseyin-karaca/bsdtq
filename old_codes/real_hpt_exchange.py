# IMPORTS
import numpy as np
import pandas as pd
import random 
import torch
import os

from lightgbm import LGBMRegressor
from sdt_sklearn import BoostedSDT, BoostedSDTQ

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingStd, RollingMean
from datasetsforecast.long_horizon import LongHorizon, LongHorizonInfo

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse 

from auxiliary import cumsum_plot


# PARAMETERS

hpt_lgb = True
hpt_bsdtq = True
hpt_bsdt = True

hpt_lgb = False
hpt_bsdtq = False
hpt_bsdt = False


group = "Exchange"
info = LongHorizonInfo[group]
freq = info.freq
horizon = info.horizons[0]
N = horizon

window_sizes = [4, 12, 24, 36]
shifts = [1, 24, 48, 72, 96]
lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,24]

# Results of hyperparameter search for future reference
best_params_lgb = {'num_leaves': 31, 'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.05} #for exchange N = 1*horizon score =-0.03624026011772086
best_params_bsdt = {'boosting_iters': 1, 'criterion': 'mse', 'depth': 3, 'epochs': 20, 'lr_sdt': 0.1} # for exchange 1*horizon, score: -0.012500967864126938
best_params_bsdtq = {'boosting_iters': 1, 'criterion': 'mse', 'depth': 3, 'epochs': 20, 'how_many_grad': 50, 'iterate_until_converge': 20, 'lr_q': 0.1, 'lr_sdt': 0.1, 'output_dim': 5} # for exchange, 1*horizon,  score  -0.0045336674157425046

# Fix randomness for reproducibility of the results
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# GET DATA

y_df, X_df, S_df = LongHorizon.load(directory = ".",group = group)

# Normalize each series (min-max)
y_df["y"] = y_df.groupby("unique_id",observed=False)["y"] \
            .transform(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)

y_df['ds'] = y_df['ds'].astype('datetime64[ns]')
X_df['ds'] = X_df['ds'].astype('datetime64[ns]')


# Get lagging - rolling features easily by mlf.preprocess
my_init = {
    "lags": lags,
    "lag_transforms": {
        shift: [f(window_size=w)
                for w in window_sizes for f in (RollingMean, RollingStd)]
        for shift in shifts
    }
}
mlf = MLForecast(
    models =[],
    freq = freq,
    **my_init
)
lag_rol_features = mlf.preprocess(y_df)


df = lag_rol_features.merge(X_df, on = ["unique_id","ds"])
df = df.sort_values(['unique_id', 'ds'])

# Train - test split
test = df.groupby("unique_id",group_keys = False ).tail(horizon)
test_X = test.drop(columns = ["y","unique_id","ds"])
test_y = test["y"]

train = df.drop(index = test.index).groupby("unique_id",group_keys = False ).tail(10*horizon).sample(n=N, random_state=seed)
train_X = train.drop(columns = ["y","unique_id","ds"])
train_y = train["y"]


# MODELS 

# Boosted SDT with Q
if hpt_bsdtq:
    param_dist_bsdtq = {
        "epochs": [20],
        "depth": [3],
        "how_many_grad": [10, 50],
        "lr_sdt": [1e-1, 1e-2],
        "lr_q": [1e-2, 3e-2, 1e-1],
        "boosting_iters": [1],
        "criterion": ["mse"],
        "output_dim":[5,10],
        "iterate_until_converge": [20,50]
    } 

    rs_bsdtq = GridSearchCV(
        BoostedSDTQ(),
        param_dist_bsdtq,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=4,
    )

    rs_bsdtq.fit(train_X, train_y)
    print(rs_bsdtq.best_params_, rs_bsdtq.best_score_)
    preds_bsdtq = rs_bsdtq.best_estimator_.predict(test_X)
    best_params_bsdtq = rs_bsdtq.best_params_

else:
    model_bsdtq = BoostedSDTQ(**best_params_bsdtq)
    model_bsdtq.fit(train_X, train_y)
    preds_bsdtq = model_bsdtq.predict(test_X)


mse_bsdtq = mse(preds_bsdtq, test_y)




# Boosted SDT
if hpt_bsdt:
    param_dist_bsdt = {
        "epochs": [10,20],
        "depth": [2,3,4],
        "lr_sdt": [1e-1, 1e-2],
        "boosting_iters": [1,2,4],
        "criterion": ["mse"],
        "output_dim":[test_X.shape[1]]
    }

    rs_bsdt = GridSearchCV(
        BoostedSDT(),
        param_dist_bsdt,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=4,
    )

    rs_bsdt.fit(train_X,train_y)
    print(rs_bsdt.best_params_, rs_bsdt.best_score_)
    preds_bsdt = rs_bsdt.best_estimator_.predict(test_X)
    best_params_bsdt = rs_bsdt.best_params_

else:
    model_bsdt = BoostedSDT(output_dim=test_X.shape[1], **best_params_bsdt)
    model_bsdt.fit(train_X, train_y)
    preds_bsdt = model_bsdt.predict(test_X)


mse_bsdt = mse(preds_bsdt, test_y)



# LightGBM
if hpt_lgb: 
    param_dist_lgb = {
        "n_estimators":  [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves":    [31, 50, 100],
        "max_depth":     [-1, 5, 10],
    }

    rs_lgb = GridSearchCV(
        LGBMRegressor(random_state=seed, verbosity=-1),
        param_dist_lgb,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=4,
    )

    rs_lgb.fit(train_X, train_y)
    print(rs_lgb.best_params_, rs_lgb.best_score_)
    best_params_lgb = rs_lgb.best_params_
    preds_lgb = rs_lgb.best_estimator_.predict(test_X)
else:

    model_lgb = LGBMRegressor(**best_params_lgb).fit(train_X,train_y)
    preds_lgb = model_lgb.predict(test_X)


mse_lgb = mse(preds_lgb, test_y)


print(f"mse_bsdtq:{mse_bsdtq}\nmse_bsdt: {mse_bsdt}\nmse_lgb:  {mse_lgb}")



# VISUALIZE RESULTS

results = pd.DataFrame({
    "unique_id": test.reset_index()["unique_id"],
    "ds": test.reset_index()["ds"],
    "y": test["y"].values,
    "lgb": preds_lgb,
    "bsdt": preds_bsdt,
    "bsdtq": preds_bsdtq,
})

# notes to be added to the figure side
data_params = {
    "Dataset": info.name,
    "window_sizes": window_sizes,
    "lags": lags,
    "shifts": shifts,
    "N": N,
    "seed": seed
}

plot_text = "\n\n".join([
    "LGB\n" + "\n".join(f"{k}: {v}" for k, v in best_params_lgb.items()),
    "BSDT\n" + "\n".join(f"{k}: {v}" for k, v in best_params_bsdt.items()),
    "BSDTQ\n" + "\n".join(f"{k}: {v}" for k,
                            v in best_params_bsdtq.items()),
    "Data\n" + "\n".join(f"{k}: {v}" for k, v in data_params.items())
])


cumsum_plot(results,plot_text)

