
# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

from lightgbm import LGBMRegressor
from sdt_sklearn import BoostedSDT, BoostedSDTQ

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse 
from auxiliary import cumsum_plot


# PARAMETERS
hpt_bsdtq = True
hpt_bsdt = True
hpt_lgb = True

hpt_bsdtq = False
hpt_bsdt = False
hpt_lgb = False

n_iters = 100
n_series = 1
n_total = 50
n_relevant = 5
n_irrelevant = n_total-n_relevant
n_obs = 196
n_obs_train = 100
series_ids = [f"Hx{i}" for i in range(1, n_series + 1)]
ds = np.arange(1, n_obs + 1)

# Results of hyperparameter search for future reference
best_params_bsdtq = {'boosting_iters': 1, 'criterion': 'mse', 'depth': 4, 'epochs': 20, 'how_many_grad': 50, 'iterate_until_converge': 50, 'lr_q': 0.01, 'lr_sdt': 0.1, 'output_dim': 10}
best_params_bsdtq = {'boosting_iters': 1, 'criterion': 'mse', 'depth': 3, 'epochs': 20, 'how_many_grad': 50, 'iterate_until_converge':100, 'lr_q': 0.5, 'lr_sdt': 0.1, 'output_dim': 5}
best_params_bsdt = {'boosting_iters': 1, 'criterion': 'mse', 'depth': 4, 'epochs': 20, 'lr_sdt': 0.1}
best_params_lgb = {'learning_rate': 0.05, 'max_depth': -1, 'n_estimators': 200, 'num_leaves': 31}


# Fix randomness for reproducibility of the results
seed = 0
rng = np.random.default_rng(seed)
random.seed(seed)
np.random.seed(seed)


# SYNTHETIC DATA GENERATION

# Random generator and weights for relevant features
weights = rng.uniform(0.5, 1.5, size=n_relevant)
weights = [10, 7,6,5,4]
weights = [round((i/sum(weights)),2) for i in weights]

# Generate exogenous features
relevant = rng.normal(size=(n_series, n_obs, n_relevant))
irrelevant = rng.normal(size=(n_series, n_obs, n_irrelevant))

# Compute y as linear combination of relevant features + noise
y = np.einsum("sti,i->st", relevant, weights) + rng.normal(scale=0.03, size=(n_series, n_obs))

# Build the DataFrame in long format
data = {
    "unique_id": np.repeat(series_ids, n_obs),
    "ds": np.tile(ds, n_series),
    "y": y.flatten(),
}

# Add relevant features
for i in range(n_relevant):
    data[f"relevant_{i+1}"] = relevant[:, :, i].flatten()

# Add irrelevant features
for j in range(n_irrelevant):
    data[f"irrelevant_{j+1}"] = irrelevant[:, :, j].flatten()

df_synthetic = pd.DataFrame(data)

# Normalize each series (min-max)
df_synthetic["y"] = df_synthetic.groupby("unique_id",observed=False)["y"] \
                .transform(lambda x: 2*(x - x.min())/(x.max() - x.min()) - 1)

# Train - test splitting
test_synt = df_synthetic[df_synthetic["ds"]>n_obs_train] 
train_synt = df_synthetic.drop(test_synt.index)

Y_df_synt = train_synt[["unique_id","ds","y"]]
X_df_synt = train_synt.drop(columns="y")

Y_test_df_synt = test_synt[["unique_id","ds","y"]]
X_test_df_synt = test_synt.drop(columns="y")

train_X = X_df_synt.set_index(["unique_id","ds"])
train_y = Y_df_synt.set_index(["unique_id","ds"])

test_X = X_test_df_synt.set_index(["unique_id","ds"])
test_y = Y_test_df_synt.set_index(["unique_id","ds"])



# MODELS

# Boosted SDT with Q
if hpt_bsdtq: 
    param_dist_bsdtq = {
        "epochs": [20],
        "depth": [3,4],
        "how_many_grad": [50],
        "lr_sdt": [1e-1, 1e-2],
        "lr_q": [1e-2, 1e-3],
        "boosting_iters": [1],
        "criterion": ["mse"],
        "output_dim":[1,10],
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

# Visualize Q Matrix
q0 = model_bsdtq.Qs_["Q_0"]
plt.figure
plt.matshow(q0)
plt.colorbar(location='bottom', ticks=[-0.02, 0.08, 0.18])
plt.savefig("q0.pdf",bbox_inches="tight")


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
        LGBMRegressor(random_state=0, verbosity=-1),
        param_dist_lgb,
        #n_iter=100,
        cv=3,
        scoring="neg_mean_squared_error",
        #random_state=0,
        n_jobs=-1,
        verbose=4,
    )

    rs_lgb.fit(train_X, train_y)
    print(rs_lgb.best_params_, rs_lgb.best_score_)
    best_params_lgb = rs_lgb.best_params_
    preds_lgb = rs_lgb.best_estimator_.predict(test_X)

else:
    model_lgb = LGBMRegressor(**best_params_lgb)
    model_lgb.fit(train_X, train_y)
    preds_lgb = model_lgb.predict(test_X)

mse_lgb = mse(test_y, preds_lgb)


print(f"mse_bsdtq:{mse_bsdtq}\nmse_bsdt: {mse_bsdt}\nmse_lgb:  {mse_lgb}")



# VISUALIZE RESULTS

results = pd.DataFrame({
    "unique_id": test_synt["unique_id"],
    "ds": test_synt["ds"],
    "y": test_synt["y"],
    "lgb": preds_lgb,
    "bsdt": preds_bsdt,
    "bsdtq": preds_bsdtq,
})


# notes to be added to the figure side
data_params = {
    "Dataset": "Synthetic",
    "N": 100,
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
