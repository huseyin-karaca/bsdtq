"""
Generates LaTeX tables for experimental results and saves them to the
'manuscript/table_figures/' directory.

This script performs the following steps:
1.  Loads experimental results (evaluation metrics, dataset statistics, and
    hyperparameters) from pickled files, or runs the experiments if the files
    do not exist.
2.  Generates and saves a LaTeX table for dataset statistics.
3.  Generates and saves LaTeX tables for model performance (MSE) and training
    times, highlighting the best results.
4.  Generates and saves a LaTeX table summarizing the hyperparameter search
    space for each model.
"""

import numpy as np
import pandas as pd
import pickle
import os

# -------------------------------------------------------------------
# Helper: compact scientific formatter using NumPy (e.g., 1.23e4)
# -------------------------------------------------------------------


def _sci_fmt(val: float, precision: int = 2, exp_digits: int = 1) -> str:
    """Return *val* as scientific string with NumPy's formatter.

    Handles NaN/inf gracefully by returning a hyphen.
    """
    if pd.isna(val) or np.isinf(val):
        return "-"
    return np.format_float_scientific(val, precision=precision, unique=False, exp_digits=exp_digits)

# -------------------------------------------------------------------
# 1. LOAD DATA OR RUN EXPERIMENTS
# -------------------------------------------------------------------

# Import the main experiment function from each dataset-specific script
from clean_m4 import main as main_m4
from clean_weather import main as main_weather
from clean_exchange import main as main_exchange
from clean_ett import main as main_ett
from clean_synt import main as main_synt

def load_or_run_experiments():
    """
    Loads results from disk if they exist, otherwise runs all experiments.
    """
    if os.path.exists("all_eval_dfs.pkl"):
        print("Loading cached results from disk...")
        with open("all_eval_dfs.pkl", "rb") as f:
            all_eval_dfs = pickle.load(f)
        with open("all_dataset_stats.pkl", "rb") as f:
            all_dataset_stats = pickle.load(f)
        with open("all_hyperparams.pkl", "rb") as f:
            all_hyperparams = pickle.load(f)
    else:
        print("Running experiments to generate results...")
        eval_df_m4, dataset_stats_m4, hyperparams_m4 = main_m4()
        eval_df_weather, dataset_stats_weather, hyperparams_weather = main_weather()
        eval_df_exchange, dataset_stats_exchange, hyperparams_exchange = main_exchange()
        eval_df_ett, dataset_stats_ett, hyperparams_ett = main_ett()
        eval_df_synt, dataset_stats_synt, hyperparams_synt = main_synt()

        all_eval_dfs = {
            "M4": eval_df_m4, "Weather": eval_df_weather,
            "Exchange": eval_df_exchange, "ETTm2": eval_df_ett,
            "Synthetic": eval_df_synt,
        }
        all_dataset_stats = {
            "M4": dataset_stats_m4, "Weather": dataset_stats_weather,
            "Exchange": dataset_stats_exchange, "ETTm2": dataset_stats_ett,
            "Synthetic": dataset_stats_synt,
        }
        all_hyperparams = {
            "M4": hyperparams_m4, "Weather": hyperparams_weather,
            "Exchange": hyperparams_exchange, "ETTm2": hyperparams_ett,
            "Synthetic": hyperparams_synt,
        }

        # Cache results to disk for future runs
        with open("all_eval_dfs.pkl", "wb") as f: pickle.dump(all_eval_dfs, f)
        with open("all_dataset_stats.pkl", "wb") as f: pickle.dump(all_dataset_stats, f)
        with open("all_hyperparams.pkl", "wb") as f: pickle.dump(all_hyperparams, f)
        print("Experiments finished and results cached.")
        
    return all_eval_dfs, all_dataset_stats, all_hyperparams

all_eval_dfs, all_dataset_stats, all_hyperparams = load_or_run_experiments()

# Create the output directory if it doesn't exist
output_dir = "manuscript/table_figures"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------------
# 2. GENERATE TABLE 1: DATASET STATISTICS
# -------------------------------------------------------------------
print("Generating Table 1: Dataset Statistics...")
dataset_table = pd.concat(all_dataset_stats.values(), axis=1)

# Format integers without decimal points, and floats with 4 decimal places
def format_stats(val):
    if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
        return f"{int(val)}"
    return f"{val:.4f}"

table_code = dataset_table.sort_index().style.format(format_stats).to_latex(
    buf=os.path.join(output_dir, "dataset_stats.tex"),
    column_format="@{}lccccc@{}",
    position_float="centering",
    hrules=True,
    caption="Descriptive statistics for each dataset.",
    label="tab:dataset_stats",
    position="t"
)
print(f"Saved dataset statistics table to {os.path.join(output_dir, 'dataset_stats.tex')}")


# -------------------------------------------------------------------
# 3. GENERATE TABLES 2 & 3: MSE & TRAINING TIME
# -------------------------------------------------------------------
print("Generating Tables 2 & 3: MSE and Training Time...")

# Combine all evaluation results into a single DataFrame with a MultiIndex
combined_df = pd.concat(all_eval_dfs, axis=1)
combined_df.columns.names = ['Model', 'Metric']

# Separate the combined DataFrame into MSE and Training Time
df_mse = combined_df.xs('MSE', level='Metric', axis=1)
df_training_time = combined_df.xs('Training Time (s)', level='Metric', axis=1)

# --- MSE Table ---
# Use a temporary DataFrame to handle NaNs correctly during highlighting
df_mse_styled = df_mse.fillna(float('inf'))
df_mse_styled.sort_index().style \
    .highlight_min(axis=0, props='bfseries:') \
    .format(_sci_fmt) \
    .to_latex(
        buf=os.path.join(output_dir, "mses.tex"),
        column_format="@{}lccccc@{}",
        position_float="centering",
        hrules=True,
        caption="Test MSE for each model across datasets. Best results on each dataset are \\textbf{bolded}.",
        label="tab:mses",
        position="t"
    )
print(f"Saved MSE table to {os.path.join(output_dir, 'mses.tex')}")

# --- Training Time Table ---
df_training_time.sort_index().style \
    .format("{:.2f}", na_rep="-") \
    .to_latex(
        buf=os.path.join(output_dir, "times.tex"),
        column_format="@{}lccccc@{}",
        position_float="centering",
        hrules=True,
        caption="Training times (in seconds) for each model across datasets. All experiments are run on a M3 Macbook Pro.",
        label="tab:times",
        position="t!"
    )
print(f"Saved training time table to {os.path.join(output_dir, 'times.tex')}")


# -------------------------------------------------------------------
# 4. GENERATE TABLE 4: HYPERPARAMETERS
# -------------------------------------------------------------------
print("Generating Table 4: Hyperparameters...")

# We use the hyperparameters from one of the runs (e.g., M4) as they are the same
# across all datasets in this setup.
# Drop rows that contain "BSDT-MLP" or "SDT" in the "Model" column as they share the same hyperparameter space with BSDT-Q.
# Drop rows that contain "L1 Regularization", "Batch Size", and "Optimizer" in the "Parameter" column as they are not relevant directly to the hyperparameter search space, as there are very limited space for these parameters.

hyperparams = all_hyperparams["M4"] \
    .drop(index=["BSDT-MLP","SDT"], level=0) \
    .drop(index=["L1 Regularization", "Batch Size","Optimizer"], level = 1) \
    .drop(index="Trees per Layer", level=1) \
    .rename(index={"L2 Regularization": "L1-L2 Regularization"}, level=1) \
    .rename(index={"BSDT-Q": "BSDT Variants"}, level=0)

# Generate a LaTeX string with pandas Styler
latex_str = hyperparams.sort_index().style.to_latex(
    column_format='@{}lll@{}',
    position="t!",
    position_float='centering',
    hrules=True,
    multirow_align='c',
    clines="skip-last;data",
    caption='Hyperparameter search space for each model.',
    label='tab:searchspace',
    convert_css=False,
)

# Apply post-processing fixes for better visual appearance in the paper
# Remove the final cline before the bottomrule
fixed_latex = latex_str.replace('\\cline{1-3}\n\\bottomrule', '\\bottomrule')
fixed_latex = fixed_latex.replace('\\cline{1-3}', '\\midrule')
# Fix the misaligned header created by the multi-index
fixed_latex = fixed_latex.replace(
    ' &  & Search Space \\\\\nModel & Parameter &  ', 
    'Model & Parameter & Search Space'
)

# Save the final LaTeX table to a file
with open(os.path.join(output_dir, "hyperparams.tex"), "w") as f:
    f.write(fixed_latex)
print(f"Saved hyperparameter table to {os.path.join(output_dir, 'hyperparams.tex')}")

print("\nAll tables generated successfully.")





