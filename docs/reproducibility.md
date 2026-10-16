# Reproducing BSDT-Q

## Reference releases

- **Article:** H. Karaca and S. S. Kozat, *IEEE Signal Processing Letters*, vol. 33, pp. 186–190, 2026. DOI: [10.1109/LSP.2025.3639345](https://doi.org/10.1109/LSP.2025.3639345).
- **Archived software:** Code Ocean capsule **v1**, published 2025. DOI: [10.24433/CO.9409420.v1](https://doi.org/10.24433/CO.9409420.v1). [Open capsule](https://codeocean.com/capsule/9934558/tree/v1).
- **Earlier preprint:** [arXiv:2509.12920v1](https://arxiv.org/abs/2509.12920v1). Its evaluation differs from the published article. In particular, it uses ETTh2; the article and `clean_ett.py` use ETTm2.

The capsule identity, authors, title, version, and destination were verified through [DataCite](https://api.datacite.org/dois/10.24433/co.9409420.v1). The capsule interface returned HTTP 403 to automated access in this environment, so its file tree, dependency versions, execution instructions, and successful reproduction were not independently checked. Follow the capsule's own instructions when opening it interactively. A DOI identifies an archived release; it does not by itself establish that every output has been independently reproduced.

## What this checkout does

`BoostedSDTR` in `bsdtq.py` constructs K transform–tree blocks. Each block receives the original input through its own transform. The forward pass sums their predictions and regularizers. `_CustomLoss` adds prediction MSE to tree-routing and parameter penalties, and skorch trains the ensemble with a shared loss.

The article's Algorithm 1 instead describes explicit stagewise residual fitting and updates of Q. There is no separate residual-fitting loop in this class, and the linear layers use PyTorch's default initialization rather than explicit identity initialization. The website's diagram represents the sequential training procedure in the article's Algorithm 1, rather than the jointly optimized additive implementation in this checkout. Avoid treating the code as a literal transcription of every step in Algorithm 1.

Supported transform types are `identity`, `linear` (a square bias-free map), and `mlp`. The configurations select a linear transform for BSDT-Q, an MLP for BSDT-MLP, and an identity transform with one layer and one tree for SDT.

## Running locally

See the root README for environment creation and runner commands. `requirements.txt` is an unpinned convenience list derived from imports, not an archived or tested lockfile. Record Python, PyTorch, and dependency versions, the commit, seed, and any configuration edits for new experiments.

| Runner | Dataset / setting in this checkout |
| :--- | :--- |
| `clean_synt.py` | 5 synthetic series; 50 input features, 10 informative; ARMA noise; 200 training points and horizon 12 per series |
| `clean_m4.py` | M4 Quarterly; 5 selected series; seed 123; horizon from `M4Info` |
| `clean_weather.py` | Weather; target `OT`; 10-minute frequency; horizon 12 |
| `clean_exchange.py` | Exchange; inspect the runner for series selection and horizon |
| `clean_ett.py` | ETTm2; target `OT`; 15-minute frequency; horizon 48 |

Real-data runners contain the author-specific cache directory `/Users/huseyinkaraca/Documents/datasets`. Change it to an accessible directory before running them. `datasetsforecast` may download datasets on the first run. Run from the repository root so local imports and relative output paths resolve.

Dataset runners use Optuna through AutoMLForecast and return evaluation metrics, dataset statistics, and hyperparameter search-space metadata. Reported `Training Time (s)` is the duration of the selected best trial in the runner, not the total search or end-to-end wall-clock time.

`clean_tables.py` runs all five experiments when caches are absent and writes:

- `all_eval_dfs.pkl`, `all_dataset_stats.pkl`, `all_hyperparams.pkl` in the current directory;
- `dataset_stats.tex`, `mses.tex`, `times.tex`, `hyperparams.tex` under `manuscript/table_figures/`.

It checks only the evaluation cache before trying to load all three files. Keep the three caches together; move all three aside when rerunning changed settings. The module executes the workflow on import.

## Methodological details to preserve and report

The supplied published article and this checkout do not agree on every detail. For example, the synthetic ARMA moving-average coefficient is 0.75 in the article and 0.95 in `clean_synt.py`. Do not silently change these values when comparing results; establish which configuration the archived release uses.

The M4, Weather, and synthetic runners standardize targets using the full series before splitting. If designing a new leakage-controlled evaluation, fit scaling only on the training partition and document that protocol change; its results need not match the article. The time validation helper splits by row order and is not a per-series temporal splitter. Inspect the resulting partition when using multiple series.

The published results and training times are evidence for the reported benchmark configurations, not a guarantee of performance on new datasets or hardware. This documentation update did not rerun model training or independently reproduce Table II.
