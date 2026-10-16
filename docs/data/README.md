# Published results

`results.csv` transcribes **Table II** of H. Karaca and S. S. Kozat,
“Soft Gradient Boosting With Learnable Feature Transforms for Sequential
Regression,” *IEEE Signal Processing Letters*, vol. 33, pp. 186–190, 2026.
DOI: https://doi.org/10.1109/LSP.2025.3639345.

Values are test mean squared error (lower is better), at the precision printed
in the published table. These are reported paper results, not measurements
produced during website development. M4 is the quarterly subset. ETTm2 is
sampled every 15 minutes; do not substitute the ETTh2 results in arXiv v1.
The paper labels CatBoost as “Catboost” and BSDT-Q as “BSDT-Q (Ours)”; names
are normalized here for display. Each dataset has its own target scale.

The website's improvement percentages use
`100 * (1 - BSDT-Q MSE / comparison MSE)` and the rounded published values.
“Strongest baseline” means the lowest MSE among the other eight models in
this table, including the SDT and BSDT-MLP ablations.

For the archived code and execution environment, use Code Ocean capsule v1:
https://doi.org/10.24433/CO.9409420.v1.
