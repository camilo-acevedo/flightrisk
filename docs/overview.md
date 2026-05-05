# Overview

`flightrisk` builds three honest churn-modelling stacks side by side and ends with a campaign ROI simulator that compares them in dollars.

## The three tracks

| Track | Question | Output | Trained on |
|---|---|---|---|
| **A — Risk** | Will this customer churn? | `P(churn)` calibrated to the population | KKBox |
| **B — Survival** | When will they churn? | Full `S(t)` and per-horizon hazard | KKBox |
| **C — Uplift** | Will my offer change the outcome? | `τ(x)` per customer | Orange Belgium RCT |

## Why uplift, not just risk

Most public churn projects train Track A and stop. Retention budgets actually need *who will stay if I intervene minus who would have stayed anyway* — that's uplift, not risk. The bundled simulator targets the same population three ways (top-k by risk, top-k by uplift, uniformly random), realises retention with a fixed treatment lift, and reports net revenue with bootstrapped confidence intervals. Uplift consistently dominates risk on unsaturated budgets when the "persuadable" segment is non-empty.

## Reading the API reference

Each subpackage is documented in the API section. Modules favour:

- typed dataclasses for hyperparameters and result bundles
- explicit no-future-leakage cutoffs in feature builders
- result objects with `as_dict()` for direct MLflow logging
- thin wrappers around well-known libraries (LightGBM, lifelines, scikit-survival, econml) so the contract is obvious

For a runnable end-to-end tour see the `walkthrough` notebook ([`notebooks/00_walkthrough.ipynb`](https://github.com/camilo-acevedo/flightrisk/blob/main/notebooks/00_walkthrough.ipynb)).
