# configs/

These YAML files are **reference defaults**, not Hydra-driven runtime configs.

## How they are consumed

The `flightrisk` click CLI exposes the same parameters as command-line options:

```powershell
flightrisk train risk     --estimator lightgbm --calibration isotonic
flightrisk train survival --estimator rsf      --horizon-days 90
flightrisk train uplift   --estimator t_learner
```

The values that ship under [`configs/model/*.yaml`](model/) are the same defaults the CLI uses. Override them via CLI flags or write a thin loader if you need full file-driven configs:

```python
import yaml
with open("configs/model/risk_lgbm.yaml") as fh:
    cfg = yaml.safe_load(fh)
```

## Layout

| File | Purpose |
|---|---|
| [`config.yaml`](config.yaml) | Top-level paths, MLflow experiment, and pointers to per-tier configs. |
| [`data/kkbox.yaml`](data/kkbox.yaml) | KKBox source, files, horizon, cutoff. |
| [`data/orange_belgium.yaml`](data/orange_belgium.yaml) | Orange Belgium RCT metadata. |
| [`features/default.yaml`](features/default.yaml) | Feature pipeline switches (rolling windows, payment dynamics). |
| [`model/risk_lgbm.yaml`](model/risk_lgbm.yaml) | LightGBM hyperparameters for Track A. |
| [`model/survival_cox.yaml`](model/survival_cox.yaml) | Cox PH defaults for Track B. |
| [`model/uplift_tlearner.yaml`](model/uplift_tlearner.yaml) | Defaults for the T-learner on Track C. |
| [`eval/default.yaml`](eval/default.yaml) | Metric switches and simulator parameters. |

## Why this is not Hydra

The earlier scaffold included `defaults:` and `hydra:` sections in `config.yaml`, but no code path called `@hydra.main` or `compose()`. Honest reality: the CLI is click-driven and these YAMLs are documentation. Wiring Hydra is an option for the next iteration if multi-run sweeps become necessary.
