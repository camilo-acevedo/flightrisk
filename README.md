# flightrisk

> Identify customers on the verge before they take off.

[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The pitch (30 seconds)

Most public churn projects predict `P(churn)`. Retention budgets actually need *who will stay if I intervene minus who would have stayed anyway* — that's uplift, not risk. **flightrisk** runs **risk**, **survival**, and **uplift** on identical splits, then runs a fixed-budget campaign simulator that turns each model into expected retained revenue with bootstrapped confidence bands. The headline chart compares all three policies in dollars.

## The technical paragraph

Two real public datasets, three modelling tracks, one comparison.

- **Track A — Risk.** LightGBM and XGBoost (focal-loss ready) on the **KKBox WSDM** subscription dataset (~1M users, ~400M transaction rows, daily listening logs). Calibrated with isotonic and Platt; reported with AUC, PR-AUC, Brier, ECE, and decile lift.
- **Track B — Survival.** Cox PH (lifelines) and Random Survival Forest (scikit-survival) on KKBox — the timestamps are the whole point. Reported with Harrell C-index, time-dependent AUC averaged across horizons, and integrated Brier score.
- **Track C — Uplift.** T-, X-, DR-learners on LightGBM and a CausalForestDML from econml, trained on the **Orange Belgium Uplift Benchmark** — the only public real-telco dataset with an actual RCT, because uplift evaluation against non-randomised assignment is theatre. Reported with Qini coefficient (N²-normalised so random rankings sit near zero), AUUC, and uplift@k%.
- **Business simulator.** Fixed budget *B*, top-k by `P(churn)` (Track A) vs top-k by `τ(x)` (Track C) vs uniformly random; simulated retained revenue net of treatment cost with bootstrapped CIs.

## Stack

Python 3.11 · LightGBM · XGBoost · lifelines · scikit-survival · econml · MLflow · Pydantic + pandera · pytest + hypothesis · DVC + Kaggle ingestion · FastAPI · Streamlit · matplotlib · joblib · click

## Quickstart

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"

# 1. Pull raw datasets via DVC (or Kaggle fallback for KKBox)
flightrisk data pull
flightrisk data validate

# 2. Build feature matrices for both datasets
flightrisk features build --cutoff 2017-02-28

# 3. Train each track end-to-end (artifacts land in MLflow + reports/)
flightrisk train risk --estimator lightgbm --calibration isotonic
flightrisk train survival --estimator rsf --horizon-days 90
flightrisk train uplift --estimator t_learner

# 4. Headline ROI chart with bootstrapped CIs
flightrisk simulate --budgets 100000,250000,500000

# 5. Serve and explore
.\.venv\Scripts\python -m uvicorn flightrisk.serving.api:app --port 8000
.\.venv\Scripts\python -m streamlit run flightrisk/serving/streamlit_app.py
```

## Repo layout

```
flightrisk/
├── flightrisk/                     source package
│   ├── data/                       loaders, pandera schemas, DVC + Kaggle ingestion
│   ├── features/                   time-aware KKBox + Orange transforms
│   ├── models/
│   │   ├── risk/                   Track A (LightGBM, XGBoost, calibration)
│   │   ├── survival/               Track B (Cox, Random Survival Forest)
│   │   └── uplift/                 Track C (T/X/DR-learners, causal forest)
│   ├── eval/                       metrics for each track + ROI simulator
│   ├── serving/                    FastAPI + Streamlit
│   ├── utils/                      paths, logging, seeding, MLflow helpers
│   ├── cli.py                      `flightrisk` click entry point
│   └── config.py                   pydantic-settings runtime config
├── configs/                        per-experiment YAML tree
├── tests/                          pytest + fixtures (76 tests)
├── data/                           gitignored, DVC-tracked
└── reports/                        generated, gitignored
```

## CLI surface

```
flightrisk
├── data
│   ├── pull           dvc pull (kaggle fallback for KKBox)
│   └── validate       load + pandera-validate raw data
├── features
│   └── build          KKBox + Orange feature bundles → data/features/
├── train
│   ├── risk           Track A → reports/risk/<run_id>/
│   ├── survival       Track B → reports/survival/<run_id>/
│   └── uplift         Track C → reports/uplift/<run_id>/
└── simulate           ROI chart + comparison CSV → reports/simulator/
```

## Datasets

| Dataset | Role | Why it's here |
|---|---|---|
| KKBox WSDM Churn Prediction | Tracks A and B | Real subscription data with timestamps, cancellations, refunds, daily usage logs. Big enough to stress every modelling choice. |
| Orange Belgium Uplift Benchmark | Track C only | The only public real-telco dataset with an actual RCT. Without it, uplift evaluation is theatre. |

Both raw datasets are content-hashed and pinned in MLflow run manifests. License terms documented in [data/README.md](data/README.md).

## Evaluation philosophy

| Track | Split | Headline metric | Honest baselines |
|---|---|---|---|
| Risk | Strict temporal cutoff (no shuffling) | AUC, PR-AUC, Brier, ECE, decile lift | reliability diagram + decile lift |
| Survival | Random within KKBox feature build | C-index, time-dependent AUC, integrated Brier | per-horizon S(t) sample plot |
| Uplift | Stratified RCT folds preserving treatment × outcome | Qini, AUUC, uplift@10/20/30 | random ranking ≈ 0 |

## Definition of done — checklist

- [x] `make report` reproduces every figure end-to-end from the two public datasets.
- [x] One headline chart: simulated retained revenue across the three tracks at three budget levels with CIs.
- [x] All three tracks trainable end-to-end and registered in MLflow.
- [x] FastAPI service with one swappable endpoint per track.
- [x] Streamlit demo with SHAP for risk, S(t) for survival, decile bar for uplift.
- [x] README paragraph that a non-ML PM gets in 30 seconds, plus a technical paragraph that a senior data scientist respects.

## License

MIT. Datasets retain their original licenses (see [data/README.md](data/README.md)).
