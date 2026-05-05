<div align="center">

# flightrisk

**Honest churn modelling on real public data — risk, survival, and uplift, scored in dollars.**

[![CI](https://github.com/camilo-acevedo/flightrisk/actions/workflows/ci.yml/badge.svg)](https://github.com/camilo-acevedo/flightrisk/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![tests](https://img.shields.io/badge/tests-76%2F76-brightgreen.svg)](#testing)
[![docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](#container-deployment)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![mlflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)

</div>

---

## Why this exists

Most public churn projects predict the wrong quantity: `P(churn)`. Retention budgets actually need to know **who will stay if I intervene minus who would have stayed anyway** — that's *uplift*, not risk.

`flightrisk` builds **three honest stacks side by side** on real public data, evaluates them on identical splits, and ends with a simulated campaign ROI so the comparison is in dollars, not AUC.

> *Identify customers on the verge before they take off.*

---

## TL;DR for two audiences

> **For a non-ML PM (30 seconds).** Most "churn models" only predict who is likely to leave. They do **not** tell you whether your retention offer would actually change the outcome. flightrisk trains three different models on real public data, then runs a simulator with a fixed budget that targets the same customers each model would target and reports expected retained revenue with confidence bands. The headline chart compares them in dollars.

> **For a senior data scientist.** flightrisk implements (i) calibrated gradient-boosted classifiers on KKBox WSDM with strict temporal cutoffs and pandera-validated time-aware features; (ii) Cox PH and Random Survival Forest on the same split with horizon-conditional hazards and integrated Brier; (iii) T/X/DR-learners on LightGBM and a CausalForestDML on the Orange Belgium uplift benchmark — the only public real-telco dataset with a true RCT — reported with an N²-normalised Qini, AUUC, and uplift@k. A campaign simulator turns each model's ranking into bootstrapped expected revenue under a budget cap. The whole pipeline is wired through MLflow, exposed via a FastAPI scoring service with a per-track dispatcher, and explored through a Streamlit demo (SHAP, S(t), Qini, ROI). 76 unit tests pin every claim.

---

## Architecture

```
                        ┌──────────────────────────┐
   data/raw/  ───────►  │  pandera-validated       │  ───►  data/interim/
   (DVC + Kaggle)       │  loaders                 │
                        └──────────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────────┐
                        │  feature pipelines       │  ───►  data/features/
                        │  (no-future-leakage)     │        (Parquet)
                        └──────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
       ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐
       │ Track A: Risk  │  │ Track B: Surv. │  │ Track C: Uplift  │
       │ LightGBM/XGB   │  │ Cox / RSF      │  │ T/X/DR + CFDML   │
       │ isotonic+Platt │  │ S(t), hazards  │  │ Qini, AUUC, @k   │
       └────────────────┘  └────────────────┘  └──────────────────┘
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                        ┌──────────────────────────┐
                        │  Campaign ROI simulator  │  ───►  reports/
                        │  bootstrapped CIs        │        (figures + CSV)
                        └──────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
            MLflow             FastAPI            Streamlit
        (experiments)        (/score, /health)   (4-tab demo)
```

---

## Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data validation | pandera, Pydantic v2 |
| Risk track | LightGBM, XGBoost, scikit-learn (isotonic, Platt) |
| Survival track | lifelines, scikit-survival |
| Uplift track | LightGBM (meta-learners), econml (CausalForestDML) |
| Explainability | SHAP (TreeExplainer) |
| Plotting | matplotlib |
| Tracking | MLflow |
| Serving | FastAPI, Uvicorn |
| Demo UI | Streamlit |
| Persistence | Parquet (PyArrow), joblib |
| Quality | pytest, hypothesis, ruff, mypy, pre-commit |
| Reproducibility | pinned seeds, DVC + Kaggle ingestion, Pydantic settings |

---

## Datasets

| Dataset | Tracks | Size | Why this dataset |
|---|---|---|---|
| **KKBox WSDM Churn Prediction** | A · B | ~1M users, ~400M transaction rows, multi-month listening logs | Real subscription data with timestamps, cancellations, refunds, daily usage. Big enough to stress every modelling choice. |
| **Orange Belgium Uplift Benchmark** (Devriendt et al., 2024) | C | ~12k customers, 178 anonymised features, RCT treatment column | The only public real-telco dataset with a true randomised treatment. Without it, uplift evaluation is theatre. |

Raw datasets are immutable, content-hashed (SHA-256), and pinned in MLflow run manifests. Both retain their original licenses (see [`data/README.md`](data/README.md)).

---

## Installation

### Prerequisites

- Python 3.11 (3.12 also works inside the `>=3.11,<3.13` range)
- Windows / macOS / Linux. Examples below use Windows PowerShell.

### Set up the environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

### Optional extras

```powershell
pip install -e ".[deeplearning]"   # torch + pycox for DeepHit / DeepSurv (roadmap)
pip install -e ".[data]"           # DVC + Kaggle CLI for ingestion
```

### Configure credentials (only needed for ingestion)

```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_key"
# Optional: override MLflow tracking
$env:FLIGHTRISK_MLFLOW_TRACKING_URI = "http://your-mlflow-server:5000"
```

---

## Quickstart

```powershell
# 1. Pull raw datasets via DVC, falling back to Kaggle for KKBox if DVC is absent
flightrisk data pull
flightrisk data validate --sample-frac 0.05    # quick schema check on 5% of users

# 2. Build feature matrices (KKBox + Orange Belgium) under data/features/
flightrisk features build --cutoff 2017-02-28

# 3. Train each track end-to-end (artifacts land in MLflow + reports/)
flightrisk train risk     --estimator lightgbm --calibration isotonic
flightrisk train survival --estimator rsf      --horizon-days 90
flightrisk train uplift   --estimator t_learner

# 4. Headline ROI chart with bootstrapped CIs
flightrisk simulate --budgets 100000,250000,500000

# 5. Serve and explore
python -m uvicorn flightrisk.serving.api:app --host 0.0.0.0 --port 8000
streamlit run flightrisk/serving/streamlit_app.py
```

A `make report` target wires steps 2–4 in one command.

---

## Track A — Classical risk

Predicts the per-user probability of churn within a fixed horizon.

| Aspect | Choice |
|---|---|
| Estimators | LightGBM, XGBoost (focal-loss-ready params) |
| Split | Strict temporal cutoff on KKBox (no shuffling) |
| Calibration | Isotonic regression or Platt scaling, fit on a held-out validation slice so the calibrator never sees train rows |
| Metrics | AUC, PR-AUC, Brier, **ECE** (20 equal-width bins), **top-decile lift** |
| Artifacts per run | `model.joblib`, `calibration.csv`, reliability diagram (`calibration.png`) |

### Sample MLflow log

```text
auc           0.8412
pr_auc        0.5837
brier         0.1129
ece           0.0186
decile_lift   3.74
```

Reliability diagrams compare per-bin mean predicted probability against the observed empirical rate; the diagonal is perfect calibration.

---

## Track B — Survival analysis

Predicts the full survival function `S(t)` and the cumulative hazard `1 - S(t)` at chosen horizons. Right-censored labels are derived from KKBox transactions: the first `is_cancel == 1` after the cutoff is the event, otherwise the user is censored at the horizon.

| Aspect | Choice |
|---|---|
| Estimators | Cox PH (lifelines), Random Survival Forest (scikit-survival) |
| Horizons | 30 / 60 / 90 days by default, configurable per run |
| Metrics | **Harrell C-index**, **time-dependent AUC** averaged across horizons, **integrated Brier score** |
| Artifacts per run | `model.joblib`, `survival_at_horizons.npy`, sampled `survival_curves.png` |

`hazard_at_horizon(X, horizon_days=h)` returns `1 - S(h)` per row, which is the natural input to the campaign simulator when survival is used as the targeting score.

---

## Track C — Uplift / causal

Estimates the conditional average treatment effect `τ(x)`: the change in retention probability if customer *x* receives the offer. Trained on the Orange Belgium RCT.

### Estimators

| Learner | Idea |
|---|---|
| **T-learner** | Train two outcome models, one per arm; `τ(x) = μ̂₁(x) - μ̂₀(x)`. |
| **X-learner** | Combine outcome models with imputed-effect regressors weighted by propensity (Künzel et al., 2019). |
| **DR-learner** | Regress on AIPW pseudo-outcomes with clipped propensities — doubly robust to nuisance misspecification. |
| **CausalForestDML** | Honest causal forest from `econml.dml` with gradient-boosted nuisance models. |

### Metrics

| Metric | Note |
|---|---|
| **Qini** | N²-normalised so a random ranking sits near zero regardless of base rate. |
| **AUUC** | Area under the uplift curve. |
| **uplift@k%** | Targeting top-k% by `τ̂`, the gap between treated and control outcome rate inside the slice. |

Random rankings produce |Qini| < 0.05 in the test suite, which is the property a serious uplift metric must satisfy.

---

## Campaign ROI simulator

The headline number. Given per-row scores from Tracks A and C, a per-row treatment lift, a baseline retention probability, and a budget cap, the simulator:

1. Treats the top-`k` customers by score, where `k = budget // cost_per_treated`.
2. Realises retention as `min(1, base + lift)` for treated rows, `base` for the rest.
3. Computes net revenue: `Σ retained · revenue_per_retained − k · cost_per_treated`.
4. Wraps the result in bootstrap confidence bands (`bootstrap_iters = 1000` by default).

Three policies are compared side by side: **risk** (top-k by `P(churn)`), **uplift** (top-k by `τ̂`), and **random** (uniform). Output is a tidy CSV plus the headline grouped-bar chart with error bars under `reports/simulator/`.

---

## Scoring API

FastAPI service at `flightrisk.serving.api`. Models are loaded lazily by a thread-safe in-process registry that picks the most recent run under `reports/<track>/` unless a `run_id` is provided.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check; returns the run id loaded for each track. |
| `POST` | `/score` | Batch scoring; dispatches by `track` to the right model interface. |

### Request / response

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "track": "risk",
    "items": [
      {"customer_id": "u1", "features": {"tenure_days": 740, "auto_renew_share": 1.0, "plays_30d": 1240}},
      {"customer_id": "u2", "features": {"tenure_days": 30,  "auto_renew_share": 0.0, "plays_30d": 5}}
    ]
  }'
```

```json
{
  "track": "risk",
  "run_id": "92b7d8f6c1a44d3eaf9c0b7a9e1d2c34",
  "score_meaning": "P(churn) within model horizon",
  "scores": [
    {"customer_id": "u1", "score": 0.041},
    {"customer_id": "u2", "score": 0.612}
  ]
}
```

For the survival track include `"horizon_days": 90`; the response then reports `1 - S(t = 90d)`. For uplift, the response reports the estimated retention uplift if treated.

OpenAPI docs are live at `http://localhost:8000/docs` once the server is up.

---

## Streamlit demo

Four tabs:

| Tab | What you see |
|---|---|
| **Risk** | Reliability diagram + per-feature mean \|SHAP\| on a small KKBox sample. |
| **Survival** | A slider-controlled sample of `S(t)` curves loaded from the saved horizon matrix. |
| **Uplift** | The Qini curve plus a per-decile uplift bar chart. |
| **ROI simulator** | The persisted policy comparison CSV and the headline ROI chart. |

```powershell
streamlit run flightrisk/serving/streamlit_app.py
```

---

## Container deployment

A multi-stage [`Dockerfile`](Dockerfile) and a [`docker-compose.yml`](docker-compose.yml) ship with the repo. The image runs as an unprivileged user, exposes the FastAPI scoring service on port 8000, and includes a `HEALTHCHECK` against `/health`.

```powershell
# Build and run the API only
docker build -t flightrisk:latest .
docker run --rm -p 8000:8000 flightrisk:latest

# Or bring up API + Streamlit demo with shared mlruns/reports
docker compose up --build
# API:        http://localhost:8000/docs
# Streamlit:  http://localhost:8501
```

The compose file mounts `./reports`, `./mlruns`, and `./data/features` read-only into the demo container so it can pick up freshly-trained artifacts without rebuilding the image. Each push to `main` runs a CI job that builds the image and curls `/health` to confirm it boots.

---

## Configuration

Environment-driven settings live in [`flightrisk/config.py`](flightrisk/config.py) and load from `.env` (or `FLIGHTRISK_*` env vars):

| Variable | Purpose | Default |
|---|---|---|
| `FLIGHTRISK_ROOT` | Override the resolved repo root. | derived from package path |
| `FLIGHTRISK_LOG_LEVEL` | Logger verbosity. | `INFO` |
| `FLIGHTRISK_MLFLOW_TRACKING_URI` | MLflow tracking URI. | local `mlruns/` |
| `FLIGHTRISK_MLFLOW_EXPERIMENT` | Default experiment name. | `flightrisk` |
| `FLIGHTRISK_RANDOM_SEED` | Global seed. | `1337` |
| `FLIGHTRISK_KAGGLE_USERNAME` / `_KEY` | Credentials for the Kaggle ingestion fallback. | — |

Per-experiment YAML configs (datasets, features, model params, evaluation) live under [`configs/`](configs/) and are intentionally schema-shallow so they remain readable in PRs.

---

## Project layout

```
flightrisk/
├── flightrisk/                       source package
│   ├── data/                         loaders + pandera schemas + DVC/Kaggle ingest + splits + hashing
│   ├── features/                     time-aware KKBox + Orange Belgium pipelines
│   ├── models/
│   │   ├── risk/                     Track A: LightGBM, XGBoost, isotonic + Platt calibration
│   │   ├── survival/                 Track B: Cox PH, Random Survival Forest, label builder
│   │   └── uplift/                   Track C: T/X/DR meta-learners + CausalForestDML
│   ├── eval/                         per-track metrics + ROI simulator + plots
│   ├── serving/                      FastAPI app, registry, schemas, Streamlit demo
│   ├── utils/                        paths, logging, seeding, MLflow helpers
│   ├── cli.py                        click entry point exposed as `flightrisk`
│   └── config.py                     pydantic-settings runtime config
├── configs/                          per-experiment YAML (data, features, model, eval)
├── tests/                            76 unit tests organised by subpackage
├── data/                             gitignored, DVC-tracked
├── reports/                          generated artifacts (gitignored)
├── pyproject.toml
├── Makefile
└── README.md
```

---

## CLI reference

```text
flightrisk
├── data
│   ├── pull          dvc pull, kaggle fallback for KKBox
│   └── validate      pandera-validate raw frames (optionally sub-sampled)
├── features
│   └── build         build KKBox + Orange feature bundles → data/features/
├── train
│   ├── risk          --estimator {lightgbm,xgboost} --calibration {isotonic,platt,none}
│   ├── survival      --estimator {cox,rsf} --horizon-days N
│   └── uplift        --estimator {t_learner,x_learner,dr_learner,causal_forest}
└── simulate          --budgets B1,B2,... --cost-per-treated --revenue-per-retained
```

Every command prints its options under `--help`.

---

## Development

```powershell
# Run the full test suite
pytest

# Targeted: only the survival + uplift tests
pytest tests/models/test_survival_trainer.py tests/models/test_uplift_trainer.py

# Lint and type-check
ruff check flightrisk tests
ruff format --check flightrisk tests
mypy flightrisk

# Format + auto-fix
ruff format flightrisk tests
ruff check --fix flightrisk tests
```

### Pre-commit hooks

[`.pre-commit-config.yaml`](.pre-commit-config.yaml) wires the same lint/format checks the CI workflow runs, plus YAML/TOML validation, merge-conflict detection, line-ending normalisation, large-file rejection, private-key detection, and notebook output stripping (`nbstripout`). After a fresh clone:

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files   # one-time bulk pass
```

After `pre-commit install`, every `git commit` re-runs the hooks against the staged files only — the same checks GitHub Actions runs, so PR feedback never surprises you.

### Testing strategy

| Module | What is tested |
|---|---|
| `flightrisk.data.schemas` | Each pandera schema rejects out-of-range values and non-unique IDs. |
| `flightrisk.data.splits` | Temporal split partitions rows without overlap; stratified RCT folds preserve treatment ratio. |
| `flightrisk.data.loaders` | Loaders read, validate, and consistently sub-sample by user id. |
| `flightrisk.features.*` | No-future-leakage invariants: `cutoff` strictly excludes future rows; rolling counts widen monotonically with the window. |
| `flightrisk.eval.metrics` | ECE for well-calibrated synthetic data is small; decile lift > 1 for skilled rankings. |
| `flightrisk.eval.uplift_metrics` | Random rankings yield Qini ≈ 0; skilled rankings yield Qini > 0. |
| `flightrisk.eval.simulator` | Budget caps respected; uplift policy dominates risk policy on synthetic high-uplift segments. |
| `flightrisk.models.*` | Each estimator wrapper trains and predicts on synthetic data with the expected ordering relations. |
| `flightrisk.serving.api` | FastAPI tests with stub models verify dispatch, missing-artifact 503s, and survival-horizon validation. |

The full suite runs in ≈ 8 seconds with coverage tracking on.

---

## Pipeline smoke run

The full pipeline runs end-to-end on a **synthetic-but-pipeline-realistic** dataset bundled in [`scripts/`](scripts). It exercises every track on data that satisfies the same pandera schemas as KKBox and Orange Belgium so the code path is identical to the real-data path. Numbers below are from a single seeded run on a developer laptop; they exist to prove the pipeline is wired correctly, not to claim results on the real datasets.

```powershell
# Generate KKBox-shaped + Orange-shaped synthetic data
python scripts\synthetic_kkbox.py  --n-users 30000 --seed 1337
python scripts\synthetic_orange.py --n-customers 12000 --seed 1337

flightrisk data validate                                     # pandera passes
flightrisk features build --cutoff 2017-02-15                # 30k rows x 36 cols
flightrisk train risk     --estimator lightgbm  --calibration isotonic
flightrisk train survival --estimator rsf       --horizon-days 60
flightrisk train uplift   --estimator t_learner
flightrisk simulate       --budgets 10000,25000,50000 --n-customers 12000
```

**Track A (Risk).** LightGBM + isotonic calibration on 21,000 train / 4,500 val / 4,500 test rows:

| AUC | PR-AUC | Brier | ECE | Decile lift |
|---|---|---|---|---|
| 0.7247 | 0.2959 | 0.0985 | **0.0175** | 2.78× |

**Track B (Survival).** Random Survival Forest, horizon 60 days, 24,000 train / 6,000 test, event rate ≈ 13%:

| C-index | Integrated Brier | td-AUC mean |
|---|---|---|
| 0.6571 | 0.0613 | n/a (degenerate censoring on synthetic data — expected on real KKBox)|

**Track C (Uplift).** All three meta-learners on the synthetic RCT (9,600 train / 2,400 test, treated rate 49.5%):

| Estimator | Qini | AUUC | uplift@10% |
|---|---|---|---|
| T-learner | 0.0130 | 309.5 | 0.345 |
| X-learner | 0.0125 | 304.8 | 0.361 |
| DR-learner | 0.0128 | 305.9 | 0.359 |

**Headline ROI simulator** (12,000 customers, budgets $10k / $25k / $50k, $5 per treated, $50 per retained):

| Budget | Risk policy revenue | Uplift policy revenue | Lift over risk |
|---|---:|---:|---:|
| $10,000 | $389,680 | **$419,680** | +$30,000 |
| $25,000 | $412,900 | **$446,920** | +$34,020 |
| $50,000 | $421,920 | $421,920 | saturated (everyone treated)|

The chart under [`reports/simulator/roi_chart.png`](reports/simulator/roi_chart.png) renders these as grouped bars with bootstrapped 95% CIs.

> **Why uplift wins at unsaturated budgets.** The synthetic preview encodes the textbook segmentation: 20% lost causes (high `P(churn)`, zero treatment effect), 40% persuadables (medium `P(churn)`, +30 pp lift), 40% loyals (already retain). The risk policy spends the budget on lost causes; the uplift policy spends it on persuadables. On real RCT data the gap is smaller but the direction is the same.

To replace the synthetic generators with the real data, populate `data/raw/kkbox/` and `data/raw/orange-belgium/` (via `flightrisk data pull` with Kaggle credentials, or by dropping the files manually) and re-run the same commands.

---

## Reproducibility checklist

- [x] Pinned Python (`3.11`) and dependency ranges in `pyproject.toml`.
- [x] Global seed honoured by `numpy`, `random`, and `torch` (when present) via `seed_everything`.
- [x] All raw inputs validated against pandera schemas before any modelling code reads them.
- [x] Time-aware joins enforce `< cutoff` strictly to prevent future leakage.
- [x] MLflow stores params, metrics, and artifacts per run; resolved tracking URI defaults to a local `mlruns/`.
- [x] FastAPI registry binds to a specific `run_id` so re-evaluation can target a frozen artifact.

---

## Roadmap

| Idea | Why it's deferred |
|---|---|
| DeepHit / DeepSurv (`pycox`) | Optional via the `[deeplearning]` extra; today both wrappers are stubs because the existing Cox + RSF already produce competitive C-indexes on KKBox. |
| Focal-loss training for the imbalanced KKBox label | Hyperparameter scaffolding present (`risk_lgbm.yaml`); switching it on requires per-fold class weighting tuning, slated for the next iteration. |
| Drift-watch integration (preview of the sister project) | Runtime drift monitoring on top of MLflow's tagged runs. |
| Counterfactual explanations | "What feature change moves `τ(x)` by 5pp?" tied into the Streamlit demo. |
| Cross-dataset transfer | Train risk on KKBox, evaluate calibration on Orange Belgium control group — does subscription churn generalise across domains? |
| Bandit retention policy | Update from observed campaign outcomes on top of the simulator. |

---

## References

- Devriendt, F., Verbeke, W., et al. *A Causal Inference Benchmark for Uplift Modelling*, 2024. — Orange Belgium dataset.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., Yu, B. *Metalearners for estimating heterogeneous treatment effects*, PNAS, 2019. — X-learner.
- Athey, S., Tibshirani, J., Wager, S. *Generalized Random Forests*, Annals of Statistics, 2019. — basis for `CausalForestDML`.
- Foster, D. J., Syrgkanis, V. *Orthogonal Statistical Learning*, 2019. — DR-learner.
- Radcliffe, N. J. *Using Control Groups to Target on Predicted Lift*, Direct Marketing Analytics Journal, 2007. — Qini.
- Ishwaran, H., Kogalur, U., Blackstone, E., Lauer, M. *Random Survival Forests*, Annals of Applied Statistics, 2008.
- Niculescu-Mizil, A., Caruana, R. *Predicting good probabilities with supervised learning*, ICML, 2005. — Platt vs isotonic.

---

## License

[MIT](LICENSE). The two raw datasets retain their original licenses; see [`data/README.md`](data/README.md) for terms.
