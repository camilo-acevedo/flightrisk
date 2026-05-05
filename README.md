# flightrisk

> Identify customers on the verge before they take off.

**flightrisk** builds three honest churn-modeling stacks side by side on real public data, then scores them in dollars instead of AUC.

## The pitch (30 seconds)

Most public churn projects predict `P(churn)`. Retention budgets actually need *who will stay if I intervene minus who would have stayed anyway* — that's uplift, not risk. flightrisk runs **risk**, **survival**, and **uplift** on identical splits, then runs a fixed-budget campaign simulator that turns each model into expected retained revenue with bootstrapped confidence bands. The headline chart compares all three in dollars.

## The technical paragraph

Two real datasets, three tracks, one comparison.

- **Track A — Risk.** LightGBM and XGBoost (focal loss for imbalance) on KKBox WSDM (~1M users, ~400M transactions). Calibrated with isotonic and Platt; reported with AUC, PR-AUC, Brier, ECE, and decile lift.
- **Track B — Survival.** Cox PH (lifelines), DeepHit and DeepSurv (pycox), Random Survival Forest. Same KKBox split — timestamps are the whole point. Reported with time-dependent AUC, integrated Brier, C-index.
- **Track C — Uplift.** T/X/DR-learners on LightGBM, causal forests via econml. Trained on the **Orange Belgium Uplift Benchmark** — the only public real-telco dataset with an actual RCT — because uplift evaluation against non-randomized assignment is theater. Reported with Qini, AUUC, uplift@k%.
- **Business simulator.** Fixed budget *B*, top-k by `P(churn)` (Track A) vs top-k by `τ(x)` (Track C); simulated retained revenue with bootstrapped CIs.

Stack: Python 3.11+ · LightGBM · XGBoost · lifelines · pycox · scikit-survival · econml · causalml · MLflow · Hydra · Pydantic · pandera · pytest + hypothesis · DVC · FastAPI · Streamlit.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

make data
make train
make report
```

## Layout

```
flightrisk/
├── flightrisk/          source package
│   ├── data/            KKBox + Orange loaders, pandera schemas
│   ├── features/        time-aware transforms + tests
│   ├── models/
│   │   ├── risk/        Track A
│   │   ├── survival/    Track B
│   │   └── uplift/      Track C
│   ├── eval/            metrics + business simulator
│   └── serving/         FastAPI + Streamlit
├── configs/             Hydra configs per experiment
├── tests/               pytest + hypothesis
├── data/                gitignored, DVC-tracked
└── reports/             generated, gitignored
```

## License

MIT. Datasets retain their original licenses (see [data/README.md](data/README.md)).
