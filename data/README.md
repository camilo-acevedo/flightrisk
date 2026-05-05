# Data layer

Raw datasets are immutable and DVC-tracked; nothing under `data/` is committed except `.gitkeep` and this file.

## Datasets

### KKBox WSDM Churn Prediction (Tracks A and B)

* Source: Kaggle competition `kkbox-churn-prediction-challenge`.
* Size: ~1M users, ~400M transaction rows, multi-month listening logs.
* License: Released by KKBox under the competition terms; non-commercial research use.
* Files used: `train.csv`, `members_v3.csv`, `transactions.csv`, `user_logs.csv`.

### Orange Belgium Uplift Benchmark (Track C)

* Source: Devriendt et al., 2024. Available via Hugging Face Datasets and the
  authors' supplementary material.
* Size: ~12k customers, 178 features, treatment column from a real RCT.
* License: CC-BY 4.0 per the authors' release.
* Why this dataset: the only public real-telco dataset with a randomized
  treatment column, which is required for honest uplift evaluation.

## Layout

```
data/
├── raw/                immutable downloads, DVC-tracked
│   ├── kkbox/
│   └── orange-belgium/
├── interim/            cleaned, joined, time-aligned
└── features/           Parquet, partitioned by date
```

Both raw datasets are pinned by content hash inside MLflow run manifests. Any
schema change is enforced by pandera at build time.
