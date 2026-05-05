# Data layer

Raw datasets are immutable; nothing under `data/` is committed except `.gitkeep` and this file. DVC is the **intended** versioning mechanism — see "DVC bootstrap" below — but the repository today does not ship a populated `.dvc/` because the raw datasets live behind Kaggle credentials. The `flightrisk data pull` CLI tries `dvc pull` first and falls back to the Kaggle CLI when `.dvc/config` is missing or DVC is not installed.

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

## DVC bootstrap (optional)

To put the raw bundles under DVC instead of relying on Kaggle re-downloads:

```powershell
pip install -e ".[data]"
dvc init
dvc remote add -d origin s3://your-bucket/flightrisk    # or gdrive, ssh, etc.
dvc add data/raw/kkbox data/raw/orange-belgium
git add data/raw/kkbox.dvc data/raw/orange-belgium.dvc .dvc
git commit -m "track raw datasets with DVC"
dvc push
```

After this, `flightrisk data pull` will succeed via DVC alone. Until then it
falls back to the Kaggle CLI (`KAGGLE_USERNAME` / `KAGGLE_KEY` required) for
KKBox; Orange Belgium has to be dropped under `data/raw/orange-belgium/` by
hand from the authors' supplementary release.

## Synthetic stand-ins

Want to run the full pipeline without credentials? Use the bundled generators:

```powershell
python scripts\synthetic_kkbox.py  --n-users 30000
python scripts\synthetic_orange.py --n-customers 12000
```

The generated frames satisfy the same pandera schemas as the real bundles, so
every feature transform, model, metric, API and Streamlit panel runs unchanged.
