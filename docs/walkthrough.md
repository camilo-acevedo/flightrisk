# Walkthrough

The end-to-end tour lives in
[`notebooks/00_walkthrough.ipynb`](https://github.com/camilo-acevedo/flightrisk/blob/main/notebooks/00_walkthrough.ipynb).
It generates synthetic data shaped like KKBox and Orange Belgium, builds
features, trains all three tracks, runs the ROI simulator, and shows how to
launch the FastAPI service and Streamlit demo.

## Run it from a fresh clone

```powershell
git clone https://github.com/camilo-acevedo/flightrisk.git
cd flightrisk

py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pip install jupyterlab

jupyter lab notebooks/00_walkthrough.ipynb
```

Each cell is annotated with what it does and which subpackage it exercises so
the notebook doubles as a guided tour of the codebase.
