from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from flightrisk.utils.logging import get_logger
from flightrisk.utils.paths import get_paths

_log = get_logger(__name__)


def generate(
    n_customers: int = 12_000,
    *,
    n_features: int = 178,
    treated_share: float = 0.5,
    seed: int = 1337,
) -> pd.DataFrame:
    """Generate an Orange Belgium-shaped RCT dataset.

    Half the features are noise, the rest carry signal for the outcome and
    a heterogeneous treatment effect that the uplift track should recover.

    :param n_customers: Number of customers.
    :param n_features: Number of feature columns (defaults to 178 like the
        original benchmark).
    :param treated_share: Probability of being assigned to the treatment arm.
    :param seed: Random seed.
    :returns: A frame with ``f0..f{n-1}``, ``treatment``, ``outcome``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_customers, n_features)).astype(np.float32)
    informative = X[:, :8]

    base_logit = (
        0.6 * informative[:, 0]
        - 0.4 * informative[:, 1]
        + 0.3 * informative[:, 2]
        - 0.2 * informative[:, 3]
        - 1.5
    )
    base_p = 1 / (1 + np.exp(-base_logit))

    tau = np.clip(0.05 + 0.25 * (informative[:, 4] > 0) + 0.10 * (informative[:, 5] > 0.5), 0.0, 0.6)

    treatment = (rng.uniform(size=n_customers) < treated_share).astype(int)
    p = np.clip(base_p + tau * treatment, 0.001, 0.999)
    outcome = (rng.uniform(size=n_customers) < p).astype(int)

    cols = [f"f{i}" for i in range(n_features)]
    frame = pd.DataFrame(X, columns=cols)
    frame["treatment"] = treatment
    frame["outcome"] = outcome
    return frame


def write(frame: pd.DataFrame, *, root: Path | None = None) -> Path:
    """Persist the generated frame as a Parquet file.

    :param frame: Frame from :func:`generate`.
    :param root: Optional override for the project root.
    :returns: The path written.
    """
    base = (root or get_paths().data_raw) / "orange-belgium"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "orange_belgium.parquet"
    frame.to_parquet(path, index=False)
    _log.info("synthetic orange-belgium written to %s", path)
    return path


def main() -> None:
    """Command-line entry point.

    :returns: ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=12_000)
    parser.add_argument("--n-features", type=int, default=178)
    parser.add_argument("--treated-share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    write(
        generate(
            n_customers=args.n_customers,
            n_features=args.n_features,
            treated_share=args.treated_share,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
