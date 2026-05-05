from __future__ import annotations

import click

from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)


@click.group(help="flightrisk command-line interface.")
def main() -> None:
    """Top-level CLI group; subcommands attach below."""


@main.group("data", help="Dataset ingestion and validation.")
def data_group() -> None:
    """Data subcommands."""


@data_group.command("pull", help="Pull raw datasets via DVC.")
def data_pull() -> None:
    """Trigger ``dvc pull`` for the raw data layer.

    Implementation lands in step 1 of the build order.
    """
    _log.info("data pull is not yet implemented; coming in step 1.")


@main.group("features", help="Feature engineering pipelines.")
def features_group() -> None:
    """Feature subcommands."""


@features_group.command("build", help="Build the feature matrix.")
def features_build() -> None:
    """Build the feature matrix.

    Implementation lands in step 2 of the build order.
    """
    _log.info("features build is not yet implemented; coming in step 2.")


@main.group("train", help="Model training tracks.")
def train_group() -> None:
    """Training subcommands."""


@train_group.command("risk")
def train_risk() -> None:
    """Train Track A (risk). Lands in step 3."""
    _log.info("train risk is not yet implemented; coming in step 3.")


@train_group.command("survival")
def train_survival() -> None:
    """Train Track B (survival). Lands in step 4."""
    _log.info("train survival is not yet implemented; coming in step 4.")


@train_group.command("uplift")
def train_uplift() -> None:
    """Train Track C (uplift). Lands in step 5."""
    _log.info("train uplift is not yet implemented; coming in step 5.")


@main.command("simulate", help="Run the campaign ROI simulator.")
def simulate() -> None:
    """Run the campaign ROI simulator. Lands in step 6."""
    _log.info("simulate is not yet implemented; coming in step 6.")


if __name__ == "__main__":
    main()
