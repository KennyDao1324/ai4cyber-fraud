"""Data loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainingConfig
from .constants import EXPECTED_BAF_COLUMNS


@dataclass
class TemporalData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


def load_baf_csv(path: str | Path, config: TrainingConfig) -> pd.DataFrame:
    """Load and validate a BAF CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run scripts/download_baf.py or place the CSV in data/raw."
        )
    frame = pd.read_csv(path, low_memory=False)
    unnamed = [column for column in frame.columns if column.lower().startswith("unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)

    required = {config.target_column, config.time_column}
    missing_required = required - set(frame.columns)
    if missing_required:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing_required)}")
    if not set(frame[config.target_column].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{config.target_column} must be binary (0/1)")
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows")

    # Record optional schema differences for diagnostics.
    frame.attrs["missing_expected_columns"] = sorted(EXPECTED_BAF_COLUMNS - set(frame.columns))
    return frame


def _bounded_stratified_sample(
    frame: pd.DataFrame,
    limit: int | None,
    config: TrainingConfig,
    seed_offset: int,
) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    strata = frame[config.time_column].astype(str) + "_" + frame[config.target_column].astype(str)
    try:
        sampled, _ = train_test_split(
            frame,
            train_size=limit,
            random_state=config.random_seed + seed_offset,
            stratify=strata,
        )
    except ValueError:
        sampled = frame.sample(n=limit, random_state=config.random_seed + seed_offset)
    return sampled.sort_index().reset_index(drop=True)


def make_temporal_data(frame: pd.DataFrame, config: TrainingConfig) -> TemporalData:
    """Split records by month and identify model input columns."""
    train = frame[frame[config.time_column].isin(config.train_months)]
    validation = frame[frame[config.time_column].isin(config.validation_months)]
    test = frame[frame[config.time_column].isin(config.test_months)]
    if min(len(train), len(validation), len(test)) == 0:
        counts = frame[config.time_column].value_counts().sort_index().to_dict()
        raise ValueError(f"One temporal split is empty. Observed month counts: {counts}")

    train = _bounded_stratified_sample(train, config.max_train_rows, config, 0)
    validation = _bounded_stratified_sample(validation, config.max_validation_rows, config, 1)
    test = _bounded_stratified_sample(test, config.max_test_rows, config, 2)

    excluded = {
        config.target_column,
        config.time_column,
        *config.protected_features,
        *config.excluded_features,
    }
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No model features remain after exclusions")

    categorical = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric = [column for column in feature_columns if column not in categorical]
    return TemporalData(train, validation, test, feature_columns, numeric, categorical)


def build_preprocessor(data: TemporalData) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, data.numeric_columns),
            ("cat", categorical_pipeline, data.categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def dataset_profile(frame: pd.DataFrame, config: TrainingConfig) -> dict:
    """Calculate dataset summary statistics."""
    target = frame[config.target_column]
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "fraud_count": int(target.sum()),
        "fraud_rate": float(target.mean()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "month_counts": {
            str(key): int(value)
            for key, value in frame[config.time_column].value_counts().sort_index().items()
        },
        "fraud_rate_by_month": {
            str(key): float(value)
            for key, value in frame.groupby(config.time_column)[config.target_column].mean().items()
        },
        "missing_expected_columns": frame.attrs.get("missing_expected_columns", []),
    }
