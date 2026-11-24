from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_atm_type(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()
    value = value.replace(" ", "_")

    mapping = {
        "ban.": "bank_branch",
        "mal.": "mall",
        "sta.": "standalone"
    }

    return mapping.get(value, value)


def clean_features(df):
    config = load_config()
    df = df.copy()

    sentinel_column = config["preprocessing"]["sentinel_column"]
    sentinel_value = config["preprocessing"]["sentinel_value"]

    df[sentinel_column] = df[sentinel_column].replace(
        sentinel_value, np.nan
    )

    df["atm_type"] = df["atm_type"].map(normalize_atm_type)

    features = (
        config["data"]["numerical_features"]
        + config["data"]["categorical_features"]
    )

    return df[features]

def build_preprocessor():
    config = load_config()

    numerical_features = config["data"]["numerical_features"]
    categorical_features = config["data"]["categorical_features"]

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=config["preprocessing"]["numerical_imputation"]
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=config["preprocessing"]["categorical_imputation"]
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor