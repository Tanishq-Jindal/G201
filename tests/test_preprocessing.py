import numpy as np
import pandas as pd
import joblib

from src.mp014.preprocessing import (
    ROOT,
    clean_features,
    normalize_atm_type
)


# Create sample input without using raw CSV files
def sample_input():
    return pd.DataFrame([{
        "nearby_shops": 25,
        "days_to_salary": 5,
        "footfall_index": 70,
        "distance_to_market_km": 1.5,
        "avg_ticket_rs": 2500,
        "other_atms_1km": 3,
        "atm_type": "mall",
        "area": "Market"
    }])


# Test 1: Sentinel value handling
def test_sentinel_value():

    df = sample_input()

    df.loc[0, "avg_ticket_rs"] = -1

    cleaned = clean_features(df)

    assert pd.isna(cleaned.loc[0, "avg_ticket_rs"])


# Test 2: ATM type spelling normalization
def test_spelling_normalization():

    assert normalize_atm_type("ban.") == "bank_branch"
    assert normalize_atm_type("mal.") == "mall"
    assert normalize_atm_type("sta.") == "standalone"

    assert normalize_atm_type(" MALL ") == "mall"
    assert normalize_atm_type("Bank Branch") == "bank_branch"


# Test 3: Champion model prediction
def test_model_prediction():

    model_path = ROOT / "models/model.joblib"

    assert model_path.exists()

    model = joblib.load(model_path)

    df = sample_input()

    cleaned = clean_features(df)

    predictions = model.predict(cleaned)

    assert len(predictions) == 1
    assert np.isfinite(predictions[0])