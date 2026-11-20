
from pathlib import Path
import re
import string
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

TRAIN_FILE = DATA_DIR / "train.csv"
LATER_FILE = DATA_DIR / "later.csv"
HOLDOUT_FILE = DATA_DIR / "holdout_features.csv"

TARGET = "withdrawal_thousand_rs"
GROUP = "area"
DATE = "date"

SENTINEL_VALUES = {-1, -999, 999, 9999}


def load_data():
    """Load the original project datasets."""
    train = pd.read_csv(TRAIN_FILE)
    later = pd.read_csv(LATER_FILE)
    holdout = pd.read_csv(HOLDOUT_FILE)

    return train, later, holdout


def check_leaking_column(train, holdout):
    """Find columns missing from holdout and check target correlation."""

    print("\n========== 1. LEAKING COLUMN ==========")

    missing = sorted(
        (set(train.columns) - set(holdout.columns)) - {TARGET}
    )

    print("Columns missing from holdout:", missing)

    for column in missing:
        if pd.api.types.is_numeric_dtype(train[column]):
            correlation = train[column].corr(train[TARGET])
            print(
                f"{column}: correlation with target = "
                f"{correlation:.6f}"
            )


def check_unit_mixup(train):
    """Compare numerical statistics across area groups."""

    print("\n========== 2. UNIT MIX-UP ==========")

    numeric_columns = train.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        if column == TARGET:
            continue

        print(f"\nColumn: {column}")

        stats = train.groupby(GROUP)[column].agg(
            ["min", "mean", "max"]
        )

        print(stats)

        group_means = stats["mean"].abs()
        nonzero_means = group_means[group_means > 0]

        if len(nonzero_means) > 1:
            print(
                "Largest/smallest absolute group mean ratio:",
                round(
                    nonzero_means.max() / nonzero_means.min(),
                    4
                )
            )


def check_sentinel_values(train):
    """Find repeated special numbers in numerical columns."""

    print("\n========== 3. SENTINEL VALUES ==========")

    numeric_columns = train.select_dtypes(
        include="number"
    ).columns

    found = False

    for column in numeric_columns:
        counts = train[column].value_counts()

        for value in SENTINEL_VALUES:
            count = int(counts.get(value, 0))

            if count > 0:
                found = True
                print(
                    f"Column: {column}, "
                    f"Value: {value}, Count: {count}"
                )

    if not found:
        print("No specified sentinel values found.")


def check_duplicate_rows(train):
    """Count extra duplicate records."""

    print("\n========== 4. DUPLICATE ROWS ==========")

    count = int(train.duplicated().sum())

    print("Number of extra duplicate rows:", count)



def check_stuck_sensor(train):
    """Find readings that remain identical across consecutive days."""

    print("\n========== 5. STUCK SENSOR ==========")

    df = train.copy()
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df = df.dropna(subset=[DATE])

    excluded = {TARGET, "cash_reloaded_next_day", "days_to_salary", "record_id"}

    numeric_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]

    found = False

    for column in numeric_columns:
        daily = df.groupby(DATE)[column].agg(
            unique_values="nunique",
            reading="first"
        ).sort_index()

        # A date qualifies only when every row has the same reading.
        daily = daily[daily["unique_values"] == 1]

        dates = list(daily.index)
        if not dates:
            continue

        run_start = dates[0]
        previous_date = dates[0]
        run_value = daily.loc[run_start, "reading"]
        run_length = 1

        for current_date in dates[1:]:
            current_value = daily.loc[current_date, "reading"]

            if (
                current_date == previous_date + pd.Timedelta(days=1)
                and current_value == run_value
            ):
                run_length += 1
            else:
                if run_length >= 7:
                    found = True
                    print(
                        f"Column: {column}, value: {run_value}, "
                        f"days: {run_length}, "
                        f"start: {run_start.date()}, "
                        f"end: {previous_date.date()}"
                    )

                run_start = current_date
                run_value = current_value
                run_length = 1

            previous_date = current_date

        if run_length >= 7:
            found = True
            print(
                f"Column: {column}, value: {run_value}, "
                f"days: {run_length}, "
                f"start: {run_start.date()}, "
                f"end: {previous_date.date()}"
            )

    if not found:
        print("No qualifying stuck-sensor run found.")
def check_target_typos(train):
    """Inspect unusually large target values and gaps."""

    print("\n========== 6. TARGET TYPOS ==========")

    target = train[TARGET].dropna().sort_values()

    print("Target descriptive statistics:")
    print(target.describe())

    print("\nLargest 30 target values:")
    print(target.tail(30).to_string(index=False))

    unique_values = target.drop_duplicates().sort_values()
    gaps = unique_values.diff()

    print("\nLargest gaps between consecutive target values:")
    print(
        pd.DataFrame({
            "previous_value": unique_values.shift(),
            "current_value": unique_values,
            "gap": gaps
        })
        .sort_values("gap", ascending=False)
        .head(15)
        .to_string(index=False)
    )


def normalize_category(value):
    """Normalize spelling and formatting for comparison."""

    value = str(value).strip().lower()
    value = value.translate(
        str.maketrans("", "", string.punctuation)
    )
    return re.sub(r"\s+", "", value)


def check_spelling_variants(train, later):
    """Compare text categories in train and clean later data."""

    print("\n========== 7. SPELLING VARIANTS ==========")

    text_columns = train.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        if column in [TARGET, "record_id", DATE]:
            continue

        print(f"\nColumn: {column}")

        print("\nTraining values:")
        print(train[column].value_counts(dropna=False).to_string())

        if column in later.columns:
            print("\nLater clean values:")
            print(later[column].value_counts(dropna=False).to_string())

            real_levels = (
                later[column]
                .dropna()
                .map(normalize_category)
                .nunique()
            )

            print("Normalized category count in later:", real_levels)


def main():
    train, later, holdout = load_data()

    print("Training dataset shape:", train.shape)
    print("Later dataset shape:", later.shape)
    print("Holdout dataset shape:", holdout.shape)

    check_leaking_column(train, holdout)
    check_unit_mixup(train)
    check_sentinel_values(train)
    check_duplicate_rows(train)
    check_stuck_sensor(train)
    check_target_typos(train)
    check_spelling_variants(train, later)

    print("\n========== INVESTIGATION COMPLETE ==========")


if __name__ == "__main__":
    main()
