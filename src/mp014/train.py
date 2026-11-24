import hashlib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.mp014.preprocessing import (
    ROOT,
    load_config,
    clean_features,
    build_preprocessor
)


# Load and clean training dataset
def load_training_data():
    config = load_config()

    train_path = ROOT / config["data"]["train_path"]

    df = pd.read_csv(train_path)

    X = clean_features(df)
    y = df[config["data"]["target"]]

    return X, y


# Split dataset into training and validation sets
def split_data(X, y):
    config = load_config()

    return train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )


# Initialize regression models
def get_models():
    config = load_config()

    settings = config["models"]
    seed = config["training"]["random_state"]

    return {
        "linear_regression": LinearRegression(
            **settings["linear_regression"]
        ),

        "random_forest": RandomForestRegressor(
            **settings["random_forest"],
            random_state=seed,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            **settings["gradient_boosting"],
            random_state=seed
        )
    }


# Train, evaluate and track all models
def main():

    config = load_config()

    # Configure MLflow
    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    # Calculate original training dataset SHA-256 hash
    train_path = ROOT / config["data"]["train_path"]

    data_sha256 = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()

    print("Dataset SHA-256:", data_sha256)

    # Load and split dataset
    X, y = load_training_data()

    X_train, X_val, y_train, y_val = split_data(X, y)

    models = get_models()
    results = {}

    # Train and track each model separately
    for name, model in models.items():

        print(f"\nTraining: {name}")

        with mlflow.start_run(run_name=name) as run:

            # Create preprocessing and model pipeline
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("model", model)
            ])

            # Train model
            pipeline.fit(X_train, y_train)

            # Generate validation predictions
            predictions = pipeline.predict(X_val)

            # Calculate MAE
            mae = mean_absolute_error(
                y_val,
                predictions
            )

            # Log model information
            mlflow.log_param("model_name", name)
            mlflow.log_param("data_sha256", data_sha256)

            mlflow.log_param(
                "random_state",
                config["training"]["random_state"]
            )

            mlflow.log_param(
                "test_size",
                config["training"]["test_size"]
            )

            # Log model hyperparameters
            mlflow.log_params(config["models"][name])

            # Log dataset split sizes
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))

            # Log evaluation metric
            mlflow.log_metric("mae", mae)

            # Save trained pipeline in MLflow
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle"
            )

            # Store model URI for later registration
            mlflow.set_tag(
                "model_uri",
                model_info.model_uri
            )

            # Store experiment results
            results[name] = {
                "mae": mae,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri
            }

            print(f"MAE: {mae:.4f}")
            print("Run ID:", run.info.run_id)
            print("Model URI:", model_info.model_uri)

    # Select model with lowest MAE
    best_model = min(
        results,
        key=lambda name: results[name]["mae"]
    )

    best_result = results[best_model]

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    print("Best model:", best_model)
    print("Best MAE:", round(best_result["mae"], 4))
    print("Best Run ID:", best_result["run_id"])
    print("Best Model URI:", best_result["model_uri"])


if __name__ == "__main__":
    main()