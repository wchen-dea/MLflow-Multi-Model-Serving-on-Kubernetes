from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sys
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient


CHAMPION_SCORE_THRESHOLD = 0.75
REGISTERED_MODEL_NAME = "housing-xgboost-champion"
MODEL_PIP_REQUIREMENTS = [
    "mlflow==3.14.0",
    "numpy==2.5.1",
    "pandas==2.3.3",
    "scikit-learn==1.9.0",
    "scipy==1.18.0",
    "xgboost==3.3.0",
]
MODEL_CONDA_ENV = {
    "name": "housing-xgboost-env",
    "channels": ["conda-forge"],
    "dependencies": [
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pip",
        {"pip": MODEL_PIP_REQUIREMENTS},
    ],
}


def eval_metrics(pred, actual):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def plot_correlation_with_MedHouseVal(df):
    correlations = df.corr(numeric_only=True)["MedHouseVal"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.drop(labels=["MedHouseVal"]).plot(kind="bar", ax=ax)
    ax.set_title("Feature Correlation With MedHouseVal")
    ax.set_ylabel("Pearson Correlation")
    ax.set_xlabel("Feature")
    fig.tight_layout()

    artifact_path = Path("plot_correlation_with_MedHouseVal.png")
    fig.savefig(artifact_path)
    plt.close(fig)
    return artifact_path


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Housing-XGBoost-Experiment")

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target
full_df = X.copy()
full_df["MedHouseVal"] = y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

with mlflow.start_run(run_name="xgboost-default"):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse, mae, r2 = eval_metrics(y_pred, y_test)
    mlflow.log_metrics(
        {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
    )
    mlflow.log_params(model.get_params())
    mlflow.log_params(
        {
            "champion_score_threshold": CHAMPION_SCORE_THRESHOLD,
            "registered_model_name": REGISTERED_MODEL_NAME,
        }
    )

    plot_path = plot_correlation_with_MedHouseVal(full_df)
    mlflow.log_artifact(str(plot_path), artifact_path="plots")

    if r2 >= CHAMPION_SCORE_THRESHOLD:
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            conda_env=MODEL_CONDA_ENV,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        registered_version = next(
            version.version
            for version in versions
            if version.run_id == mlflow.active_run().info.run_id
        )
        mlflow.set_tag("champion_status", "passed")
        mlflow.set_tag("registered_model_version", registered_version)
        print(
            {
                "champion_status": "passed",
                "registered_model_name": REGISTERED_MODEL_NAME,
                "registered_model_version": registered_version,
                "model_uri": model_info.model_uri,
            }
        )
    else:
        mlflow.set_tag("champion_status", "failed")
        print(
            {
                "champion_status": "failed",
                "reason": f"r2 {r2:.4f} < {CHAMPION_SCORE_THRESHOLD:.2f}",
            }
        )

    print(
        {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "plot": str(plot_path),
        }
    )