import mlflow
import numpy as np
from scipy.stats import loguniform
import sys
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient


CHAMPION_SCORE_THRESHOLD = 0.75
REGISTERED_MODEL_NAME = "wine-classifier-champion"
MODEL_PIP_REQUIREMENTS = [
    "mlflow==3.14.0",
    "numpy==2.5.1",
    "scikit-learn==1.9.0",
    "scipy==1.18.0",
    "skops==0.14.0",
]
MODEL_CONDA_ENV = {
    "name": "wine-classifier-env",
    "channels": ["conda-forge"],
    "dependencies": [
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pip",
        {"pip": MODEL_PIP_REQUIREMENTS},
    ],
}

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                solver="saga",
                max_iter=10000,
                tol=1e-3,
                random_state=42,
            ),
        ),
    ]
)


def eval_metrics(pred, actual):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average="weighted", zero_division=0)
    recall = recall_score(actual, pred, average="weighted", zero_division=0)
    f1 = f1_score(actual, pred, average="weighted", zero_division=0)
    return accuracy, precision, recall, f1

# Set th experiment 
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Wine-Quality-II")

# Load wine quality dataset
X, y = datasets.load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Define distribution to pick parameter values from
distributions = {
    "classifier__C": loguniform(1e-3, 1e2),
    "classifier__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
}

# Initialize random search instance
search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    # Optimize for classification accuracy
    scoring="accuracy",
    # Use 5-fold cross validation
    cv=5,
    # Try 100 samples. Note that MLflow only logs the top 5 runs.
    n_iter=100,
    random_state=42,
)

# Start a parent run
with mlflow.start_run(run_name="hyperparameter-tuning"):
    search = search_cv.fit(X_train, y_train)
    mlflow.log_params(search_cv.best_params_)

    # Evaluate the best model on test dataset
    y_pred = search_cv.best_estimator_.predict(X_test)
    accuracy, precision, recall, f1 = eval_metrics(y_pred, y_test)
    mlflow.log_metrics(
        {
            "accuracy_X_test": accuracy,
            "precision_weighted_X_test": precision,
            "recall_weighted_X_test": recall,
            "f1_weighted_X_test": f1,
        }
    )
    mlflow.log_params(
        {
            "champion_score_threshold": CHAMPION_SCORE_THRESHOLD,
            "registered_model_name": REGISTERED_MODEL_NAME,
        }
    )

    if accuracy >= CHAMPION_SCORE_THRESHOLD:
        model_info = mlflow.sklearn.log_model(
            sk_model=search_cv.best_estimator_,
            name="model",
            serialization_format="skops",
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
                "reason": f"accuracy {accuracy:.4f} < {CHAMPION_SCORE_THRESHOLD:.2f}",
            }
        )

    print(
        {
            "accuracy_X_test": accuracy,
            "precision_weighted_X_test": precision,
            "recall_weighted_X_test": recall,
            "f1_weighted_X_test": f1,
        }
    )
