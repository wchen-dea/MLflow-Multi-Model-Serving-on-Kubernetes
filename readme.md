# MLflow Multi-Model Serving on Kubernetes (uv)

This project is configured as a uv-native Python project.

Operating model:

- MLflow is used for ML lifecycle management: experiment tracking, metric logging, champion evaluation, and model registry
- uv is used for operations: environment sync, dependency execution, and running project commands

It covers:

- Hyperparameter tuning with RandomizedSearchCV
- Champion-gated wine and California training with MLflow tracking and registry integration
- Deploying the same model-serving stack to local Kubernetes or production Kubernetes
- Sending dedicated inference requests for wine and California housing payloads

## Prerequisites

- Docker Desktop (or Docker Engine)
- Minikube
- kubectl
- uv (<https://docs.astral.sh/uv/>)

## Quick Start (Makefile)

Use the built-in Make targets:

```bash
make help
```

Useful targets:

- make env-health: recreate .venv and verify dependency imports
- make build-images: build local serving images from latest MLflow model artifacts
- make load-images: validate and load local model images into Minikube
- make deploy: unified Kubernetes deployment path for local or production
- make mlflow-server: run the MLflow tracking and registry server
- make train-all: run both MLflow-managed training flows
- make infer-wine: run wine inference against the deployed service
- make infer-california: run california inference against the deployed service

## Responsibilities

MLflow handles:

- experiment creation and run tracking
- metric and parameter logging
- champion threshold decisions
- model registration in MLflow Model Registry

uv handles:

- environment creation and dependency sync
- running Python training and inference entrypoints
- invoking operational commands through Make targets

## Deployment Path

This project keeps a single Kubernetes deployment path shared by:

- local Docker + Minikube
- production Kubernetes with registry-hosted images

1. Prepare Python environment:

```bash
make env-health
```

1. Start MLflow tracking server in a separate terminal:

```bash
make mlflow-server
```

1. Run the MLflow-managed training lifecycle:

```bash
make train-all
```

This runs both model training flows, evaluates champion thresholds, and registers passing models into MLflow Model Registry.

1. Deploy to local Minikube using local images:

```bash
make deploy
```

This local path automatically:

- starts or reuses Minikube
- ensures the namespace exists
- builds local serving images from the latest MLflow artifacts
- loads those images into Minikube
- deploys the rendered Kubernetes manifest

1. Or deploy to production using registry images:

```bash
make deploy LOCAL_CLUSTER=false \
  NAMESPACE=your-namespace \
  WINE_IMAGE=registry.example.com/your-team/wine-classifier:tag \
  CALIFORNIA_IMAGE=registry.example.com/your-team/california-housing-xgboost:tag
```

This uses the same manifest path, but skips Minikube-specific build/load steps.

1. Send inference request:

```bash
make infer-wine
make infer-california
```

## Manual uv Commands (optional)

If you prefer not to use Make targets, use direct uv/kubectl commands below.

### 1) Install dependencies with uv

From the project directory:

```bash
uv sync
```

This is the operational entrypoint for restoring the environment.

### 2) Start local Kubernetes

```bash
minikube start --driver=docker
kubectl create namespace mlflow-kserve-test --dry-run=client -o yaml | kubectl apply -f -
```

### 3) Build and deploy the model service locally

Use the shared Kubernetes manifest:

```bash
make build-images
make load-images
kubectl create namespace mlflow-kserve-test --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f .build/model-serving.rendered.yml
kubectl rollout status deployment/wine-classifier -n mlflow-kserve-test
```

For production, render and deploy the same path with different image values:

```bash
make deploy LOCAL_CLUSTER=false \
  NAMESPACE=your-namespace \
  WINE_IMAGE=registry.example.com/your-team/wine-classifier:tag \
  CALIFORNIA_IMAGE=registry.example.com/your-team/california-housing-xgboost:tag
```

### 4) Start MLflow tracking server

In terminal A:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

This server backs the full ML lifecycle for the project, including experiment tracking and model registry.

### 5) Run MLflow-managed training

In terminal B:

```bash
uv run python train/train-wine-hpt.py
uv run python train/train-california-xgboost.py
```

Both scripts:

- log runs, params, and metrics to MLflow
- evaluate a champion threshold
- register the model only when the threshold is met

MLflow UI and run links are available at:

- <http://127.0.0.1:5000>

### 6) Port-forward and send inference request

In terminal C:

```bash
kubectl -n mlflow-kserve-test port-forward svc/wine-classifier 8080:8080
```

In terminal D, send wine inference with Python:

```bash
uv run python infer/infer-wine.py
```

Send California housing inference with Python:

```bash
uv run python infer/infer-california.py
```

## Notes

- The file k8s/model-serving.yml is the shared Kubernetes manifest template used for both local and production deployment.
- The rendered deployment manifest is written to .build/model-serving.rendered.yml.
- The Kubernetes manifests expect local images `wine-classifier:latest` and `california-housing-xgboost:latest`; `make build-images` creates them from the latest local MLflow artifacts.
- The `train/` folder contains data sourcing and training scripts.
- The `model/` folder contains model image build and serving assets.
- The `infer/` folder contains runtime inference clients.
- The `k8s/` folder contains Kubernetes deployment manifests.
