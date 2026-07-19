#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/.build"
WINE_CONTEXT="$BUILD_DIR/wine"
CALIFORNIA_CONTEXT="$BUILD_DIR/california"
WINE_IMAGE=${WINE_IMAGE:-wine-classifier:latest}
CALIFORNIA_IMAGE=${CALIFORNIA_IMAGE:-california-housing-xgboost:latest}
BUILD_ENGINE=${BUILD_ENGINE:-docker}

latest_model_dir() {
  local base_dir=$1
  local latest
  latest=$(ls -td "$base_dir"/* 2>/dev/null | head -n 1 || true)
  if [[ -z "$latest" ]]; then
    echo "No model bundle found under $base_dir" >&2
    exit 1
  fi
  printf '%s' "$latest"
}

write_mlserver_config() {
  local target_dir=$1
  cat > "$target_dir/app.py" <<'EOF'
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_NAME = "mlflow-model"
MODEL_URI = str(Path(__file__).resolve().parent / "model-bundle")
MODEL = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI()


class RequestInput(BaseModel):
  name: str
  shape: list[int]
  datatype: str
  data: list


class InferenceRequest(BaseModel):
  inputs: list[RequestInput]


@app.get("/v2/health/ready")
def ready():
  return {"ready": True}


@app.get("/v2/models")
def list_models():
  return {"models": [{"name": MODEL_NAME}]}


@app.get("/v2/models/{model_name}")
def model_metadata(model_name: str):
  if model_name != MODEL_NAME:
    raise HTTPException(status_code=404, detail="model not found")
  return {
    "name": MODEL_NAME,
    "versions": [],
    "platform": "mlflow-pyfunc",
    "inputs": [],
    "outputs": [],
    "parameters": {},
  }


@app.post("/v2/models/{model_name}/infer")
def infer(model_name: str, request: InferenceRequest):
  if model_name != MODEL_NAME:
    raise HTTPException(status_code=404, detail="model not found")
  if not request.inputs:
    raise HTTPException(status_code=400, detail="missing inputs")

  values = np.asarray(request.inputs[0].data)
  if values.ndim == 1:
    values = values.reshape(1, -1)

  predictions = MODEL.predict(values)
  output = np.asarray(predictions)

  return {
    "model_name": MODEL_NAME,
    "outputs": [
      {
        "name": "predict",
        "shape": list(output.shape),
        "datatype": str(output.dtype).upper(),
        "data": output.tolist(),
      }
    ],
  }
EOF
}

WINE_MODEL_DIR=$(latest_model_dir "$ROOT_DIR/mlruns/1/models")
CALIFORNIA_MODEL_DIR=$(latest_model_dir "$ROOT_DIR/mlruns/2/models")

rm -rf "$WINE_CONTEXT" "$CALIFORNIA_CONTEXT"
mkdir -p "$WINE_CONTEXT/model-bundle" "$CALIFORNIA_CONTEXT/model-bundle"

cp -R "$WINE_MODEL_DIR/artifacts/." "$WINE_CONTEXT/model-bundle/"
cp -R "$CALIFORNIA_MODEL_DIR/artifacts/." "$CALIFORNIA_CONTEXT/model-bundle/"
cp "$ROOT_DIR/model/Dockerfile.mlserver" "$WINE_CONTEXT/Dockerfile.mlserver"
cp "$ROOT_DIR/model/Dockerfile.mlserver" "$CALIFORNIA_CONTEXT/Dockerfile.mlserver"

write_mlserver_config "$WINE_CONTEXT"
write_mlserver_config "$CALIFORNIA_CONTEXT"

cd "$ROOT_DIR"

if [[ "$BUILD_ENGINE" == "minikube" ]]; then
  (
    cd .build/wine
    minikube image build -f Dockerfile.mlserver -t "$WINE_IMAGE" .
  )
  (
    cd .build/california
    minikube image build -f Dockerfile.mlserver -t "$CALIFORNIA_IMAGE" .
  )
else
  (
    cd .build/wine
    docker build -f Dockerfile.mlserver -t "$WINE_IMAGE" .
  )
  (
    cd .build/california
    docker build -f Dockerfile.mlserver -t "$CALIFORNIA_IMAGE" .
  )
fi

echo "Built local images:"
echo "  $WINE_IMAGE"
echo "  $CALIFORNIA_IMAGE"