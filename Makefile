.PHONY: help uv-sync env-health k8s-up k8s-ns build-images load-images render-manifest deploy mlflow-server tune train-xgboost train-all infer-wine infer-california full teardown

SHELL := /bin/zsh
PYTHON := uv run python

NAMESPACE := mlflow-kserve-test
SERVICE_WINE := wine-classifier
SERVICE_CALIFORNIA := california-housing
LOCAL_CLUSTER ?= true
WINE_IMAGE := wine-classifier:latest
CALIFORNIA_IMAGE := california-housing-xgboost:latest
WINE_PORT := 8080
CALIFORNIA_PORT := 8081
MODEL_ENDPOINT := /v2/models/mlflow-model/infer
INFER_URL := http://localhost:$(WINE_PORT)$(MODEL_ENDPOINT)
MANIFEST_TEMPLATE := k8s/model-serving.yml
RENDERED_MANIFEST := .build/model-serving.rendered.yml
BUILD_SCRIPT := model/build-local-images.sh
TRAIN_WINE_SCRIPT := train/train-wine-hpt.py
TRAIN_CALIFORNIA_SCRIPT := train/train-california-xgboost.py
INFER_WINE_SCRIPT := infer/infer-wine.py
INFER_CALIFORNIA_SCRIPT := infer/infer-california.py

help:
	echo "Available targets:"
	echo "  uv-sync       - Install/update dependencies with uv"
	echo "  env-health    - Recreate .venv and run a quick dependency health check"
	echo "  k8s-up        - Start local minikube cluster"
	echo "  k8s-ns        - Create Kubernetes namespace"
	echo "  build-images  - Build local model-serving images from latest MLflow artifacts"
	echo "  load-images   - Validate and load local model images into Minikube"
	echo "  deploy        - Unified Kubernetes deploy path for local or production"
	echo "  mlflow-server - Start MLflow tracking server (foreground)"
	echo "  tune          - Run hyperparameter tuning"
	echo "  train-xgboost - Run XGBoost training with MLflow logging"
	echo "  train-all     - Run tuning and XGBoost training sequentially"
	echo "  infer-wine    - Port-forward wine service and submit wine inference request"
	echo "  infer-california - Port-forward california service and submit california inference request"
	echo "  full          - Run end-to-end local pipeline sequentially"
	echo "  teardown      - Delete minikube cluster, remove local images, and clean build artifacts"

uv-sync:
	uv sync

env-health:
	rm -rf .venv
	uv sync
	$(PYTHON) -c "import mlflow, sklearn, scipy, requests; print('env healthy:', mlflow.__version__, sklearn.__version__, scipy.__version__, requests.__version__)"

k8s-up:
	minikube start --driver=docker

k8s-ns:
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

build-images:
	chmod +x $(BUILD_SCRIPT)
	BUILD_ENGINE=minikube WINE_IMAGE=$(WINE_IMAGE) CALIFORNIA_IMAGE=$(CALIFORNIA_IMAGE) ./$(BUILD_SCRIPT)

load-images:
	@if [[ "$(LOCAL_CLUSTER)" == "true" ]]; then \
		echo "Images already loaded into minikube by build-images. Skipping."; \
	else \
		docker image inspect $(WINE_IMAGE) >/dev/null 2>&1 || { \
			echo "Missing local image: $(WINE_IMAGE)"; \
			echo "Build or tag it locally before running make load-images."; \
			exit 1; \
		}; \
		docker image inspect $(CALIFORNIA_IMAGE) >/dev/null 2>&1 || { \
			echo "Missing local image: $(CALIFORNIA_IMAGE)"; \
			echo "Build or tag it locally before running make load-images."; \
			exit 1; \
		}; \
		minikube image load --overwrite=true $(WINE_IMAGE); \
		minikube image load --overwrite=true $(CALIFORNIA_IMAGE); \
	fi

render-manifest:
	mkdir -p .build
	sed \
		-e 's|__NAMESPACE__|$(NAMESPACE)|g' \
		-e 's|__WINE_IMAGE__|$(WINE_IMAGE)|g' \
		-e 's|__CALIFORNIA_IMAGE__|$(CALIFORNIA_IMAGE)|g' \
		$(MANIFEST_TEMPLATE) > $(RENDERED_MANIFEST)

deploy: render-manifest
	@if [[ "$(LOCAL_CLUSTER)" == "true" ]]; then \
		$(MAKE) k8s-up; \
		$(MAKE) k8s-ns; \
		$(MAKE) build-images; \
	else \
		kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -; \
	fi
	kubectl apply -f $(RENDERED_MANIFEST)
	kubectl rollout restart deployment/$(SERVICE_WINE) -n $(NAMESPACE) || true
	kubectl rollout restart deployment/$(SERVICE_CALIFORNIA) -n $(NAMESPACE) || true
	kubectl rollout status deployment/$(SERVICE_WINE) -n $(NAMESPACE)
	kubectl rollout status deployment/$(SERVICE_CALIFORNIA) -n $(NAMESPACE)

mlflow-server:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

tune:
	$(PYTHON) $(TRAIN_WINE_SCRIPT)

train-xgboost:
	$(PYTHON) $(TRAIN_CALIFORNIA_SCRIPT)

train-all:
	$(MAKE) tune
	$(MAKE) train-xgboost

infer-wine:
	set -e; \
	kubectl -n $(NAMESPACE) port-forward svc/$(SERVICE_WINE) $(WINE_PORT):8080 >/tmp/$(SERVICE_WINE)-port-forward.log 2>&1 & \
	PF_PID=$$!; \
	trap 'kill $$PF_PID >/dev/null 2>&1 || true' EXIT; \
	sleep 2; \
	INFER_URL=$(INFER_URL) $(PYTHON) $(INFER_WINE_SCRIPT); \
	echo

infer-california:
	set -e; \
	kubectl -n $(NAMESPACE) port-forward svc/$(SERVICE_CALIFORNIA) $(CALIFORNIA_PORT):8080 >/tmp/$(SERVICE_CALIFORNIA)-port-forward.log 2>&1 & \
	PF_PID=$$!; \
	trap 'kill $$PF_PID >/dev/null 2>&1 || true' EXIT; \
	sleep 2; \
	INFER_URL=http://localhost:$(CALIFORNIA_PORT)$(MODEL_ENDPOINT) $(PYTHON) $(INFER_CALIFORNIA_SCRIPT); \
	echo

full:
	$(MAKE) uv-sync
	$(MAKE) deploy
	echo "Start MLflow server in another terminal: make mlflow-server"
	$(MAKE) train-all

teardown:
	pkill -f "mlflow server" 2>/dev/null || true
	minikube delete || true
	docker rmi -f $(WINE_IMAGE) $(CALIFORNIA_IMAGE) 2>/dev/null || true
	rm -rf .build mlflow.db
