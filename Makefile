.PHONY: help install ingest feature train validate test up down clean

help:
	@echo "🚀 MLOps Pipeline Commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make ingest    - Run Data Ingestion (MinIO -> Processed)"
	@echo "  make feature   - Run Feature Engineering (Processed -> Featured)"
	@echo "  make validate  - Run Data Validation (Pandera)"
	@echo "  make train     - Run Model Training (Featured -> Model)"
	@echo "  make test      - Run Unit Tests"
	@echo "  make up        - Start Docker Stack (API + Monitor + MLflow)"
	@echo "  make down      - Stop Docker Stack"
	@echo "  make clean     - Clean temporary files"

install:
	pip install -r requirements.txt

# Pipeline Steps
ingest:
	PYTHONPATH=. python src/components/data_ingestion.py

feature:
	PYTHONPATH=. python src/components/feature_engineering.py

validate:
	PYTHONPATH=. python src/components/data_validation.py

train:
	export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 && \
	export AWS_ACCESS_KEY_ID=admin && \
	export AWS_SECRET_ACCESS_KEY=password123 && \
	PYTHONPATH=. python src/components/model_training.py --config config/config.yaml --params config/params.yaml

# Testing
test:
	PYTHONPATH=. pytest tests/

# Docker
up:
	docker compose up -d --build

down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache

# ... existing commands ...

# Deployment Helper
# Replace 'your_username' with your actual Docker Hub username
DOCKER_USER = abhyyshake
IMAGE_NAME = kkbox-churn-api

deploy:
	@echo "🚀 Building Production Image with REAL Model..."
	docker build -t $(DOCKER_USER)/$(IMAGE_NAME):latest -f docker/Dockerfile.api .
	
	@echo "☁️  Pushing to Docker Hub..."
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest
	
	@echo "✅ Deployment Complete! The new model is live on Docker Hub."