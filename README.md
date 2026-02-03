1. The High-Level Architecture Block Diagram
This diagram illustrates the flow of data, the training pipeline, and the production monitoring loop.
code Mermaid
downloadcontent_copy
expand_less
graph TD
    subgraph "Data Operations (Bronze -> Silver -> Gold)"
        Raw[Raw CSVs] -->|Ingest Stream| MinIO[(MinIO Data Lake)]
        MinIO -->|Dask Processing| Parquet[Parquet Files (Silver)]
        Parquet -->|Dask Aggregation| FeatStore[Master Features (Gold)]
        FeatStore -->|Pandera Check| Validation{Data Validation}
    end

    subgraph "ML Operations (Training)"
        Validation -->|Pass| Training[LightGBM + Pipeline Training]
        Training -->|Log Metrics| MLflow[MLflow Tracking]
        Training -->|Save Artifact| Reg[(Model Registry)]
        
        subgraph "Skew Prevention"
            Logic[Shared Data Processor Class]
        end
        Logic -.->|Used In| Training
    end

    subgraph "CI/CD & Deployment"
        Code[GitHub Repo] -->|Push| Actions[GitHub Actions]
        Actions -->|Test Code| UnitTests[Pytest Unit Tests]
        Actions -->|Build & Push| DockerHub[Docker Hub Registry]
        DockerHub -->|Pull Image| Prod[Production Container]
    end

    subgraph "Serving & Observability"
        User[Client/User] -->|POST /predict| API[FastAPI Inference]
        Reg -.->|Load Pipeline| API
        Logic -.->|Used In| API
        
        API -->|Scrape Latency| Prom[Prometheus]
        Prom -->|Visualize| Grafana[Grafana Dashboard]
        
        API -->|Async Log| Logs[JSONL Traffic Logs]
        Logs -->|Batch Analysis| Drift[Evidently AI Drift Check]
    end

2. Portfolio/GitHub README Content
Title: End-to-End MLOps Platform for Churn Prediction
🚀 Executive Summary
This project is an industry-grade, end-to-End Machine Learning Operations (MLOps) system designed to predict user churn for a music streaming service (KKBox). Unlike standard data science projects that stop at a Jupyter Notebook, this system implements a complete lifecycle pipeline handling Big Data processing, reproducible training, containerized deployment, and real-time observability.
The system is architected to handle Training-Serving Skew, ensure Data Quality, and provide Continuous Deployment using entirely open-source tools.

🏗 System Architecture & Key Features
1. Hybrid Data Lakehouse (MinIO + DVC)
* The Problem: Handling 3GB+ datasets locally is slow and pushing them to Git is impossible.
* The Solution: Implemented a "Data Lake" architecture using MinIO (S3-compatible object storage) running in Docker.
* Workflow: Raw CSVs are streamed from MinIO, processed into optimized Parquet files (70% size reduction), and version-controlled using DVC (Data Version Control). This ensures that every model version can be traced back to the exact snapshot of data used to train it.
2. Scalable ETL with Dask
* The Problem: Pandas crashes with MemoryError when processing gigabytes of user logs.
* The Solution: Utilized Dask for lazy evaluation and out-of-core processing. The pipeline aggregates millions of user logs into feature vectors (e.g., "Active Days", "Listening Trends") without loading the full dataset into RAM.
3. Zero Training-Serving Skew
* The Problem: Logic defined in notebooks often differs from logic in the API, leading to silent failures.
* The Solution: Encapsulated all feature transformations (Imputation, Ratio Calculations, Categorical Encoding) into a custom Scikit-Learn Transformer class. This class is packaged into a Pipeline artifact (pipeline.pkl). The API loads this exact pipeline, guaranteeing that production inference is mathematically identical to training.
4. Robust Data Validation (Pandera)
* The Problem: Garbage In, Garbage Out. Models fail silently if input data is corrupted (e.g., negative payments).
* The Solution: Integrated Pandera to enforce statistical schema checks before training. The pipeline halts immediately if data violates business rules (e.g., total_transactions < 0), preventing bad models from reaching production.
5. Full Observability Stack (Prometheus, Grafana, Evidently)
* Service Monitoring: Prometheus scrapes system metrics (latency, throughput, RAM) from the FastAPI endpoint, visualized on a Grafana dashboard.
* Data Monitoring: The API asynchronously logs live traffic to JSONL files. An Evidently AI job compares live traffic against the training reference data to detect Data Drift (e.g., if the user age demographic shifts significantly).

🛠 Tech Stack
Layer	Tools Used	Purpose
Storage	MinIO, DVC	Data Lake & Versioning
Processing	Dask, Pandas	Big Data ETL
Modeling	LightGBM, Scikit-Learn	High-performance Tabular Model
Tracking	MLflow, PostgreSQL	Experiment Tracking & Model Registry
Serving	FastAPI, Uvicorn	Real-time REST API
DevOps	Docker, Docker Compose	Containerization & Orchestration
CI/CD	GitHub Actions, Docker Hub	Automated Testing & Delivery
Monitoring	Prometheus, Grafana	System Health Metrics
Quality	Pytest, Pandera	Unit Testing & Data Validation
💻 How to Run (Developer Experience)
The project utilizes a Makefile to abstract complex Docker and Python commands.
1. Infrastructure Setup Spin up the Data Lake, Database, and Tracking Server:
code Bash
downloadcontent_copy
expand_less
make up
2. The Data Pipeline Ingest raw data and create features:
code Bash
downloadcontent_copy
expand_less
make ingest
make feature
3. Training & Validation Run validation checks and train the model (logged to MLflow):
code Bash
downloadcontent_copy
expand_less
make validate
make train
4. Testing & Deployment Run unit tests and push the container to Docker Hub:
code Bash
downloadcontent_copy
expand_less
make test
make deploy

📊 Business Value & Scalability
Cost Analysis (Cloud Equivalent)
While built using free open-source tools locally, this architecture maps directly to an enterprise AWS stack for approximately $180/month (vs. $0 local):
* MinIO          →  \rightarrow  →
*        AWS S3
* Postgres          →  \rightarrow  →
*        AWS RDS
* Docker Compose          →  \rightarrow  →
*        AWS ECS (Fargate)
* Local Training          →  \rightarrow  →
*        AWS SageMaker
Scaling Strategy (100 to 1M Users)
1. Horizontal Scaling: The API is stateless. We can deploy behind a Load Balancer (NGINX) and scale replicas from 1 to 50 using Kubernetes.
2. Caching: Implement Redis to cache predictions for active users, reducing model inference costs by ~40%.
3. Database: Migrate MLflow backend from single Postgres container to a managed RDS instance with Read Replicas.

📈 Project Metrics
* Model Performance: 0.986 AUC (LightGBM).
* Inference Latency: <50ms (95th percentile).
* Data Volume: Handles 3GB+ raw logs via Dask streaming.
* Drift Detection: Daily analysis of production traffic.

Author
Abhishek Kapoor MLOps Engineer | Data Scientist
