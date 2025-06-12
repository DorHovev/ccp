# Customer Churn Pipeline

## Overview
Customer Churn Pipeline is an end-to-end MLOps solution for predicting customer churn. It integrates data ingestion, batch processing, machine learning, and real-time API serving, with robust monitoring and visualization using Prometheus and Grafana. The pipeline is containerized using Docker Compose for easy deployment and scalability.

## Features
- **Batch Data Processing:** Loads and preprocesses customer data from CSV files and databases.
- **Machine Learning Model:** Predicts customer churn using a trained scikit-learn model.
- **REST API:** FastAPI-based service for real-time churn prediction.
- **Database Integration:** Uses PostgreSQL for data storage and management.
- **Monitoring:** Prometheus and Grafana dashboards for metrics and system health.
- **Containerized Deployment:** All components run in isolated Docker containers.
- **CI/CD: Automated testing, build, and deployment with GitHub Actions.
## How It Works

The Customer Churn Pipeline is designed to automate the process of predicting customer churn using a combination of batch processing, machine learning, and real-time API serving. Here's a step-by-step overview of the workflow:

### 1. Data Ingestion & Preprocessing
- **Batch Processor** (`batch_processor/`):  
    Reads input data from CSV files in the project root or batch_processor/input_data/.
    Processes data, applies feature engineering, and predicts churn using the pre-trained model.
    Persists results to the PostgreSQL database.
    Exposes Prometheus metrics for monitoring.
    Logs all operations and errors using Loguru.
    Schedule: Runs daily at 12:00 PM (configurable in batch_processor/config.py).es the cleaned data in a PostgreSQL database for further processing and record-keeping.

### 2. Model Prediction
- **Batch Processing**  
    Reads input data from CSV files in the project root or batch_processor/input_data/.
    Processes data, applies feature engineering, and predicts churn using the pre-trained model.
    Persists results to the PostgreSQL database.
    Exposes Prometheus metrics for monitoring.
    Logs all operations and errors using Loguru.
    Schedule: Runs daily at 12:00 PM (configurable in batch_processor/config.py).
- **REST API**  
    Endpoint: /predict
    Method: POST
    Input: JSON with required features (see API docs for schema).
    Output: Churn prediction (0 or 1).
    Validation: Pydantic models ensure input correctness.
    Health Check: /health endpoint for liveness/readiness probes.
    Metrics: Prometheus metrics for request count, latency, and errors.
    Docs: Accessible at http://localhost:8000/docs (Swagger UI).
### 4. Monitoring & Visualization
- **Prometheus:**  
  - Collects metrics from both the batch processor and API service (such as request counts, errors, and processing times).
- **Grafana:**  
  - Visualizes these metrics using pre-built dashboards (see `dashboard/`).
  - Helps you monitor system health, model performance, and resource usage.
- **Pushgateway, cAdvisor, Node Exporter:**  
  - Support advanced monitoring of container and system metrics.

### 5. Containerized Deployment
- **Docker Compose:**  
  - Orchestrates all services (database, batch processor, API, monitoring stack) for easy setup and scaling.
  - Ensures each component runs in its own isolated environment.

## What the Project Includes

- **api_service/**  
  FastAPI app for real-time churn prediction, including:
  - Model loading and inference
  - REST endpoints (`/predict`, `/health`)
  - Logging and error handling
  - Prometheus metrics integration

- **batch_processor/**  
  Batch processing engine for:
  - Data ingestion from CSV/database
  - Data cleaning and preprocessing
  - Batch prediction and result storage
  - Monitoring and metrics export

- **dashboard/**  
  Grafana dashboard JSON files for:
  - API and batch processing metrics
  - System and Docker container monitoring

- **docker-compose.yml**  
  Configuration for multi-service orchestration, including:
  - PostgreSQL database
  - Batch processor
  - API service
  - Prometheus, Grafana, Pushgateway, cAdvisor, Node Exporter

- **prometheus.yml, promtail-config.yaml**  
  Configuration files for Prometheus and log collection.

- **churn_model.pickle**  
  Pre-trained machine learning model for churn prediction.

- **tests/**  
  (If present) Automated tests for API and batch processing logic.

- **Data_preparation.ipynb**  
  Jupyter notebook for data exploration and model training (if you want to retrain or experiment).
- **CI/CD**  
    GitHub Actions: Automated workflow for:
    Linting and testing
    Building Docker images
    Deploying containers
## Setup
### Prerequisites
- Docker & Docker Compose

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ccp
   ```
2. **Start the pipeline:**
   ```bash
   docker-compose up --build
   ```
3. **Access services:**
   - API: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Grafana: [http://localhost:3000](http://localhost:3000)
   - Prometheus: [http://localhost:9090](http://localhost:9090)

## Usage
- **Batch Processing:** Automatically loads and processes data from `batch_processor/input_data/`.
- **API Service:**
  - Send POST requests to `/predict` endpoint with customer features to get churn predictions.
  - Example request:
    ```json
    {
      "customerID": "12345",
      "gender": "Female",
      "SeniorCitizen": 0,
      ...
    }
    ```
- **Monitoring:**
  - Metrics are available at `/metrics` endpoints and visualized in Grafana dashboards (see `dashboard/` directory for sample dashboards).

## Project Structure
- `api_service/` - FastAPI app for real-time predictions
- `batch_processor/` - Batch data ingestion, preprocessing, and prediction
- `dashboard/` - Grafana dashboard JSONs
- `docker-compose.yml` - Multi-service orchestration
- `prometheus.yml`, `promtail-config.yaml` - Monitoring configuration

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

