# ATM Cash Demand Prediction — Lucknow

**Course:** SCSE3040 – MLOps  
**Project ID:** MP014  
**Group:** G201 | **Batch:** 48  
**Faculty:** Dr. Saurabh Shrivastava

## Project Overview

This mini project develops an end-to-end MLOps pipeline to predict daily ATM cash withdrawals for a fictional cooperative bank in Lucknow. The goal is to support cash replenishment planning, reduce cash shortages, and minimize idle cash.

## Problem Statement

Predict `withdrawal_thousand_rs` (daily withdrawals in thousands of rupees) using ATM location and activity features.

- **Problem type:** Regression
- **Evaluation metric:** Mean Absolute Error (MAE)
- **Training data:** January–June 2026 (`train.csv`)
- **Monitoring data:** July–August 2026 (`later.csv`)
- **Holdout data:** September 2026 (`holdout_features.csv`)

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, DVC, MLflow, FastAPI, Pydantic, Docker, GitHub Actions, Kubernetes, and Pytest.

## Project Workflow

1. **Data analysis and cleaning:** Check seven possible data-quality issues and prepare training data.
2. **Version control:** Track datasets with DVC and code with Git/GitHub.
3. **Model development:** Train regression models, track experiments, and register the champion model in MLflow.
4. **Prediction API:** Implement `/predict` and `/health` with FastAPI and input validation.
5. **Deployment and testing:** Configure Docker, automated tests, GitHub Actions CI, Kubernetes, and JSON logging.
6. **Monitoring and retraining:** Calculate PSI, evaluate MAE, and apply the model-promotion rule.
7. **Final output:** Generate holdout predictions and complete `reports/findings.yaml`.

## Team Members

| Name | Enrollment Number |
|---|---|
| Tanishq Jindal | S24CSEU1874 |
| Kanishk Saini | S24CSEU1861 |
| Lakshya Choudhary | S24CSEU1886 |
| Nitish Kumar | S24CSEU1882 |

## Expected Deliverables

DVC metadata, cleaning and training code, MLflow tracking database and registered model, FastAPI service, Docker and Kubernetes configuration, pytest tests, GitHub Actions CI, drift and retraining results, `reports/findings.yaml`, and `predictions/holdout_predictions.csv`.

## Submission

**Deadline:** 29 November 2026, 11:59 PM.

Each member must commit from their own GitHub account and individually submit the same public repository link.

---

*SCSE3040 – MLOps Mini Project | Group G201*
