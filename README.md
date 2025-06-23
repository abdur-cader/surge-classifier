
# Surge Classifier

## Overview

This project is an extension of [Surge Predictor](https://github.com/abdur-cader/surge-predictor) — designed for deployment using Docker, MLflow, FastAPI and Render.

## Features

✅ Machine learning surge level classification (Random Forest, XGBoost, LightGBM, CatBoost)  
✅ FastAPI + Uvicorn for serving predictions as a REST API  
✅ MLflow for model tracking and experiment management  
✅ Dockerized for easy containerization and deployment  
✅ CI/CD pipeline with GitHub Actions  
✅ Deployed live on Render  

## Tech Stack

- Python
- scikit-learn, XGBoost, LightGBM, CatBoost
- FastAPI, Uvicorn
- MLflow
- Docker
- GitHub Actions (CI/CD)
- Render (deployment)

## Project Structure

.
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
│   Dockerfile
│   .github/workflows/ci.yml
│
├───data
│   │   train.xlsx
│   └───models
│           model.pkl
│           preprocessor.pkl
│
├───app
│       api.py
│       mlflow_setup.py
│
└───helpers
        training.py
        utils.py

## Features Used for Prediction
_With example row_

| year: `int` | month: `int` | day: `int` | parcel_count: `int` | day_of_week: `str` | is_weekend: `str` | is_holiday: `int` | is_holiday_soon: `int` | fleet_available: `int` | total_parcel_weight: `float` | avg_parcel_weight: `float` | avg_parcel_volume_size: `float` |
|-------|-------|-----|--------------|--------------|-------------|------------|-----------------|----------------|--------------------|------------------|-----------------------|
| 2023 | 12 | 29 | 12671 | Friday | 0 | 1 | 0 | 124 | 5362.324 | 1.0276 | 0.12 |

Num of features: `12`

## Target Variable

| surge_level |
|-------------|

---
## Usage

Workflow file: .github/workflows/github-actions.yml
**Note:** For this program to work, you will need to load certain models and a preprocessor which are not included specifically in this repository. You may find them from the source project right [here](https://github.com/abdur-cader/surge-predictor/tree/main/data/models)
1. Download `xgb_model.pkl`, `cb_model.pkl`, `lgb_model.pkl`, and `preprocessor.pkl`.
2. Store them in a folder called `data/`, located at the root of this repository.

## Deployment
The app is deployed on Render.
