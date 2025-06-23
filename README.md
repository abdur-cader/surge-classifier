
# Surge Classifier

## Overview

This project is a variant of [Surge Predictor](https://github.com/abdur-cader/surge-predictor) — designed for deployment using Docker, MLflow, FastAPI and Render.

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

Workflow file: `.github/workflows/github-actions.yml`
- **Note:** For this program to work, you will need to load certain models and a preprocessor which are not included specifically in this repository. You may find them from the source project right [here](https://github.com/abdur-cader/surge-predictor/tree/main/data/models)
1. Download `xgb_model.pkl`, `cb_model.pkl`, `lgb_model.pkl`, and `preprocessor.pkl`.
2. Store them in a folder called `data/`, located at the root of this repository.
3. Certain lines have been commented out in the workflow file to prevent pipeline errors due to missing models. You can uncomment them once the above steps are complete.

To Run: 
```bash
docker-compose up --build
```

## Deployment
The app is deployed on Render.
