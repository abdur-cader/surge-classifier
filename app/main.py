from fastapi import FastAPI, HTTPException, Query
import joblib
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


app = FastAPI()

xgb = joblib.load("data/xgb_model.pkl")
cb = joblib.load("data/cb_model.pkl")
lgb = joblib.load("data/lgb_model.pkl")
preprocessor = joblib.load("data/preprocessor.pkl")
X_test = joblib.load('data/x_test.pkl')
y_test = joblib.load('data/y_test.pkl')

@app.get("/")
def home():
    return {"message": "Surge Prediction API"}

@app.post("/predict")
def predict(model_choice: str, features: list[float]):

    if model_choice in ["xgboost", "xgb"]:
        model = xgb
    elif model_choice in ["cb", "catboost"]:
        model = cb
    elif model_choice in ["lgb", "lightgbm"]:
        model = lgb
    else:
        model = xgb

    actual_len = len(features)
    expected_len = preprocessor.n_features_in_
    if len(features) != expected_len:
        raise HTTPException(
            status_code=400,
            detail = f"Expected {expected_len} features, found {actual_len}"
        )
    
    input_data = np.array(features).reshape(1, -1)
    processed = preprocessor.transform(input_data)
    pred = model.predict(processed).tolist()
    return pred


@app.get("/score")
def score(
    model_choice: str,
    acc_flag: bool = False,
    f1_flag: bool = False,
    recall_flag: bool = False,
    precision_flag: bool = False,
    conf_flag: bool = False,
    roc_flag: bool = True
):
    
    if model_choice in ["xgboost", "xgb"]:
        model = xgb
    elif model_choice in ["cb", "catboost"]:
        model = cb
    elif model_choice in ["lgb", "lightgbm"]:
        model = lgb
    else:
        model = xgb

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(X_test.shape)
    return_var = {}

    input_sample = X_test[:5]
    signature = infer_signature(
        input_sample,
        model.predict(input_sample)
    )

    mlflow.set_experiment("default")
    with mlflow.start_run(run_name=f"{model_choice}_score_run", nested=True):
        mlflow.set_tag("model", model_choice)

        if roc_flag:
            roc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            return_var["roc_auc"] = roc
            mlflow.log_metric("roc_auc", roc)

        if acc_flag:
            acc = accuracy_score(y_test, y_pred)
            return_var["accuracy"] = acc
            mlflow.log_metric("accuracy", acc)

        if f1_flag:
            f1 = f1_score(y_test, y_pred, average="weighted")
            return_var["f1"] = f1
            mlflow.log_metric("f1", f1)

        if recall_flag:
            recall = recall_score(y_test, y_pred, average="weighted")
            return_var["recall"] = recall
            mlflow.log_metric("recall", recall)

        if precision_flag:
            precision = precision_score(y_test, y_pred, average="weighted")
            return_var["precision"] = precision
            mlflow.log_metric("precision", precision)

        if conf_flag:
            conf = confusion_matrix(y_test, y_pred)
            return_var["confusion_matrix"] = conf.tolist()

        if model_choice in ["xgboost", "xgb"]:
            mlflow.xgboost.log_model(model, name=f"{model_choice}_model", input_example=input_sample, signature=signature)
        elif model_choice in ["catboost", "cb"]:
            mlflow.catboost.log_model(model, name=f"{model_choice}_model", input_example=input_sample, signature=signature)
        elif model_choice in ["lightgbm", "lgb"]:
            mlflow.lightgbm.log_model(model, name=f"{model_choice}_model", input_example=input_sample, signature=signature)

    return return_var

    