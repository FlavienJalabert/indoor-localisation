"""Tabular models and preprocessing pipelines."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Build the ColumnTransformer for numeric and categorical features."""

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ]
    )


def train_knn(
    X_train: pd.DataFrame, y_train: np.ndarray, preprocessor, *, knn_params: dict
) -> Pipeline:
    """Train KNN regressor pipeline."""

    model = KNeighborsRegressor(**knn_params)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_rf(
    X_train: pd.DataFrame, y_train: np.ndarray, preprocessor, *, rf_params: dict
) -> Pipeline:
    """Train RandomForest regressor pipeline."""

    model = RandomForestRegressor(**rf_params)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgb_xy(
    X_train: pd.DataFrame, y_train: np.ndarray, preprocessor, *, xgb_params: dict
) -> Tuple[Pipeline, Pipeline]:
    """Train two XGBoost regressors for X and Y separately."""

    xgb_x = XGBRegressor(**xgb_params)
    xgb_y = XGBRegressor(**xgb_params)

    pipeline_x = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", xgb_x)])
    pipeline_y = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", xgb_y)])

    pipeline_x.fit(X_train, y_train[:, 0])
    pipeline_y.fit(X_train, y_train[:, 1])

    return pipeline_x, pipeline_y


def predict_xy_from_xgb(xgb_x: Pipeline, xgb_y: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict (X,Y) from two trained XGBoost pipelines."""

    pred_x = xgb_x.predict(X)
    pred_y = xgb_y.predict(X)
    return np.column_stack([pred_x, pred_y])
