#!/usr/bin/env python3
"""
XGBoost Training Script.
Trains a regression model to predict LLM-assigned relevance scores from the extracted features.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys_path = str(Path(__file__).parent.parent)
if sys_path not in sys.path:
    sys.path.append(sys_path)

from src import config
from src.data_loader import load_candidates
from src.feature_engine import extract_features

def main():
    print("Loading all candidates...")
    df_all = load_candidates(config.CANDIDATES_FILE)

    labels_path = Path(config.MODELS_DIR) / "llm_labels.csv"
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return

    print("Loading LLM labels...")
    labels = pd.read_csv(labels_path)

    print("Extracting features for ALL candidates...")
    # Extract features for all candidates to ensure consistent pipeline
    df_features = extract_features(df_all)

    # Join labels onto features
    df_train = df_features.merge(labels[['candidate_id', 'llm_score']], on='candidate_id', how='inner')

    print(f"\nTraining set size: {len(df_train)}")
    print(f"Score distribution:\n{df_train['llm_score'].value_counts().sort_index()}")

    # Select feature columns
    feature_cols = [col for col in df_train.columns if col.startswith('feat_')]

    # Remove constant or all-null features
    for col in feature_cols[:]:
        if df_train[col].nunique() <= 1 or df_train[col].isna().all():
            feature_cols.remove(col)
            print(f"  Dropped constant/null feature: {col}")

    print(f"Using {len(feature_cols)} features for training")

    # Prepare data
    X = df_train[feature_cols].values.astype(np.float32)
    y = df_train['llm_score'].values.astype(np.float32)

    # Handle NaNs
    X = np.nan_to_num(X, nan=-1.0)

    print(f"X shape: {X.shape}")
    print(f"y range: [{y.min()}, {y.max()}], mean: {y.mean():.2f}\n")

    # Cross validation
    print("Starting 5-fold cross validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        cv_scores.append({"fold": fold, "rmse": float(rmse), "mae": float(mae)})
        print(f"  Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    avg_rmse = np.mean([s['rmse'] for s in cv_scores])
    avg_mae = np.mean([s['mae'] for s in cv_scores])
    print(f"\nAverage CV RMSE: {avg_rmse:.4f}")
    print(f"Average CV MAE:  {avg_mae:.4f}")

    # Train final model
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importances
    importances = final_model.feature_importances_
    importance_pairs = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 20 Most Important Features:")
    for feat, imp in importance_pairs[:20]:
        print(f"  {feat:45s} {imp:.4f}")

    print("\nBottom 10 Least Important Features:")
    for feat, imp in importance_pairs[-10:]:
        print(f"  {feat:45s} {imp:.4f}")

    # Save model and config
    out_model_path = Path(config.MODELS_DIR) / "xgboost_model.pkl"
    out_config_path = Path(config.MODELS_DIR) / "feature_config.json"
    out_report_path = Path(config.MODELS_DIR) / "training_report.json"

    joblib.dump(final_model, out_model_path)
    print(f"\nModel saved to {out_model_path}")

    with open(out_config_path, "w") as f:
        json.dump({
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "training_samples": len(X),
            "cv_rmse": float(avg_rmse),
            "cv_mae": float(avg_mae),
        }, f, indent=2)
    print(f"Feature config saved to {out_config_path}")

    report = {
        "cv_scores": cv_scores,
        "feature_importances": [
            {"feature": f, "importance": float(i)} 
            for f, i in importance_pairs
        ],
        "hyperparameters": final_model.get_params(),
        "training_samples": len(X),
        "score_distribution": {str(k): int(v) for k, v in df_train['llm_score'].value_counts().to_dict().items()},
    }
    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Sanity check
    print("\nSanity Check (Predictions on Training Set):")
    y_pred_all = final_model.predict(X)
    for true_score in range(6):
        mask = y == true_score
        if mask.sum() == 0:
            continue
        preds = y_pred_all[mask]
        print(f"True score {true_score}: predicted mean={preds.mean():.2f}, "
              f"min={preds.min():.2f}, max={preds.max():.2f}, n={mask.sum()}")

if __name__ == "__main__":
    main()
