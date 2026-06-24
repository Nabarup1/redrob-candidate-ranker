import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys_path = str(Path(__file__).parent.parent)
if sys_path not in sys.path:
    sys.path.append(sys_path)

from src import config

_MODEL = None
_FEATURE_COLS = None

def load_model(model_path: str = None, config_path: str = None):
    """
    Load the pre-trained XGBoost model and feature configuration.
    Caches the model in a module-level variable to avoid reloading.
    
    This function is called once at the start of the ranking pipeline.
    """
    global _MODEL, _FEATURE_COLS
    
    if model_path is None:
        model_path = str(Path(config.MODELS_DIR) / "xgboost_model.pkl")
    if config_path is None:
        config_path = str(Path(config.MODELS_DIR) / "feature_config.json")
        
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run scripts/train_model.py first.")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please run scripts/train_model.py first.")
        
    _MODEL = joblib.load(model_path)
    with open(config_path, 'r') as f:
        conf = json.load(f)
    _FEATURE_COLS = conf['feature_columns']
    
    print(f"Loaded XGBoost model with {conf['n_features']} features")
    print(f"Training CV RMSE: {conf.get('cv_rmse', 'N/A')}")


def score_candidates(df: pd.DataFrame) -> pd.Series:
    """
    Score all candidates using the pre-trained XGBoost model.
    
    Args:
        df: DataFrame with all feat_* columns from feature_engine
        
    Returns:
        Series of float scores (higher = better fit for the JD)
        
    Raises:
        RuntimeError: If model hasn't been loaded yet
        KeyError: If required feature columns are missing from df
    """
    if _MODEL is None or _FEATURE_COLS is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Verify all required feature columns exist
    missing = set(_FEATURE_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    
    if df[_FEATURE_COLS].isna().all(axis=1).any():
        print("Warning: Some candidates have NaN in all features.")
        
    # Extract features in the correct column order
    X = df[_FEATURE_COLS].values.astype(np.float32)
    
    # Handle NaN values the same way as training
    X = np.nan_to_num(X, nan=-1.0)
    
    # Run inference
    raw_scores = _MODEL.predict(X)
    
    # The raw scores are on the 0-5 scale (matching LLM labels)
    # Normalize to 0-1 for easier downstream combination
    normalized = np.clip(raw_scores / 5.0, 0.0, 1.0)
    
    return pd.Series(normalized, index=df.index, name='xgb_score')

# Quick self-test
if __name__ == "__main__":
    from src.data_loader import load_candidates
    from src.feature_engine import extract_features
    import time
    
    print("Loading data...")
    df = load_candidates(config.CANDIDATES_FILE)
    print("Extracting features...")
    df = extract_features(df)
    
    t0 = time.time()
    load_model()
    t1 = time.time()
    print(f"Model load time: {(t1-t0)*1000:.2f} ms")
    
    t2 = time.time()
    scores = score_candidates(df)
    t3 = time.time()
    print(f"Inference time for {len(df)} rows: {(t3-t2)*1000:.2f} ms\n")
    
    print(f"Score distribution:")
    print(f"  min:  {scores.min():.4f}")
    print(f"  max:  {scores.max():.4f}")
    print(f"  mean: {scores.mean():.4f}")
    print(f"  std:  {scores.std():.4f}")
    
    # Print top 10 by score
    top10 = df.loc[scores.nlargest(10).index, ['candidate_id', 'profile_current_title']]
    top10['score'] = scores.nlargest(10).values
    print(f"\nTop 10 candidates by XGBoost score:")
    print(top10.to_string(index=False))
    
    # Print bottom 5 by score
    bottom5 = df.loc[scores.nsmallest(5).index, ['candidate_id', 'profile_current_title']]
    bottom5['score'] = scores.nsmallest(5).values
    print(f"\nBottom 5 candidates by XGBoost score:")
    print(bottom5.to_string(index=False))
