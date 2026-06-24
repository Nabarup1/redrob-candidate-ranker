#!/usr/bin/env python3
"""
Behavioral Viability Scorer module.
Computes a 'hireability index' from redrob_signals.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime

def compute_behavioral_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute hireability index from behavioral signals.
    
    Args:
        df: DataFrame containing the 'sig_' columns.
        
    Returns:
        Series of floats between 0.1 and 1.0
        0.1 = effectively unhirable (ghost candidate)
        1.0 = maximum engagement and availability
    """
    start_time = time.time()
    print("Computing behavioral scores...")
    
    # ---------------------------------------------------------
    # 1. RECENCY SCORE (25% weight)
    # ---------------------------------------------------------
    reference_date = datetime(2026, 6, 1)
    last_active = pd.to_datetime(df['sig_last_active_date'], errors='coerce').fillna(datetime(2020, 1, 1))
    days_since_active = (reference_date - last_active).dt.days
    
    # Clamp to >= 0 (in case of future dates)
    days_since_active = days_since_active.clip(lower=0)
    
    conditions = [
        days_since_active <= 7,
        days_since_active <= 30,
        days_since_active <= 60,
        days_since_active <= 90,
        days_since_active <= 180
    ]
    choices = [1.0, 0.9, 0.7, 0.5, 0.3]
    recency_score = pd.Series(np.select(conditions, choices, default=0.1), index=df.index)
    
    # ---------------------------------------------------------
    # 2. RESPONSIVENESS SCORE (30% weight)
    # ---------------------------------------------------------
    response_rate = df['sig_response_rate'].fillna(0.0)
    response_time = df['sig_avg_response_hours'].fillna(999.0)
    
    time_conditions = [
        response_time <= 12,
        response_time <= 48,
        response_time <= 96
    ]
    time_choices = [1.0, 0.9, 0.7]
    time_penalty = pd.Series(np.select(time_conditions, time_choices, default=0.5), index=df.index)
    
    responsiveness_score = (response_rate * 0.7) + (response_rate * time_penalty * 0.3)
    
    # ---------------------------------------------------------
    # 3. ENGAGEMENT SCORE (20% weight)
    # ---------------------------------------------------------
    open_to_work = df['sig_open_to_work'].fillna(False).astype(float)
    is_applying = (df['sig_applications_30d'].fillna(0) > 0).astype(float)
    profile_complete = df['sig_profile_completeness'].fillna(0.0) / 100.0
    recruiter_interest = (df['sig_saved_recruiters_30d'].fillna(0) / 10.0).clip(upper=1.0)
    
    engagement_score = (
        open_to_work * 0.35 +
        is_applying * 0.20 +
        profile_complete * 0.25 +
        recruiter_interest * 0.20
    )
    
    # ---------------------------------------------------------
    # 4. RELIABILITY SCORE (15% weight)
    # ---------------------------------------------------------
    interview_rate = df['sig_interview_completion'].fillna(0.5)
    # Replace -1 with 0.5 (neutral) for offer acceptance
    offer_rate = df['sig_offer_acceptance'].replace(-1, 0.5).fillna(0.5)
    
    reliability_score = (interview_rate * 0.6) + (offer_rate * 0.4)
    
    # ---------------------------------------------------------
    # 5. VERIFICATION SCORE (10% weight)
    # ---------------------------------------------------------
    verified_email = df['sig_verified_email'].fillna(False).astype(float)
    verified_phone = df['sig_verified_phone'].fillna(False).astype(float)
    linkedin = df['sig_linkedin_connected'].fillna(False).astype(float)
    
    verification_score = (verified_email * 0.4) + (verified_phone * 0.3) + (linkedin * 0.3)
    
    # COMPUTE FINAL SCORE
    behavioral_score = (
        recency_score * 0.25 +
        responsiveness_score * 0.30 +
        engagement_score * 0.20 +
        reliability_score * 0.15 +
        verification_score * 0.10
    )
    
    behavioral_score = behavioral_score.clip(lower=0.1, upper=1.0)
    
    elapsed = time.time() - start_time
    print(f"Computed behavioral scores for {len(df)} rows in {elapsed:.3f}s")
    
    return behavioral_score

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add the project root to sys.path so 'src' module can be found
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from src.data_loader import load_candidates
        from src import config
        
        df_raw = load_candidates(config.CANDIDATES_FILE)
        
        scores = compute_behavioral_score(df_raw)
        
        df_raw['behavioral_score'] = scores
        
        print("\nScore Statistics:")
        print(df_raw['behavioral_score'].describe())
        
        print("\nTop 5 Candidates by Behavioral Score:")
        best = df_raw.sort_values('behavioral_score', ascending=False).head(5)
        for _, row in best.iterrows():
            print(f"  ID: {row['candidate_id']} | Score: {row['behavioral_score']:.3f} | Last Active: {row['sig_last_active_date']}")
            
        print("\nBottom 5 Candidates by Behavioral Score:")
        worst = df_raw.sort_values('behavioral_score', ascending=True).head(5)
        for _, row in worst.iterrows():
            print(f"  ID: {row['candidate_id']} | Score: {row['behavioral_score']:.3f} | Last Active: {row['sig_last_active_date']}")
            
    except Exception as e:
        print(f"Error running standalone test: {e}")
