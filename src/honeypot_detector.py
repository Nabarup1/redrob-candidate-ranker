#!/usr/bin/env python3
"""
Honeypot detector module.
Identifies trap candidates with logically impossible profiles.
"""

import time
import pandas as pd
import numpy as np

def detect_honeypots(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Detect honeypot candidates with logically impossible profiles.
    
    Args:
        df: The DataFrame with all features extracted
        
    Returns:
        is_honeypot: boolean Series (True = flagged as honeypot)
        suspicion_score: float Series (0.0 to 1.0, higher = more suspicious)
    """
    start_time = time.time()
    print("Running honeypot detection...")
    
    # ---------------------------------------------------------
    # RULE 1: Experience vs Education Timeline Mismatch (+0.4)
    # ---------------------------------------------------------
    # e.g., 15 years experience but graduated in 2018 (max exp = 2026 - 2018 = 8 years)
    max_possible_exp = 2026 - df['edu_latest_end_year']
    rule_1 = (df['edu_latest_end_year'] > 0) & (df['profile_years_of_exp'] > max_possible_exp + 2)
    
    # ---------------------------------------------------------
    # RULE 2: Career Duration vs Years of Experience Mismatch (+0.3)
    # ---------------------------------------------------------
    career_months = df['career_total_months']
    exp_months = df['profile_years_of_exp'] * 12
    # Flag if career total is > 150% of claimed experience, or < 30% of claimed experience
    rule_2 = (career_months > exp_months * 1.5) | ((exp_months > 0) & (career_months < exp_months * 0.3))
    
    # ---------------------------------------------------------
    # RULE 3: Skill Proficiency Without Substance (+0.5)
    # ---------------------------------------------------------
    # 'expert' in 8+ skills but 0 endorsements. We approximate zero_endorsement_advanced
    # using skills_total_endorsements == 0.
    rule_3 = (df['skills_advanced_count'] >= 8) & (df['skills_total_endorsements'] <= 2)
    
    # ---------------------------------------------------------
    # RULE 4: Title / Description Mismatch (+0.3)
    # ---------------------------------------------------------
    # Marketing Manager title but heavy production ML keywords in description
    if 'feat_dim1_is_non_eng_title' in df.columns:
        rule_4 = (df['feat_dim1_is_non_eng_title'] == 1) & (
            (df['feat_dim2_production_kw_count'] >= 2) | 
            (df['feat_dim3_retrieval_desc_count'] >= 2)
        )
    else:
        rule_4 = pd.Series(False, index=df.index)
        
    # ---------------------------------------------------------
    # RULE 5: Salary Range Inversion (+0.2)
    # ---------------------------------------------------------
    # min_salary > max_salary
    rule_5 = (df['sig_salary_min'] > df['sig_salary_max']) & (df['sig_salary_max'] > 0)
    
    # ---------------------------------------------------------
    # RULE 6: Signup Date > Last Active Date (+0.3)
    # ---------------------------------------------------------
    signup = pd.to_datetime(df['sig_signup_date'], errors='coerce')
    last_active = pd.to_datetime(df['sig_last_active_date'], errors='coerce')
    rule_6 = (signup > last_active) & signup.notnull() & last_active.notnull()
    
    # ---------------------------------------------------------
    # RULE 7: Impossible Assessment Scores (+0.2)
    # ---------------------------------------------------------
    def check_assessments(row):
        scores = row.get('sig_assessment_scores', {})
        if not isinstance(scores, dict): return 0
        skill_names = row.get('skills_names', '')
        if not skill_names: return 0
        
        # Check if they have scores for skills they don't even claim
        names = set(skill_names.split('|'))
        for k in scores.keys():
            if k.lower() not in names:
                return 1
                
        # Check if they claim many advanced skills but bombed all assessments
        if row.get('skills_advanced_count', 0) >= 3:
            if scores and max(scores.values()) < 30:
                return 1
        return 0
        
    rule_7 = df.apply(check_assessments, axis=1) == 1
    
    # ---------------------------------------------------------
    # RULE 8: Career History Anomalies (+0.2)
    # ---------------------------------------------------------
    # Impossible average tenure (e.g., avg tenure > 20 years but they are young)
    rule_8 = (df['career_avg_tenure_months'] > 240) & (df['profile_years_of_exp'] < 10)
    
    # COMPUTE COMPOSITE SCORE
    suspicion_score = (
        rule_1.astype(float) * 0.4 +
        rule_2.astype(float) * 0.3 +
        rule_3.astype(float) * 0.5 +
        rule_4.astype(float) * 0.3 +
        rule_5.astype(float) * 0.2 +
        rule_6.astype(float) * 0.3 +
        rule_7.astype(float) * 0.2 +
        rule_8.astype(float) * 0.2
    )
    
    # Clip to max 1.0
    suspicion_score = suspicion_score.clip(upper=1.0)
    
    is_honeypot = suspicion_score > 0.7
    
    # Logging
    elapsed = time.time() - start_time
    total_honeypots = is_honeypot.sum()
    print(f"Honeypot detection complete in {elapsed:.2f}s")
    print(f"Detected {total_honeypots} honeypots out of {len(df)} candidates.")
    
    print("\nBreakdown of rule violations:")
    print(f"  Rule 1 (Exp/Edu mismatch):     {rule_1.sum()}")
    print(f"  Rule 2 (Career/Exp mismatch):  {rule_2.sum()}")
    print(f"  Rule 3 (Skill/Endorsement):    {rule_3.sum()}")
    print(f"  Rule 4 (Title/Desc mismatch):  {rule_4.sum()}")
    print(f"  Rule 5 (Salary inversion):     {rule_5.sum()}")
    print(f"  Rule 6 (Time travel login):    {rule_6.sum()}")
    print(f"  Rule 7 (Fake assessments):     {rule_7.sum()}")
    print(f"  Rule 8 (Tenure anomaly):       {rule_8.sum()}")
    
    return is_honeypot, suspicion_score

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add the project root to sys.path so 'src' module can be found
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from src.data_loader import load_candidates
        from src.feature_engine import extract_features
        from src import config
        
        df_raw = load_candidates(config.CANDIDATES_FILE)
        df_feat = extract_features(df_raw)
        
        is_hp, scores = detect_honeypots(df_feat)
        
        df_feat['is_honeypot'] = is_hp
        df_feat['suspicion_score'] = scores
        
        print(f"\nTop 5 Suspicious candidates (Score > 0):")
        suspicious_df = df_feat[df_feat['suspicion_score'] > 0].sort_values('suspicion_score', ascending=False)
        for _, row in suspicious_df.head(5).iterrows():
            print(f"  ID: {row['candidate_id']} | Score: {row['suspicion_score']:.2f} | Title: {row['profile_current_title']}")
            
    except Exception as e:
        print(f"Error running standalone test: {e}")
