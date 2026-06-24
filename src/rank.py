#!/usr/bin/env python3
"""
Redrob Hackathon -- Intelligent Candidate Discovery & Ranking
Main entry point. Produces submission.csv from candidates.jsonl.

Usage:
    python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv
"""

import argparse
import time
import sys
from pathlib import Path

# Add the project root to sys.path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# To allow direct execution while finding modules:
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_candidates
from src.feature_engine import extract_features
from src.honeypot_detector import detect_honeypots
from src.behavioral_scorer import compute_behavioral_score
from src.model_scorer import load_model, score_candidates
from src.reasoning_generator import generate_reasoning

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob JD")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    args = parser.parse_args()
    
    pipeline_start = time.time()
    
    # PHASE 1: Load Data
    print("=" * 60)
    print("PHASE 1: Loading candidates...")
    t0 = time.time()
    
    df = load_candidates(args.candidates)
    print(f"  Loaded {len(df)} candidates in {time.time()-t0:.1f}s")
    
    # PHASE 2: Extract Features
    print("=" * 60)
    print("PHASE 2: Extracting features...")
    t0 = time.time()
    
    df = extract_features(df)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"  Extracted {len(feat_cols)} features in {time.time()-t0:.1f}s")
    
    # PHASE 3: Detect Honeypots
    print("=" * 60)
    print("PHASE 3: Detecting honeypots...")
    t0 = time.time()
    
    is_honeypot, suspicion_score = detect_honeypots(df)
    n_honeypots = is_honeypot.sum()
    print(f"  Flagged {n_honeypots} honeypots in {time.time()-t0:.1f}s")
    
    df['is_honeypot'] = is_honeypot
    df['suspicion_score'] = suspicion_score
    
    # PHASE 4: Compute Behavioral Score
    print("=" * 60)
    print("PHASE 4: Computing behavioral scores...")
    t0 = time.time()
    
    df['behavioral_score'] = compute_behavioral_score(df)
    print(f"  Behavioral scores computed in {time.time()-t0:.1f}s")
    print(f"  Mean: {df['behavioral_score'].mean():.3f}, "
          f"Std: {df['behavioral_score'].std():.3f}")
    
    # PHASE 5: XGBoost Scoring
    print("=" * 60)
    print("PHASE 5: Running XGBoost inference...")
    t0 = time.time()
    
    load_model()
    df['xgb_score'] = score_candidates(df)
    print(f"  XGBoost inference completed in {time.time()-t0:.1f}s")
    
    # PHASE 6: Compute Final Score
    print("=" * 60)
    print("PHASE 6: Computing final scores...")
    
    # The final score combines:
    # 1. XGBoost technical score (the primary signal)
    # 2. Behavioral multiplier (penalizes ghosts)
    # 3. Honeypot penalty (zeros out impossible profiles)
    
    honeypot_penalty = (~df['is_honeypot']).astype(float)
    # Also apply graduated penalty for suspicious (but not flagged) candidates
    honeypot_penalty = honeypot_penalty * (1.0 - df['suspicion_score'] * 0.5)
    
    df['final_score'] = (
        df['xgb_score'] * 
        df['behavioral_score'] * 
        honeypot_penalty
    ).round(4)
    
    # PHASE 7: Select Top 100 and Rank
    print("=" * 60)
    print("PHASE 7: Selecting top 100...")
    
    # Sort by final_score descending
    # Break ties by candidate_id ascending (as required by spec)
    df_sorted = df.sort_values(
        by=['final_score', 'candidate_id'],
        ascending=[False, True]
    )
    
    top100 = df_sorted.head(100).copy()
    top100['rank'] = range(1, 101)
    
    # Verify no honeypots in top 100
    honeypots_in_top100 = top100['is_honeypot'].sum()
    print(f"  Honeypots in top 100: {honeypots_in_top100}")
    if honeypots_in_top100 > 10:
        print("  WARNING: More than 10% honeypots -- risk of disqualification!")
    
    # PHASE 8: Generate Reasoning
    print("=" * 60)
    print("PHASE 8: Generating reasoning strings...")
    t0 = time.time()
    
    reasonings = []
    for _, row in top100.iterrows():
        reasoning = generate_reasoning(row, row['rank'], row['final_score'])
        reasonings.append(reasoning)
    
    top100['reasoning'] = reasonings
    print(f"  Generated {len(reasonings)} reasoning strings in {time.time()-t0:.1f}s")
    
    # PHASE 9: Write CSV
    print("=" * 60)
    print("PHASE 9: Writing submission CSV...")
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission = top100[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    submission.columns = ['candidate_id', 'rank', 'score', 'reasoning']
    
    # Ensure score is non-increasing with rank (required by spec)
    # This should already be true from sorting, but verify and fix if needed
    for i in range(1, len(submission)):
        if submission.iloc[i]['score'] > submission.iloc[i-1]['score']:
            submission.iloc[i, submission.columns.get_loc('score')] = (
                submission.iloc[i-1]['score']
            )
    
    # Format score to reasonable precision
    submission['score'] = submission['score'].round(4)
    
    # Ensure rank is integer
    submission['rank'] = submission['rank'].astype(int)
    
    # Write CSV with proper quoting for reasoning column
    submission.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL
    
    # Defensive Checks
    # Check 1: Exactly 100 rows
    assert len(submission) == 100
    
    # Check 2: Ranks are 1-100
    assert set(submission['rank']) == set(range(1, 101))
    
    # Check 3: No duplicate candidate_ids
    assert submission['candidate_id'].nunique() == 100
    
    # Check 4: Score is non-increasing
    scores = submission['score'].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Score increases at rank {i+1}"
    
    # Check 5: All candidate_ids exist in the original data
    valid_ids = set(df['candidate_id'])
    for cid in submission['candidate_id']:
        assert cid in valid_ids, f"Unknown candidate_id: {cid}"
        
    print("  All defensive checks passed!")
    
    # Summary
    total_time = time.time() - pipeline_start
    print("=" * 60)
    print(f"COMPLETE in {total_time:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Score range: {submission['score'].min():.4f} - {submission['score'].max():.4f}")
    print(f"  Honeypots in top 100: {honeypots_in_top100}")
    
    # Print top 5 for quick sanity check
    print(f"\nTop 5 candidates:")
    for _, row in submission.head(5).iterrows():
        print(f"  Rank {row['rank']}: {row['candidate_id']} "
              f"(score={row['score']:.4f}) -- {row['reasoning'][:80]}...")
    
    # Print bottom 5 for contrast
    print(f"\nBottom 5 candidates:")
    for _, row in submission.tail(5).iterrows():
        print(f"  Rank {row['rank']}: {row['candidate_id']} "
              f"(score={row['score']:.4f}) -- {row['reasoning'][:80]}...")
    
    if total_time > 270:  # 4.5 minutes
        print("\nWARNING: Approaching 5-minute time limit!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
