#!/usr/bin/env python3
"""
Validation & Quality Assurance
Comprehensive validation script to ensure submission is format-correct and quality-correct.
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path

# Add the project root to sys.path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data_loader import load_candidates
from src.honeypot_detector import detect_honeypots


def run_official_validator(csv_path):
    """Run the provided validate_submission.py script."""
    validator_script = "data/validate_submission.py"
    if not os.path.exists(validator_script):
        print(f"[WARN] Official validator script not found at {validator_script}. Skipping.")
        return True
        
    result = subprocess.run(
        [sys.executable, validator_script, csv_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[PASS] Official validator: Submission is valid")
    else:
        print(f"[FAIL] Official validator:\n{result.stdout}\n{result.stderr}")
    return result.returncode == 0


def audit_honeypots(submission_df, candidates_df):
    """Check if any honeypots made it into the top 100."""
    is_honeypot, _ = detect_honeypots(candidates_df)
    honeypot_ids = set(candidates_df[is_honeypot]['candidate_id'])
    
    top100_ids = set(submission_df['candidate_id'])
    honeypots_in_top100 = top100_ids & honeypot_ids
    
    if not honeypots_in_top100:
        print("[PASS] Honeypot audit: No honeypots in top 100")
    else:
        print(f"[FAIL] Honeypot audit: {len(honeypots_in_top100)} honeypots found!")
        for hp_id in honeypots_in_top100:
            rank = submission_df[submission_df['candidate_id'] == hp_id]['rank'].iloc[0]
            print(f"  - {hp_id} at rank {rank}")
    
    return len(honeypots_in_top100) == 0


def audit_titles(submission_df, candidates_df):
    """Check that top-ranked candidates have relevant titles."""
    top10 = submission_df[submission_df['rank'] <= 10]
    top10_details = top10.merge(
        candidates_df[['candidate_id', 'profile_current_title', 'profile_current_industry']],
        on='candidate_id'
    )
    
    non_eng_titles = {
        "marketing manager", "operations manager", "hr manager",
        "sales executive", "accountant", "content writer",
        "graphic designer", "customer support", "civil engineer",
        "mechanical engineer",
    }
    
    suspicious_count = 0
    print("\n--- Top 10 Title Audit ---")
    for _, row in top10_details.sort_values('rank').iterrows():
        title = str(row['profile_current_title']).lower()
        flag = "[!!]" if title in non_eng_titles else "[OK]"
        if title in non_eng_titles:
            suspicious_count += 1
        print(f"  Rank {row['rank']:3d}: {flag} {row['profile_current_title']} "
              f"({row['profile_current_industry']})")
    
    if suspicious_count == 0:
        print("\n[PASS] Title audit: All top-10 titles are engineering-related")
    else:
        print(f"\n[WARN] Title audit: {suspicious_count} non-engineering titles in top 10")
    
    return suspicious_count == 0


def audit_reasoning(submission_df):
    """Check reasoning strings for quality issues."""
    issues = []
    
    # Check 1: No empty reasonings
    empty = submission_df['reasoning'].isna() | (submission_df['reasoning'] == "")
    if empty.any():
        issues.append(f"Empty reasoning at ranks: {submission_df[empty]['rank'].tolist()}")
    
    # Check 2: No duplicate reasonings
    dupes = submission_df['reasoning'].duplicated()
    if dupes.any():
        issues.append(f"Duplicate reasonings at ranks: {submission_df[dupes]['rank'].tolist()}")
    
    # Check 3: No very short reasonings (<30 chars)
    short = submission_df['reasoning'].str.len() < 30
    if short.any():
        issues.append(f"Very short reasoning at ranks: {submission_df[short]['rank'].tolist()}")
    
    # Check 4: Sample 10 random reasonings and print them for manual review
    sample = submission_df.sample(min(10, len(submission_df)), random_state=42)
    print("\n--- Sampled Reasonings for Manual Review ---")
    for _, row in sample.sort_values('rank').iterrows():
        print(f"  Rank {row['rank']:3d} (score {row['score']:.4f}):")
        print(f"    {row['reasoning']}")
        print()
    
    if not issues:
        print("[PASS] Reasoning quality: No automated issues detected")
    else:
        for issue in issues:
            print(f"[FAIL] Reasoning quality: {issue}")
    
    return len(issues) == 0


def audit_score_distribution(submission_df):
    """Check that scores have reasonable distribution."""
    scores = submission_df['score']
    
    print(f"\n--- Score Distribution ---")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Median: {scores.median():.4f}")
    
    # Check that scores are not all the same
    if scores.nunique() < 10:
        print("[FAIL] Score distribution: Too few unique scores (model not differentiating)")
        return False
    
    # Check score spread (rank 1 should be significantly higher than rank 100)
    spread = scores.iloc[0] - scores.iloc[-1]
    if spread < 0.01:
        print("[WARN] Score distribution: Very narrow spread between rank 1 and 100")
    else:
        print(f"\n[PASS] Score spread: {spread:.4f} (rank 1 vs rank 100)")
    
    return True


def audit_candidate_ids(submission_df, candidates_path):
    """Verify all submitted candidate_ids exist in the original dataset."""
    valid_ids = set()
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                valid_ids.add(record['candidate_id'])
    
    submitted_ids = set(submission_df['candidate_id'])
    invalid = submitted_ids - valid_ids
    
    if not invalid:
        print("[PASS] Candidate ID audit: All IDs exist in candidates.jsonl")
    else:
        print(f"[FAIL] Candidate ID audit: {len(invalid)} invalid IDs: {invalid}")
    
    return len(invalid) == 0


def main():
    csv_path = "output/submission.csv"
    candidates_path = "data/candidates.jsonl"
    
    print("=" * 60)
    print("SUBMISSION VALIDATION & QA REPORT")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please run src/rank.py first.")
        return 1
        
    if not os.path.exists(candidates_path):
        print(f"Error: Could not find {candidates_path}.")
        return 1
    
    # Load submission and candidates
    submission = pd.read_csv(csv_path)
    print("Loading full candidates dataset for validation...")
    candidates_df = load_candidates(candidates_path)
    
    results = []
    
    # Run all audit layers
    print("\n[Layer 1] Format")
    results.append(("Format", run_official_validator(csv_path)))
    
    print("\n[Layer 2] Honeypots")
    results.append(("Honeypots", audit_honeypots(submission, candidates_df)))
    
    print("\n[Layer 3] Titles")
    results.append(("Titles", audit_titles(submission, candidates_df)))
    
    print("\n[Layer 4] Reasoning")
    results.append(("Reasoning", audit_reasoning(submission)))
    
    print("\n[Layer 5] Scores")
    results.append(("Scores", audit_score_distribution(submission)))
    
    print("\n[Layer 6] IDs")
    results.append(("IDs", audit_candidate_ids(submission, candidates_path)))
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} [{status}]")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\nAll checks passed. Submission is perfectly valid and high quality!")
        print("You are ready to upload output/submission.csv!")
    else:
        print("\nSome checks failed. Please review the issues above.")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
