#!/usr/bin/env python3
"""
Data loader for the Redrob candidate dataset.
Reads JSONL, flattens nested structures, returns a pandas DataFrame.
"""

import gzip
import json
import time
from pathlib import Path
import pandas as pd

DEGREE_RANK = {
    "ph.d": 5, "phd": 5,
    "m.tech": 4, "m.s.": 4, "m.sc": 4, "m.e.": 4, "mba": 4,
    "b.tech": 3, "b.e.": 3, "b.sc": 3,
    "diploma": 2,
}

TIER_RANK = {
    "tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0,
}

def flatten_career(career_list):
    """Extract aggregate career features from the career_history array."""
    if not career_list:
        return {
            "career_all_titles": "",
            "career_all_companies": "",
            "career_all_industries": "",
            "career_all_descriptions": "",
            "career_num_jobs": 0,
            "career_total_months": 0,
            "career_avg_tenure_months": 0.0,
            "career_current_company": "",
            "career_current_title": ""
        }
        
    titles = []
    companies = []
    industries = []
    descriptions = []
    total_months = 0
    current_company = ""
    current_title = ""
    
    for job in career_list:
        titles.append(job.get("title", ""))
        companies.append(job.get("company", ""))
        industries.append(job.get("industry", ""))
        descriptions.append(job.get("description", ""))
        total_months += job.get("duration_months", 0)
        
        if job.get("is_current"):
            current_company = job.get("company", "")
            current_title = job.get("title", "")
    
    return {
        "career_all_titles": "|".join(titles),
        "career_all_companies": "|".join(companies),
        "career_all_industries": "|".join(industries),
        "career_all_descriptions": "\n".join(descriptions),
        "career_num_jobs": len(career_list),
        "career_total_months": total_months,
        "career_avg_tenure_months": total_months / max(len(career_list), 1),
        "career_current_company": current_company,
        "career_current_title": current_title
    }

def flatten_skills(skills_list):
    """Extract aggregate skills features."""
    if not skills_list:
        return {
            "skills_names": "",
            "skills_count": 0,
            "skills_advanced_count": 0,
            "skills_total_endorsements": 0,
            "skills_avg_duration": 0.0
        }
        
    names = []
    advanced_count = 0
    total_endorsements = 0
    total_duration = 0
    
    for s in skills_list:
        names.append(s.get("name", "").lower())
        prof = s.get("proficiency", "").lower()
        if prof in ("advanced", "expert"):
            advanced_count += 1
        total_endorsements += s.get("endorsements", 0)
        total_duration += s.get("duration_months", 0)
        
    return {
        "skills_names": "|".join(names),
        "skills_count": len(skills_list),
        "skills_advanced_count": advanced_count,
        "skills_total_endorsements": total_endorsements,
        "skills_avg_duration": total_duration / max(len(skills_list), 1)
    }

def flatten_education(edu_list):
    """Extract aggregate education features."""
    if not edu_list:
        return {
            "edu_highest_degree": "",
            "edu_best_tier": "",
            "edu_fields": "",
            "edu_latest_end_year": 0
        }
        
    highest_degree_score = -1
    highest_degree = ""
    
    best_tier_score = -1
    best_tier = ""
    
    fields = []
    latest_end = 0
    
    for e in edu_list:
        deg = e.get("degree", "").lower()
        deg_score = DEGREE_RANK.get(deg, 1)
        if deg_score > highest_degree_score:
            highest_degree_score = deg_score
            highest_degree = e.get("degree", "")
            
        tier = e.get("tier", "unknown").lower()
        tier_score = TIER_RANK.get(tier, 0)
        if tier_score > best_tier_score:
            best_tier_score = tier_score
            best_tier = e.get("tier", "unknown")
            
        fields.append(e.get("field_of_study", ""))
        
        ey = e.get("end_year", 0)
        if ey > latest_end:
            latest_end = ey
            
    return {
        "edu_highest_degree": highest_degree,
        "edu_best_tier": best_tier,
        "edu_fields": "|".join(fields),
        "edu_latest_end_year": latest_end
    }

def flatten_candidate(record: dict) -> dict:
    """Convert a nested candidate JSON into a flat dictionary."""
    flat = {"candidate_id": record.get("candidate_id", "")}
    
    # Extract profile fields
    prof = record.get("profile", {})
    flat.update({
        "profile_headline": prof.get("headline", ""),
        "profile_summary": prof.get("summary", ""),
        "profile_location": prof.get("location", ""),
        "profile_country": prof.get("country", ""),
        "profile_years_of_exp": float(prof.get("years_of_experience", 0)),
        "profile_current_title": prof.get("current_title", ""),
        "profile_current_company": prof.get("current_company", ""),
        "profile_company_size": prof.get("company_size", ""),
        "profile_current_industry": prof.get("current_industry", "")
    })
    
    # Flatten career history
    flat.update(flatten_career(record.get("career_history", [])))
    
    # Flatten skills
    flat.update(flatten_skills(record.get("skills", [])))
    
    # Flatten education
    flat.update(flatten_education(record.get("education", [])))
    
    # Extract redrob signals
    redrob = record.get("redrob_signals", {})
    flat.update({
        "sig_profile_completeness": float(redrob.get("profile_completeness_score", 0)),
        "sig_signup_date": redrob.get("signup_date", ""),
        "sig_last_active_date": redrob.get("last_active_date", ""),
        "sig_open_to_work": bool(redrob.get("open_to_work_flag", False)),
        "sig_profile_views_30d": int(redrob.get("profile_views_received_30d", 0)),
        "sig_applications_30d": int(redrob.get("applications_submitted_30d", 0)),
        "sig_response_rate": float(redrob.get("recruiter_response_rate", 0)),
        "sig_avg_response_hours": float(redrob.get("avg_response_time_hours", 0)),
        "sig_assessment_scores": redrob.get("skill_assessment_scores", {}),
        "sig_connection_count": int(redrob.get("connection_count", 0)),
        "sig_endorsements": int(redrob.get("endorsements_received", 0)),
        "sig_notice_period_days": int(redrob.get("notice_period_days", 0)),
        "sig_salary_min": float(redrob.get("expected_salary_range_inr_lpa", {}).get("min", 0)),
        "sig_salary_max": float(redrob.get("expected_salary_range_inr_lpa", {}).get("max", 0)),
        "sig_work_mode": redrob.get("preferred_work_mode", ""),
        "sig_willing_relocate": bool(redrob.get("willing_to_relocate", False)),
        "sig_github_score": float(redrob.get("github_activity_score", 0)),
        "sig_search_appear_30d": int(redrob.get("search_appearance_30d", 0)),
        "sig_saved_recruiters_30d": int(redrob.get("saved_by_recruiters_30d", 0)),
        "sig_interview_completion": float(redrob.get("interview_completion_rate", 0)),
        "sig_offer_acceptance": float(redrob.get("offer_acceptance_rate", 0)),
        "sig_verified_email": bool(redrob.get("verified_email", False)),
        "sig_verified_phone": bool(redrob.get("verified_phone", False)),
        "sig_linkedin_connected": bool(redrob.get("linkedin_connected", False))
    })
    
    return flat

def load_candidates(filepath) -> pd.DataFrame:
    """
    Load and flatten all candidates from JSONL file into a pandas DataFrame.
    
    Args:
        filepath: Path to the candidates file (.jsonl or .jsonl.gz)
        
    Returns:
        DataFrame with one row per candidate
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find data file at {filepath}")
        
    rows = []
    skipped = 0
    start_time = time.time()
    
    print(f"Loading candidates from {filepath}...")
    
    # determine opener based on extension
    if str(filepath).endswith(".gz"):
        opener = lambda: gzip.open(filepath, "rt", encoding="utf-8")
    else:
        opener = lambda: open(filepath, "r", encoding="utf-8")
        
    with opener() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                flat_record = flatten_candidate(record)
                rows.append(flat_record)
            except Exception as e:
                skipped += 1
                
            if (i + 1) % 10000 == 0:
                print(f"Loaded {i + 1} candidates...")
                
    elapsed = time.time() - start_time
    print(f"Loaded {len(rows)} candidates in {elapsed:.1f}s ({skipped} skipped)")
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Quick test if run directly
    try:
        # pyrefly: ignore [missing-import]
        from config import CANDIDATES_FILE
        filepath = CANDIDATES_FILE
    except ImportError:
        filepath = "data/candidates.jsonl"
        
    df = load_candidates(filepath)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
