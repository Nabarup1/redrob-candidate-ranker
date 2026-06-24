#!/usr/bin/env python3
"""
Data exploration and profiling for the Redrob candidate dataset.
Produces a statistical profile that informs all downstream scoring decisions.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
import statistics

# Constants 
DATA_PATH = Path("data/candidates.jsonl")
OUTPUT_PATH = Path("output/data_profile.json")

# JD-critical skill keyword sets (case-insensitive matching)
EMBEDDING_SKILLS = {"sentence transformers", "openai embeddings", "bge", "e5", "embeddings", "sentence-transformers"}
VECTOR_DB_SKILLS = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch", "chromadb"}
LLM_SKILLS = {"fine-tuning llms", "lora", "qlora", "peft", "llm", "rag", "langchain", "prompt engineering"}
RETRIEVAL_SKILLS = {"bm25", "information retrieval", "search", "ranking", "recommendation systems", "ndcg", "mrr"}
PYTHON_SKILL = {"python"}
ML_GENERAL_SKILLS = {"pytorch", "tensorflow", "scikit-learn", "xgboost", "machine learning", "deep learning", "nlp", "transformers"}
CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "deloitte", "ibm", "mphasis", "ltimindtree", "mindtree"}

def get_exp_bin(years):
    if years < 2: return "0-2"
    if years < 4: return "2-4"
    if years < 6: return "4-6"
    if years < 8: return "6-8"
    if years < 10: return "8-10"
    if years < 12: return "10-12"
    if years < 15: return "12-15"
    return "15+"

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    total_candidates = 0
    
    title_counter = Counter()
    industry_counter = Counter()
    country_counter = Counter()
    
    years_exp_list = []
    exp_bins = Counter()
    
    all_skill_names = Counter()
    skill_proficiency = Counter()
    skills_lengths = Counter()
    
    candidates_with_embeddings = 0
    candidates_with_vector_db = 0
    candidates_with_llm = 0
    candidates_with_retrieval = 0
    candidates_with_python = 0
    candidates_with_ml_general = 0
    
    company_counter = Counter()
    candidates_current_consulting = 0
    candidates_only_consulting = 0
    
    behavioral_stats = defaultdict(list)
    open_to_work_count = 0
    no_github_count = 0
    verified_email_count = 0
    verified_phone_count = 0
    linkedin_connected_count = 0
    inactive_90d_count = 0
    inactive_180d_count = 0
    
    career_lengths = Counter()
    career_industries = Counter()
    career_durations = []
    has_current_job_count = 0
    
    edu_tiers = Counter()
    edu_degrees = Counter()
    edu_fields = Counter()
    cs_ai_related_edu = 0
    
    honeypot_salary_inversion = 0
    honeypot_exp_vs_edu = 0
    honeypot_duration_mismatch = 0
    honeypot_skill_stuffer = 0
    
    ref_date = datetime(2026, 6, 1)

    print("Starting data exploration...")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                cand = json.loads(line)
            except Exception:
                continue
            
            total_candidates += 1
            if total_candidates % 10000 == 0:
                print(f"Processed {total_candidates} candidates...")
                
            prof = cand.get("profile", {})
            title = prof.get("current_title", "")
            title_counter[title] += 1
            industry_counter[prof.get("current_industry", "")] += 1
            country_counter[prof.get("country", "")] += 1
            
            exp = prof.get("years_of_experience", 0)
            years_exp_list.append(exp)
            exp_bins[get_exp_bin(exp)] += 1
            
            skills = cand.get("skills", [])
            skills_lengths[len(skills)] += 1
            
            cand_skills_lower = set()
            advanced_expert_count = 0
            zero_endorsement_adv_expert = 0
            
            for s in skills:
                s_name = s.get("name", "")
                s_name_lower = s_name.lower()
                all_skill_names[s_name] += 1
                skill_proficiency[s.get("proficiency", "")] += 1
                cand_skills_lower.add(s_name_lower)
                
                if s.get("proficiency") in ("advanced", "expert"):
                    advanced_expert_count += 1
                    if s.get("endorsements", 0) == 0:
                        zero_endorsement_adv_expert += 1
                        
            if cand_skills_lower & EMBEDDING_SKILLS: candidates_with_embeddings += 1
            if cand_skills_lower & VECTOR_DB_SKILLS: candidates_with_vector_db += 1
            if cand_skills_lower & LLM_SKILLS: candidates_with_llm += 1
            if cand_skills_lower & RETRIEVAL_SKILLS: candidates_with_retrieval += 1
            if cand_skills_lower & PYTHON_SKILL: candidates_with_python += 1
            if cand_skills_lower & ML_GENERAL_SKILLS: candidates_with_ml_general += 1
            
            current_company = prof.get("current_company", "")
            company_counter[current_company] += 1
            if current_company.lower() in CONSULTING_FIRMS:
                candidates_current_consulting += 1
                
            career = cand.get("career_history", [])
            career_lengths[len(career)] += 1
            
            only_consulting = True
            has_current = False
            total_career_months = 0
            for job in career:
                c_name = job.get("company", "").lower()
                if c_name not in CONSULTING_FIRMS:
                    only_consulting = False
                career_industries[job.get("industry", "")] += 1
                total_career_months += job.get("duration_months", 0)
                career_durations.append(job.get("duration_months", 0))
                if job.get("is_current"):
                    has_current = True
            
            if len(career) > 0 and only_consulting:
                candidates_only_consulting += 1
            if has_current:
                has_current_job_count += 1
                
            edu = cand.get("education", [])
            latest_end_year = 0
            has_cs_related = False
            for e in edu:
                edu_tiers[e.get("tier", "unknown")] += 1
                edu_degrees[e.get("degree", "")] += 1
                field = e.get("field_of_study", "")
                edu_fields[field] += 1
                
                ey = e.get("end_year", 0)
                if ey > latest_end_year:
                    latest_end_year = ey
                    
                field_lower = field.lower()
                if any(k in field_lower for k in ["computer science", "artificial intelligence", "machine learning", "data science"]):
                    has_cs_related = True
                    
            if has_cs_related:
                cs_ai_related_edu += 1
                
            redrob = cand.get("redrob_signals", {})
            for k in ["profile_completeness_score", "recruiter_response_rate", "avg_response_time_hours", 
                      "notice_period_days", "interview_completion_rate", "search_appearance_30d", "saved_by_recruiters_30d"]:
                val = redrob.get(k)
                if val is not None:
                    behavioral_stats[k].append(val)
                    
            gh = redrob.get("github_activity_score")
            if gh is not None and gh != -1:
                behavioral_stats["github_activity_score"].append(gh)
            elif gh == -1:
                no_github_count += 1
                
            oa = redrob.get("offer_acceptance_rate")
            if oa is not None and oa != -1:
                behavioral_stats["offer_acceptance_rate"].append(oa)
                
            if redrob.get("open_to_work_flag"): open_to_work_count += 1
            if redrob.get("verified_email"): verified_email_count += 1
            if redrob.get("verified_phone"): verified_phone_count += 1
            if redrob.get("linkedin_connected"): linkedin_connected_count += 1
            
            last_active = redrob.get("last_active_date")
            if last_active:
                try:
                    la_dt = datetime.strptime(last_active, "%Y-%m-%d")
                    days_inactive = (ref_date - la_dt).days
                    if days_inactive > 180:
                        inactive_180d_count += 1
                    elif days_inactive > 90:
                        inactive_90d_count += 1
                except:
                    pass
                    
            sal_min = redrob.get("expected_salary_range_inr_lpa", {}).get("min", 0)
            sal_max = redrob.get("expected_salary_range_inr_lpa", {}).get("max", 0)
            if sal_min > sal_max:
                honeypot_salary_inversion += 1
                
            if latest_end_year > 0:
                max_possible_exp = 2026 - latest_end_year
                if exp > max_possible_exp + 2:
                    honeypot_exp_vs_edu += 1
                    
            if total_career_months > (exp * 12) * 1.5 or total_career_months < (exp * 12) * 0.3:
                honeypot_duration_mismatch += 1
                
            if advanced_expert_count >= 8 and zero_endorsement_adv_expert >= 6:
                honeypot_skill_stuffer += 1

    print("Data processing complete. Generating report...\n")
    
    def calc_stats(lst):
        if not lst: return {}
        s_lst = sorted(lst)
        n = len(s_lst)
        return {
            "min": s_lst[0],
            "max": s_lst[-1],
            "mean": sum(s_lst)/n,
            "median": statistics.median(s_lst),
            "std": statistics.stdev(s_lst) if n > 1 else 0.0,
            "p25": s_lst[int(n*0.25)],
            "p75": s_lst[int(n*0.75)],
            "p90": s_lst[int(n*0.90)]
        }
    
    report = {
        "total_candidates": total_candidates,
        "title_distribution": dict(title_counter.most_common(20)),
        "industry_distribution": dict(industry_counter.most_common(20)),
        "country_distribution": dict(country_counter.most_common(20)),
        "india_percentage": (country_counter.get("India", 0) / total_candidates * 100) if total_candidates else 0,
        "experience_distribution": {
            "bins": dict(exp_bins),
            "stats": calc_stats(years_exp_list)
        },
        "skill_analysis": {
            "total_unique_skills": len(all_skill_names),
            "top_50_skills": dict(all_skill_names.most_common(50)),
            "skill_proficiency_distribution": dict(skill_proficiency),
            "skills_per_candidate_distribution": dict(skills_lengths),
            "candidates_with_critical_skills": {
                "embeddings": candidates_with_embeddings,
                "vector_db": candidates_with_vector_db,
                "llm": candidates_with_llm,
                "retrieval": candidates_with_retrieval,
                "python": candidates_with_python,
                "ml_general": candidates_with_ml_general
            }
        },
        "company_analysis": {
            "top_30_companies": dict(company_counter.most_common(30)),
            "candidates_current_consulting": candidates_current_consulting,
            "candidates_only_consulting": candidates_only_consulting
        },
        "behavioral_signals": {
            "stats": {k: calc_stats(v) for k, v in behavioral_stats.items()},
            "counts": {
                "open_to_work": open_to_work_count,
                "no_github": no_github_count,
                "verified_email": verified_email_count,
                "verified_phone": verified_phone_count,
                "linkedin_connected": linkedin_connected_count,
                "inactive_more_than_90_days": inactive_90d_count,
                "inactive_more_than_180_days": inactive_180d_count
            }
        },
        "career_history_analysis": {
            "num_jobs_distribution": dict(career_lengths),
            "most_common_industries": dict(career_industries.most_common(20)),
            "average_duration_months_per_job": sum(career_durations) / len(career_durations) if career_durations else 0,
            "candidates_with_current_job": has_current_job_count
        },
        "education_analysis": {
            "tier_distribution": dict(edu_tiers),
            "most_common_degrees": dict(edu_degrees.most_common(20)),
            "most_common_fields": dict(edu_fields.most_common(20)),
            "cs_ai_related_count": cs_ai_related_edu
        },
        "honeypot_indicators": {
            "salary_inversion": honeypot_salary_inversion,
            "experience_vs_education_mismatch": honeypot_exp_vs_edu,
            "career_duration_mismatch": honeypot_duration_mismatch,
            "skill_stuffer": honeypot_skill_stuffer
        }
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Report saved to {OUTPUT_PATH}")
    print("\n Quick Summary ")
    print(f"Total Candidates: {total_candidates}")
    print(f"% in India: {report['india_percentage']:.1f}%")
    print(f"Top 5 Titles: {list(title_counter.most_common(5))}")
    print(f"Candidates with Retrieval skills: {candidates_with_retrieval}")
    print(f"Candidates with Python: {candidates_with_python}")
    print(f"Honeypots (skill stuffers): {honeypot_skill_stuffer}")
    print(f"Honeypots (salary inversion): {honeypot_salary_inversion}")
    
if __name__ == "__main__":
    main()
