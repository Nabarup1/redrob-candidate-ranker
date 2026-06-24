#!/usr/bin/env python3
"""
Feature engineering engine for the Redrob candidate ranker.
Takes the flat DataFrame from data_loader and computes ~60 features across 8 dimensions.
"""

import time
import numpy as np
import pandas as pd

try:
    from src import config
except ImportError:
    import config

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all scoring features from candidate data.
    Adds ~60 columns prefixed with 'feat_' to the DataFrame.
    """
    start_time = time.time()
    df = df.copy()
    
    print("Starting feature extraction on {} rows...".format(len(df)))

    # Helper function for safe string lowercasing
    def safe_lower(series):
        return series.fillna("").str.lower()
        
    current_title_lower = safe_lower(df['profile_current_title'])
    all_titles_lower = safe_lower(df['career_all_titles'])
    all_desc_lower = safe_lower(df['career_all_descriptions'])
    headline_lower = safe_lower(df['profile_headline'])
    summary_lower = safe_lower(df['profile_summary'])
    skills_lower = safe_lower(df['skills_names'])
    
    # DIMENSION 1: ROLE & TITLE ALIGNMENT
    
    df['feat_dim1_is_eng_title'] = current_title_lower.isin(config.ENGINEERING_TITLES).astype(int)
    
    ai_keywords = ['ai', 'ml', 'machine learning', 'data scientist', 'nlp', 'deep learning']
    df['feat_dim1_is_ai_title'] = current_title_lower.apply(
        lambda t: 1 if any(kw in t for kw in ai_keywords) else 0
    )
    
    df['feat_dim1_is_non_eng_title'] = current_title_lower.isin(config.NON_ENGINEERING_TITLES).astype(int)
    
    def check_ever_eng_title(titles_str):
        titles = titles_str.split('|')
        return int(any(t.strip() in config.ENGINEERING_TITLES for t in titles if t.strip()))
        
    def eng_title_ratio(titles_str):
        titles = [t.strip() for t in titles_str.split('|') if t.strip()]
        if not titles: return 0.0
        eng_count = sum(1 for t in titles if t in config.ENGINEERING_TITLES)
        return eng_count / len(titles)
        
    df['feat_dim1_ever_eng_title'] = all_titles_lower.apply(check_ever_eng_title)
    df['feat_dim1_eng_title_ratio'] = all_titles_lower.apply(eng_title_ratio)
    
    senior_keywords = ['senior', 'lead', 'principal', 'staff', 'head']
    df['feat_dim1_is_senior'] = current_title_lower.apply(
        lambda t: 1 if any(kw in t for kw in senior_keywords) else 0
    )
    
    # DIMENSION 2: PRODUCTION ML EXPERIENCE
    
    def count_kws(text, kw_set):
        if not text: return 0
        return sum(1 for kw in kw_set if kw in text)
        
    df['feat_dim2_production_kw_count'] = all_desc_lower.apply(lambda x: count_kws(x, config.PRODUCTION_KEYWORDS))
    df['feat_dim2_retrieval_kw_count'] = all_desc_lower.apply(lambda x: count_kws(x, config.RETRIEVAL_DOMAIN_KEYWORDS))
    df['feat_dim2_research_kw_count'] = all_desc_lower.apply(lambda x: count_kws(x, config.RESEARCH_ONLY_KEYWORDS))
    df['feat_dim2_wrapper_kw_count'] = all_desc_lower.apply(lambda x: count_kws(x, config.LLM_WRAPPER_KEYWORDS))
    
    df['feat_dim2_production_ratio'] = df['feat_dim2_production_kw_count'] / (
        df['feat_dim2_production_kw_count'] + df['feat_dim2_research_kw_count'] + 1
    )
    
    ml_kws = ['ml', 'ai', 'model', 'algorithm', 'machine learning', 'nlp', 'embeddings']
    def has_shipped_ml(text):
        if not text: return 0
        has_ml = any(kw in text for kw in ml_kws)
        has_prod = any(kw in text for kw in config.PRODUCTION_KEYWORDS)
        return 1 if (has_ml and has_prod) else 0
        
    df['feat_dim2_shipped_ml'] = all_desc_lower.apply(has_shipped_ml)
    
    df['feat_dim2_summary_production'] = summary_lower.apply(
        lambda x: 1 if count_kws(x, config.PRODUCTION_KEYWORDS) > 0 else 0
    )
    
    # DIMENSION 3: RETRIEVAL/SEARCH/RANKING DOMAIN
    
    df['feat_dim3_retrieval_desc_count'] = df['feat_dim2_retrieval_kw_count']
    
    df['feat_dim3_retrieval_skill_count'] = skills_lower.apply(
        lambda x: count_kws(x, config.RETRIEVAL_DOMAIN_KEYWORDS)
    )
    
    retrieval_core = ['retrieval', 'search', 'ranking']
    df['feat_dim3_headline_retrieval'] = headline_lower.apply(
        lambda x: 1 if any(kw in x for kw in retrieval_core) else 0
    ) | summary_lower.apply(
        lambda x: 1 if any(kw in x for kw in retrieval_core) else 0
    )
    
    df['feat_dim3_recommendation_exp'] = all_desc_lower.apply(
        lambda x: 1 if 'recommendation' in x else 0
    )
    
    def has_search_exp(text):
        if not text: return 0
        # replace common false positive phrases
        t = text.replace("job search", "").replace("search for", "")
        return 1 if "search" in t else 0
        
    df['feat_dim3_search_system_exp'] = all_desc_lower.apply(has_search_exp)
    
    df['feat_dim3_retrieval_composite'] = (
        (df['feat_dim3_retrieval_desc_count'].clip(0, 5) / 5) * 0.4 +
        (df['feat_dim3_retrieval_skill_count'].clip(0, 5) / 5) * 0.2 +
        df['feat_dim3_headline_retrieval'] * 0.2 +
        df['feat_dim3_recommendation_exp'] * 0.1 +
        df['feat_dim3_search_system_exp'] * 0.1
    )
    
    # DIMENSION 4: TECHNICAL STACK MATCH
    
    def count_skills(skills_str, kw_set):
        skills = [s.strip() for s in skills_str.split('|') if s.strip()]
        return sum(1 for s in skills if s in kw_set)
        
    df['feat_dim4_must_have_count'] = skills_lower.apply(lambda x: count_skills(x, config.MUST_HAVE_SKILLS))
    df['feat_dim4_nice_to_have_count'] = skills_lower.apply(lambda x: count_skills(x, config.NICE_TO_HAVE_SKILLS))
    df['feat_dim4_irrelevant_count'] = skills_lower.apply(lambda x: count_skills(x, config.IRRELEVANT_DOMAIN_SKILLS))
    df['feat_dim4_has_python'] = skills_lower.apply(lambda x: 1 if 'python' in [s.strip() for s in x.split('|')] else 0)
    
    df['feat_dim4_total_skills'] = df['skills_count'].fillna(0).astype(int)
    
    df['feat_dim4_relevant_ratio'] = df['feat_dim4_must_have_count'] / (
        df['feat_dim4_must_have_count'] + df['feat_dim4_irrelevant_count'] + 1
    )
    
    # Trust score based on endorsements (if many advanced skills but 0 endorsements -> low trust)
    def calculate_skill_trust(row):
        adv_count = row['skills_advanced_count']
        if adv_count == 0: return 0.5
        avg_end = row['skills_total_endorsements'] / adv_count
        if adv_count >= 6 and avg_end < 3: return 0.2
        if adv_count >= 4 and avg_end >= 10: return 1.0
        return 0.6
        
    df['feat_dim4_skill_trust_score'] = df.apply(calculate_skill_trust, axis=1)
    
    # Advanced relevant requires iterating over the original parsed skills or assuming ratio
    # Since we flattened skills, we use an approximation:
    # advanced_relevant ~ advanced_count * relevant_ratio
    df['feat_dim4_advanced_relevant'] = (df['skills_advanced_count'] * df['feat_dim4_relevant_ratio']).astype(int)
    df['feat_dim4_relevant_skill_duration'] = df['skills_avg_duration'] * df['feat_dim4_relevant_ratio']
    
    def get_assessment_avg(score_dict):
        if not score_dict or not isinstance(score_dict, dict): return -1.0
        scores = list(score_dict.values())
        if not scores: return -1.0
        return sum(scores) / len(scores)
        
    df['feat_dim4_assessment_avg'] = df['sig_assessment_scores'].apply(get_assessment_avg)
    
    # DIMENSION 5: EXPERIENCE BAND FIT
    
    years = df['profile_years_of_exp'].fillna(0).astype(float)
    df['feat_dim5_years_exp'] = years
    df['feat_dim5_exp_band_score'] = np.exp(-0.5 * ((years - config.IDEAL_EXPERIENCE_YEARS) / 2.0) ** 2)
    df['feat_dim5_in_ideal_range'] = ((years >= config.EXPERIENCE_RANGE_MIN) & (years <= config.EXPERIENCE_RANGE_MAX)).astype(int)
    df['feat_dim5_too_junior'] = (years < config.EXPERIENCE_HARD_MIN).astype(int)
    df['feat_dim5_too_senior'] = (years > config.EXPERIENCE_HARD_MAX).astype(int)
    
    # DIMENSION 6: COMPANY QUALITY & CAREER TYPE
    
    curr_comp_lower = safe_lower(df['profile_current_company'])
    all_comps_lower = safe_lower(df['career_all_companies'])
    
    df['feat_dim6_current_consulting'] = curr_comp_lower.isin(config.CONSULTING_FIRMS).astype(int)
    
    def consulting_only(comps_str):
        comps = [c.strip() for c in comps_str.split('|') if c.strip()]
        if not comps: return 0
        return 1 if all(c in config.CONSULTING_FIRMS for c in comps) else 0
        
    def consulting_ratio(comps_str):
        comps = [c.strip() for c in comps_str.split('|') if c.strip()]
        if not comps: return 0.0
        return sum(1 for c in comps if c in config.CONSULTING_FIRMS) / len(comps)
        
    def count_product_comps(comps_str):
        comps = [c.strip() for c in comps_str.split('|') if c.strip()]
        return sum(1 for c in set(comps) if c in config.PRODUCT_COMPANIES)
        
    df['feat_dim6_consulting_only'] = all_comps_lower.apply(consulting_only)
    df['feat_dim6_consulting_ratio'] = all_comps_lower.apply(consulting_ratio)
    df['feat_dim6_product_company_count'] = all_comps_lower.apply(count_product_comps)
    df['feat_dim6_has_product_exp'] = (df['feat_dim6_product_company_count'] > 0).astype(int)
    
    df['feat_dim6_avg_tenure'] = df['career_avg_tenure_months'].fillna(0).astype(float)
    df['feat_dim6_is_title_chaser'] = ((df['feat_dim6_avg_tenure'] < 18.0) & (df['career_num_jobs'] >= 3)).astype(int)
    df['feat_dim6_career_consistency'] = df['feat_dim1_eng_title_ratio']
    
    # DIMENSION 7: LOCATION & LOGISTICS
    
    loc_lower = safe_lower(df['profile_location'])
    country_lower = safe_lower(df['profile_country'])
    
    df['feat_dim7_is_india'] = (country_lower == "india").astype(int)
    
    df['feat_dim7_preferred_city'] = loc_lower.apply(
        lambda x: 1 if any(city in x for city in config.PREFERRED_CITIES) else 0
    )
    df['feat_dim7_tier1_city'] = loc_lower.apply(
        lambda x: 1 if any(city in x for city in config.TIER1_INDIAN_CITIES) else 0
    )
    
    notice = df['sig_notice_period_days'].fillna(180).astype(int)
    df['feat_dim7_notice_period'] = notice
    
    def notice_score(n):
        if n <= 30: return 1.0
        if n <= 60: return 0.8
        if n <= 90: return 0.5
        if n <= 120: return 0.3
        return 0.1
        
    df['feat_dim7_notice_score'] = notice.apply(notice_score)
    df['feat_dim7_willing_relocate'] = df['sig_willing_relocate'].fillna(False).astype(int)
    
    workmode_map = {"onsite": 0.9, "hybrid": 1.0, "flexible": 1.0, "remote": 0.7}
    df['feat_dim7_workmode_score'] = safe_lower(df['sig_work_mode']).map(workmode_map).fillna(0.5)
    
    sal_min = df['sig_salary_min'].fillna(0)
    sal_max = df['sig_salary_max'].fillna(0)
    # Check for honeypot or completely unreasonable expectations (expected 20-50 LPA)
    df['feat_dim7_salary_reasonable'] = ((sal_min <= sal_max) & (sal_max <= 60.0) & (sal_min >= 10.0)).astype(int)
    
    # DIMENSION 8: EDUCATION SIGNAL
    
    # tier_1 = 4, tier_2 = 3, tier_3 = 2, tier_4 = 1, unknown = 0
    tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 4}
    df['feat_dim8_best_tier'] = safe_lower(df['edu_best_tier']).map(tier_map).fillna(4).astype(int)
    df['feat_dim8_good_institution'] = (df['feat_dim8_best_tier'] <= 2).astype(int)
    
    relevant_fields = ['computer science', 'artificial intelligence', 'machine learning', 'data science', 'information technology']
    df['feat_dim8_relevant_field'] = safe_lower(df['edu_fields']).apply(
        lambda x: 1 if any(f in x for f in relevant_fields) else 0
    )
    
    grad_degrees = ['ph.d', 'phd', 'm.tech', 'm.s.', 'm.sc', 'm.e.']
    df['feat_dim8_has_grad_degree'] = safe_lower(df['edu_highest_degree']).apply(
        lambda x: 1 if any(d == x for d in grad_degrees) else 0
    )
    
    tier_score_map = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3}
    df['feat_dim8_tier_score'] = df['feat_dim8_best_tier'].map(tier_score_map).fillna(0.3)
    
    elapsed = time.time() - start_time
    print(f"Extracted features in {elapsed:.1f}s")
    
    return df

if __name__ == "__main__":
    # Test script if run directly
    import sys
    from pathlib import Path
    
    # Add the project root to sys.path so 'src' module can be found
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from src.data_loader import load_candidates
        from src import config
        df_raw = load_candidates(config.CANDIDATES_FILE)
        df_feat = extract_features(df_raw)
        
        # Verify columns were added
        feat_cols = [c for c in df_feat.columns if c.startswith('feat_')]
        print(f"\nGenerated {len(feat_cols)} feature columns:")
        print(feat_cols[:10] + ["..."])
        
        print(f"\nDataFrame shape: {df_feat.shape}")
    except Exception as e:
        print(f"Error running standalone test: {e}")
