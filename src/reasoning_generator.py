import pandas as pd
import sys
from pathlib import Path

sys_path = str(Path(__file__).parent.parent)
if sys_path not in sys.path:
    sys.path.append(sys_path)

from src import config

def build_title_fragment(row: pd.Series) -> str:
    title = row.get('profile_current_title', 'Professional')
    if pd.isna(title):
        title = "Professional"
    years = row.get('profile_years_of_exp', 0.0)
    if pd.isna(years):
        years = 0.0
    return f"{title} with {years:.1f} years experience"

def build_company_fragment(row: pd.Series) -> str:
    company = row.get('profile_current_company', 'their current company')
    if pd.isna(company):
        company = "their current company"
        
    # Check for product experience
    if row.get('feat_dim6_has_product_exp', 0) > 0:
        companies_str = row.get('career_all_companies', '')
        if pd.notna(companies_str) and isinstance(companies_str, str):
            companies = companies_str.split('|')
            # Assuming config.PRODUCT_COMPANIES is a set/list of lowercased strings
            product_cos = [c for c in companies if c.strip().lower() in config.PRODUCT_COMPANIES] if hasattr(config, 'PRODUCT_COMPANIES') else []
            if product_cos:
                return f"including product company experience at {product_cos[0]}"
                
    # Check for consulting
    if row.get('feat_dim6_consulting_only', 0) > 0:
        return f"currently at {company} (consulting)"
        
    return f"at {company}"

def build_domain_fragment(row: pd.Series) -> str:
    retrieval_count = row.get('feat_dim3_retrieval_desc_count', 0)
    if pd.isna(retrieval_count): retrieval_count = 0
        
    production_count = row.get('feat_dim2_production_kw_count', 0)
    if pd.isna(production_count): production_count = 0
        
    if retrieval_count >= 3 and production_count >= 3:
        return "career demonstrates hands-on ranking/retrieval system deployment"
    elif retrieval_count >= 2:
        return "has relevant search and retrieval experience"
    elif production_count >= 3:
        return "strong production ML experience though not specifically in retrieval"
    else:
        return "limited direct retrieval or production ML experience"

def build_behavioral_fragment(row: pd.Series) -> str:
    response_rate = row.get('sig_response_rate', 0.5)
    if pd.isna(response_rate): response_rate = 0.5
        
    parts = []
    if response_rate >= 0.7:
        parts.append(f"{response_rate:.0%} recruiter response rate")
    elif response_rate < 0.2:
        parts.append(f"low recruiter response rate ({response_rate:.0%})")
        
    notice = row.get('sig_notice_period_days', 30)
    if pd.isna(notice): notice = 30
        
    if notice <= 30:
        parts.append(f"{notice}-day notice period")
    elif notice > 90:
        parts.append(f"extended {int(notice)}-day notice period")
        
    if parts:
        return "; ".join(parts)
    return "moderate engagement signals"

def build_concern_fragment(row: pd.Series, rank: int) -> str:
    concerns = []
    
    if row.get('feat_dim7_is_india', 1) == 0:
        country = row.get('profile_country', 'unknown')
        if pd.isna(country): country = "unknown"
        concerns.append(f"located outside India ({country})")
        
    notice = row.get('sig_notice_period_days', 30)
    if pd.notna(notice) and notice > 90:
        concerns.append(f"{int(notice)}-day notice period exceeds preferred 30 days")
        
    if row.get('feat_dim6_consulting_only', 0) > 0:
        concerns.append("consulting-only background")
        
    prod_kw = row.get('feat_dim2_production_kw_count', 0)
    if pd.notna(prod_kw) and prod_kw == 0:
        concerns.append("no evidence of production ML deployment in career history")
        
    if not concerns:
        return ""
        
    if rank <= 30:
        return f"Minor concern: {concerns[0]}."
    else:
        return f"Notable gap: {'; '.join(concerns[:2])}."

def generate_reasoning(candidate_row: pd.Series, rank: int, score: float) -> str:
    """
    Generate a specific, factual reasoning string for a ranked candidate.
    
    Args:
        candidate_row: A single row from the DataFrame with all original 
                       and feature columns
        rank: The candidate's rank (1-100)
        score: The candidate's final score
        
    Returns:
        A 1-2 sentence reasoning string
    """
    parts = []
    
    title_frag = build_title_fragment(candidate_row)
    company_frag = build_company_fragment(candidate_row)
    parts.append(f"{title_frag} {company_frag}")
    
    domain_frag = build_domain_fragment(candidate_row)
    parts.append(domain_frag)
    
    behavioral_frag = build_behavioral_fragment(candidate_row)
    parts.append(behavioral_frag)
    
    concern_frag = build_concern_fragment(candidate_row, rank)
    
    reasoning = "; ".join(parts) + "."
    if concern_frag:
        reasoning += " " + concern_frag
        
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."
        
    return reasoning

# Quick self-test
if __name__ == "__main__":
    import numpy as np
    
    # Mock candidate
    mock_data = {
        'profile_current_title': 'Senior ML Engineer',
        'profile_years_of_exp': 7.2,
        'profile_current_company': 'Flipkart',
        'feat_dim6_has_product_exp': 1,
        'career_all_companies': 'Flipkart|TCS',
        'feat_dim3_retrieval_desc_count': 4,
        'feat_dim2_production_kw_count': 5,
        'sig_response_rate': 0.88,
        'sig_notice_period_days': 30,
        'feat_dim7_is_india': 1,
        'feat_dim6_consulting_only': 0
    }
    
    mock_row = pd.Series(mock_data)
    
    print("Testing Rank 1:")
    print(generate_reasoning(mock_row, 1, 0.95))
    
    print("\nTesting Rank 95:")
    mock_data['feat_dim2_production_kw_count'] = 0
    mock_data['sig_response_rate'] = 0.15
    mock_row_95 = pd.Series(mock_data)
    print(generate_reasoning(mock_row_95, 95, 0.45))
