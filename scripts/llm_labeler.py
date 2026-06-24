#!/usr/bin/env python3
"""
LLM Labeler Script.
Samples candidates, queries an LLM to score their relevance, and saves to CSV.
This acts as our training data generator for the XGBoost model.
Supports OpenAI, Anthropic, Gemini, OpenRouter, and Groq via REST APIs.
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys_path = str(Path(__file__).parent.parent)
import sys
if sys_path not in sys.path:
    sys.path.append(sys_path)

from src import config
from src.data_loader import load_candidates
from src.feature_engine import extract_features
from src.honeypot_detector import detect_honeypots

# Load environment variables
load_dotenv()

# Hardcoded Job Description for the hackathon
JD_TEXT = """
Senior AI Engineer - Search and Ranking
We are looking for an engineer with 6-8 years of experience.
Must have built and deployed ranking, search, or recommendation systems at product companies.
Must not be purely from IT consulting firms without product experience.
Required skills: Python, Machine Learning, Information Retrieval, NLP.
"""

PROMPT_TEMPLATE = """You are an expert technical recruiter evaluating candidates for a specific role.

## The Job Description
{JD_TEXT}

## The Candidate Profile
{CANDIDATE_JSON}

## Your Task
Score this candidate's relevance to the job description on a scale of 0-5:

0 = Completely irrelevant (wrong domain, wrong role type entirely)
1 = Very weak fit (some tangential overlap but fundamentally wrong background)
2 = Weak fit (relevant industry but wrong specific role or missing core skills)
3 = Moderate fit (relevant background, has some required skills, but notable gaps)
4 = Strong fit (matches most requirements, minor gaps in some areas)
5 = Exceptional fit (matches the JD's description of the 'ideal candidate' closely)

## Important Scoring Guidance
- A candidate with the right TITLE but wrong DESCRIPTION should score lower than you'd expect
- A candidate at a consulting firm (TCS, Infosys, Wipro etc.) with no product company experience should score at most 2, per the JD's explicit disqualifier
- A candidate with heavy CV/speech/robotics background and no NLP/IR should score at most 2
- A candidate whose skills list contains many AI keywords but whose career descriptions show no AI work is likely a keyword stuffer -- score at most 1
- Consider behavioral signals: a great-on-paper candidate who hasn't been active in 6 months should be scored lower
- The JD says the ideal candidate has 6-8 years, shipped ranking/search/recommendation systems, and works at product companies

## Response Format
Return a JSON object with exactly two keys:
{
  "score": <integer 0-5>,
  "reasoning": "<1-2 sentence explanation>"
}

Return ONLY the JSON object. Do not wrap in markdown tags like ```json. No additional text.
"""

def sample_candidates(df: pd.DataFrame, target_total: int = 3000) -> list:
    """Stratified sampling of candidates."""
    print("Performing stratified sampling...")
    sampled_ids = set()
    samples = []
    
    bucket_size = target_total // 6
    
    # We need full features and honeypot scores
    df_feat = extract_features(df)
    is_hp, susp_scores = detect_honeypots(df_feat)
    df_feat['suspicion_score'] = susp_scores
    
    def add_to_sample(mask, limit, name):
        pool = df_feat[mask & ~df_feat['candidate_id'].isin(sampled_ids)]
        selected = pool.sample(n=min(limit, len(pool)), random_state=42)
        samples.append(selected)
        sampled_ids.update(selected['candidate_id'].tolist())
        print(f"  {name}: {len(selected)} candidates")
        return limit - len(selected)
        
    # Bucket 1: Clearly relevant
    b1_mask = (df_feat['feat_dim1_is_ai_title'] == 1) & (df_feat['feat_dim3_retrieval_skill_count'] >= 2) & (df_feat['feat_dim5_years_exp'].between(4, 12))
    rem1 = add_to_sample(b1_mask, bucket_size, "Bucket 1 (Clearly relevant)")
    
    # Bucket 2: Plausibly relevant
    b2_mask = (df_feat['feat_dim1_is_eng_title'] == 1) & (df_feat['feat_dim4_total_skills'] >= 5) & (df_feat['feat_dim5_years_exp'].between(3, 15))
    rem2 = add_to_sample(b2_mask, bucket_size + rem1, "Bucket 2 (Plausibly relevant)")
    
    # Bucket 3: Edge cases
    b3_mask = df_feat['profile_current_title'].str.lower().str.contains('data engineer|qa|junior|intern', na=False)
    rem3 = add_to_sample(b3_mask, bucket_size + rem2, "Bucket 3 (Edge cases)")
    
    # Bucket 4: Clearly irrelevant
    b4_mask = df_feat['feat_dim1_is_non_eng_title'] == 1
    rem4 = add_to_sample(b4_mask, bucket_size + rem3, "Bucket 4 (Clearly irrelevant)")
    
    # Bucket 5: Honeypots & Keyword stuffers
    b5_mask = df_feat['suspicion_score'] > 0.6
    rem5 = add_to_sample(b5_mask, bucket_size + rem4, "Bucket 5 (Honeypots)")
    
    # Bucket 6: Random remainder
    b6_mask = pd.Series(True, index=df_feat.index)
    rem6 = add_to_sample(b6_mask, bucket_size + rem5, "Bucket 6 (Random uniform)")
    
    final_sample = pd.concat(samples)
    print(f"Total sampled: {len(final_sample)}")
    return final_sample['candidate_id'].tolist()

def call_openai_api(prompt: str, model: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def call_anthropic_api(prompt: str, model: str, api_key: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def call_gemini_api(prompt: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def call_openrouter_api(prompt: str, model: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def call_groq_api(prompt: str, model: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def score_llm(candidate_json: dict, provider: str, model: str) -> dict:
    """Call the specified LLM API to score a candidate."""
    prompt = PROMPT_TEMPLATE.replace("{JD_TEXT}", JD_TEXT).replace("{CANDIDATE_JSON}", json.dumps(candidate_json, indent=2))
    
    retries = 3
    for attempt in range(retries):
        try:
            content = ""
            if provider == "openai":
                content = call_openai_api(prompt, model, os.environ["OPENAI_API_KEY"])
            elif provider == "anthropic":
                content = call_anthropic_api(prompt, model, os.environ["ANTHROPIC_API_KEY"])
            elif provider == "gemini":
                content = call_gemini_api(prompt, model, os.environ["GEMINI_API_KEY"])
            elif provider == "openrouter":
                content = call_openrouter_api(prompt, model, os.environ["OPENROUTER_API_KEY"])
            elif provider == "groq":
                content = call_groq_api(prompt, model, os.environ["GROQ_API_KEY"])
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            content = content.strip()
            
            # Clean up markdown if present
            if content.startswith("```json"): content = content[7:]
            if content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
                
            parsed = json.loads(content)
            if 'score' not in parsed or 'reasoning' not in parsed:
                raise ValueError("Missing keys in LLM response")
            return parsed
            
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts. Error: {e}")
                return {"score": 1, "reasoning": f"Error: {str(e)}"}
            time.sleep(5)

def get_raw_candidate_json(filepath, candidate_id_set):
    """Yield raw JSON for sampled candidates."""
    import gzip
    opener = gzip.open if str(filepath).endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if record['candidate_id'] in candidate_id_set:
                yield record

def load_existing_labels(path: Path) -> set:
    if path.exists():
        try:
            df = pd.read_csv(path)
            return set(df['candidate_id'].tolist())
        except:
            return set()
    return set()

def main():
    parser = argparse.ArgumentParser(description="Generate training labels using various LLMs")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "gemini", "openrouter", "groq"])
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()
    
    # Check for API key
    env_key = f"{args.provider.upper()}_API_KEY"
    if env_key not in os.environ:
        print(f"ERROR: {env_key} environment variable is not set. Please add it to your .env file.")
        sys.exit(1)
    
    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_file = models_dir / "llm_labels.csv"
    
    print(f"Using Provider: {args.provider}")
    print(f"Using Model: {args.model}")
    
    # 1. Load data and sample
    df_raw = load_candidates(config.CANDIDATES_FILE)
    sampled_ids = set(sample_candidates(df_raw, args.sample_size))
    
    # 2. Check for existing progress
    already_scored = load_existing_labels(out_file)
    to_score = sampled_ids - already_scored
    print(f"Found {len(already_scored)} already scored. Remaining to score: {len(to_score)}")
    
    if len(to_score) == 0:
        print("All candidates scored!")
        return
        
    # 3. Open output file for appending
    write_header = not out_file.exists() or out_file.stat().st_size == 0
    
    with open(out_file, 'a', encoding='utf-8') as f:
        if write_header:
            f.write("candidate_id,llm_score,llm_reasoning\n")
            
        count = 0
        pbar = tqdm(total=len(to_score))
        
        for record in get_raw_candidate_json(config.CANDIDATES_FILE, to_score):
            cid = record['candidate_id']
            try:
                res = score_llm(record, args.provider, args.model)
                score = int(res.get('score', 1))
                # clean reasoning for CSV
                reason = res.get('reasoning', '').replace('"', '""').replace('\n', ' ')
                
                f.write(f'"{cid}",{score},"{reason}"\n')
                f.flush()
                
            except Exception as e:
                print(f"\nError processing {cid}: {e}")
                
            count += 1
            pbar.update(1)
            time.sleep(1.0) # Rate limit delay
            
        pbar.close()
        
    print("\nLabeling complete. Summary of scores:")
    df_res = pd.read_csv(out_file)
    print(df_res['llm_score'].value_counts().sort_index())

if __name__ == "__main__":
    main()
