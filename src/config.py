"""
Configuration module for the Redrob Intelligent Candidate Ranker.
Centralizes all file paths, keyword taxonomies, scoring thresholds, and the Job Description.
"""

from pathlib import Path

# 1. FILE PATHS & DIRECTORIES

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
CANDIDATES_GZ_FILE = DATA_DIR / "candidates.jsonl.gz"

# Output and Model files
SUBMISSION_OUTPUT = OUTPUT_DIR / "submission.csv"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
LLM_LABELS_PATH = MODELS_DIR / "llm_labels.csv"


# 2. KEYWORD TAXONOMIES (Derived from JD)
# All sets are frozen and lowercase for safe, case-insensitive O(1) lookups.

# Job titles that indicate highly relevant engineering experience.
ENGINEERING_TITLES = frozenset([
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer", "data scientist",
    "senior data scientist", "lead data scientist", "nlp engineer",
    "research engineer", "applied scientist", "software engineer",
    "senior software engineer", "backend engineer", "platform engineer",
    "data engineer", "senior data engineer", "deep learning engineer",
    "computer vision engineer", "junior ml engineer", "junior ai engineer",
    "senior machine learning engineer"
])

# Job titles that strongly indicate the candidate is NOT an AI engineer.
NON_ENGINEERING_TITLES = frozenset([
    "marketing manager", "operations manager", "hr manager",
    "sales executive", "accountant", "content writer",
    "graphic designer", "project manager", "business analyst",
    "mechanical engineer", "civil engineer", "chemical engineer",
    "customer support", "qa engineer", "recruiter"
])

# Keywords indicating the candidate has built things for real users, not just toy/academic projects.
# The JD explicitly filters out "pure research environments" and demands "production deployment".
PRODUCTION_KEYWORDS = frozenset([
    "production", "deployed", "shipped", "served", "real-time",
    "end-to-end", "a/b test", "a/b testing", "online experiment",
    "latency", "throughput", "scaled to", "million users",
    "inference", "model serving", "feature store", "ci/cd",
    "monitoring", "alerting", "api endpoint", "microservice",
    "kubernetes", "docker", "aws", "gcp", "azure"
])

# Core domain skills for this specific role (Search/Ranking/Retrieval).
# The JD states: "We're looking for people who understood retrieval and ranking before it became fashionable."
RETRIEVAL_DOMAIN_KEYWORDS = frozenset([
    "retrieval", "ranking", "search", "recommendation",
    "embeddings", "vector search", "vector database",
    "semantic search", "hybrid search", "re-ranking",
    "bm25", "tf-idf", "inverted index", "ndcg", "mrr",
    "map", "recall@k", "precision@k", "candidate generation",
    "matching", "relevance", "information retrieval",
    "learning to rank", "sentence-transformers", "bi-encoder",
    "cross-encoder", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "elasticsearch", "opensearch", "chromadb", "rag",
    "retrieval augmented generation"
])

# Indicators of the academic/research trap explicitly warned against in the JD.
RESEARCH_ONLY_KEYWORDS = frozenset([
    "published paper", "thesis", "phd research", "academic lab",
    "research assistant", "journal", "conference paper",
    "arxiv", "no production", "theoretical"
])

# Indicators of the "LangChain wrapper" trap explicitly warned against in the JD.
LLM_WRAPPER_KEYWORDS = frozenset([
    "langchain", "llamaindex", "openai api", "chatgpt",
    "gpt wrapper", "prompt template"
])

# IT service/consulting firms.
# JD Trap: "People who have only worked at consulting firms... in their entire career."
CONSULTING_FIRMS = frozenset([
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "deloitte", "mphasis",
    "ltimindtree", "mindtree", "ibm", "ey", "kpmg", "pwc",
    "mckinsey", "bain", "bcg"
])

# Fast-paced product companies (positive signal vs consulting).
PRODUCT_COMPANIES = frozenset([
    "google", "meta", "facebook", "amazon", "microsoft",
    "apple", "netflix", "flipkart", "swiggy", "zomato",
    "razorpay", "cred", "phonepe", "paytm", "ola", "uber",
    "grab", "gojek", "linkedin", "twitter", "x", "snap",
    "salesforce", "adobe", "nvidia", "intel", "stripe",
    "square", "airbnb", "spotify", "databricks", "snowflake",
    "confluent", "samsung", "qualcomm", "oracle", "meesho",
    "myntra", "nykaa", "freshworks", "zoho", "postman",
    "atlassian", "slack", "notion", "redrob"
])

# Hard technical requirements from the JD's "Things you absolutely need" section.
MUST_HAVE_SKILLS = frozenset([
    "sentence transformers", "sentence-transformers",
    "openai embeddings", "bge", "e5", "embeddings",
    "embedding", "word2vec", "fasttext", "bert", "pinecone",
    "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "opensearch", "chromadb", "python", "ndcg", "mrr", "map",
    "ranking evaluation"
])

# Bonus skills from the JD's "Things we'd like you to have" section.
NICE_TO_HAVE_SKILLS = frozenset([
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms",
    "xgboost", "lightgbm", "learning to rank", "distributed systems",
    "onnx", "tensorrt", "triton"
])

# Skills explicitly called out as irrelevant in the JD's "Things we explicitly do NOT want" section.
IRRELEVANT_DOMAIN_SKILLS = frozenset([
    "opencv", "image classification", "object detection",
    "cnn", "gans", "computer vision", "yolo", "speech recognition",
    "tts", "asr", "robotics", "ros", "slam", "solidworks",
    "autocad", "ansys", "photoshop", "illustrator", "figma",
    "seo", "content writing", "sales", "accounting", "six sigma",
    "sap"
])


# 3. SCORING WEIGHTS AND THRESHOLDS

# JD: "5-9 years... we've used 5-9 because it's roughly where people we've hired... landed"
IDEAL_EXPERIENCE_YEARS = 7.0
EXPERIENCE_RANGE_MIN = 5.0
EXPERIENCE_RANGE_MAX = 9.0
# Hard boundaries to catch impossible resumes or massive under-qualifications
EXPERIENCE_HARD_MIN = 2.0
EXPERIENCE_HARD_MAX = 15.0

# Behavioral/Redrob signals thresholds
INACTIVE_DAYS_THRESHOLD = 90
GHOST_DAYS_THRESHOLD = 180
MIN_RESPONSE_RATE = 0.10
IDEAL_RESPONSE_RATE = 0.60

# JD: "We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher."
MAX_NOTICE_PERIOD_PREFERRED = 30
MAX_NOTICE_PERIOD_ACCEPTABLE = 90

# JD: "Location: Pune/Noida-preferred... Open to relocation candidates from Tier-1 Indian cities"
PREFERRED_CITIES = frozenset(["pune", "noida"])
TIER1_INDIAN_CITIES = frozenset([
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "gurgaon", "gurugram", "kolkata", "ahmedabad"
])

# The date to measure 'recency' metrics against (e.g., last_active_date)
REFERENCE_DATE = "2026-06-01"


# 4. FULL JD TEXT
# Used by the LLM later during offline labeling to ensure it understands the exact context.

JD_TEXT = '''Job Description: Senior AI Engineer — Founding Team Company:  Redrob AI (Series A AI-native talent intelligence platform) Location:  Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities Employment Type:  Full-time Experience Required:  5–9 years (see "what we mean by this" below) Let's be honest about this role We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before. If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it. If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it. We need someone who is  simultaneously  comfortable with two things that sound contradictory: Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning. Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for. These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher. What you'd actually be doing The high-level mandate:  own the intelligence layer of Redrob's product.  That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles. In practical terms, your first 90 days will probably look like: Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix. Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call. Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind. Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build. What we mean by "5-9 years" This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong. That said , here are the disqualifiers we actually apply: If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side. If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking  before  it became fashionable. If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or "tech lead" roles — we will probably not move forward. This role writes code. The skills inventory (please read carefully) Most JDs list 20 skills and you're supposed to have all of them. We're going to do this differently. Things you absolutely need Production experience with  embeddings-based retrieval systems  (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production. Production experience with  vector databases or hybrid search infrastructure  — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does. Strong  Python . Yes really, we care about code quality. Hands-on experience designing  evaluation frameworks for ranking systems  — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful. Things we'd like you to have but won't reject you for LLM fine-tuning experience (LoRA, QLoRA, PEFT) Experience with learning-to-rank models (XGBoost-based or neural) Prior exposure to HR-tech, recruiting tech, or marketplace products Background in distributed systems or large-scale inference optimization Open-source contributions in the AI/ML space Things we explicitly do NOT want This is the section most JDs skip but we think it's the most important: Title-chasers.  If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years. Framework enthusiasts.  If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks. People who have only worked at consulting firms  (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine. People whose primary expertise is computer vision, speech, or robotics  without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here. People whose work has been entirely on closed-source proprietary systems for 5+ years  without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think. On location, comp, and logistics Location:  Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas. Notice period:  We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher. The vibe check We genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't. We work async-first and write a lot. If you find writing painful, you'll find this role painful. We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive. We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable. How to read between the lines The "ideal candidate" we're imagining is roughly: 6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services). Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale. Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) — and can defend them with reference to systems they actually built. Located in or willing to relocate to Noida or Pune. Active on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them. We are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that — we'd rather see 10 great matches than 1000 maybes. Final note for the participants of the Redrob hackathon If you're reading this in the context of the Intelligent Candidate Discovery & Ranking Challenge: The "right answer" to this JD is not "find candidates whose skills section contains the most AI keywords." That's a trap we've explicitly built into the dataset. The right answer involves reasoning about the  gap between what the JD says and what the JD means . A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit, no matter how perfect their skill list looks. Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately. Good luck.'''
