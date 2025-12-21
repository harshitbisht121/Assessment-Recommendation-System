"""
Configuration file for SHL Recommendation System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# SHL Scraping Config
SHL_CONFIG = {
    'base_url': 'https://www.shl.com/products/product-catalog/',
    'base_domain': 'https://www.shl.com',
    'product_type_individual': 1,
    'product_type_packaged': 2,
    'page_size': 12,
    'min_assessments': 377,
    'headless': True
}

# Model Config
MODEL_CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',  # Fast and good quality
    'embedding_dim': 384,
    'top_k_retrieval': 20,  # Retrieve more for re-ranking
    'top_k_final': 10,  # Final recommendations
}

# API Config
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info'
}

# LLM Config
LLM_CONFIG = {
    'api_key': os.getenv('GOOGLE_API_KEY', ''),
    'model_name': 'gemini-2.5-flash',
    'temperature': 0.3,
    'max_tokens': 1024
}

# Evaluation Config
EVAL_CONFIG = {
    'k_values': [5, 10],
    'train_path': DATASETS_DIR / 'train.csv',
    'test_path': DATASETS_DIR / 'test.csv',
    'predictions_path': BASE_DIR / 'predictions.csv'
}

# Test Type Mapping
TEST_TYPES = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

# Logging Config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}