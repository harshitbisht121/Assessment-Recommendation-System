SHL Assessment Recommendation System
An intelligent recommendation system that helps recruiters and hiring managers find relevant SHL assessments based on natural language queries or job descriptions.

🎯 Features

Semantic Search: Uses advanced NLP to understand queries and match them with relevant assessments
Smart Balancing: Intelligently balances technical and soft skill assessments based on query requirements
LLM Integration: Leverages Google's Gemini for query enhancement and improved matching
Fast Retrieval: Uses FAISS for efficient similarity search across 377+ assessments
REST API: Fully functional FastAPI backend with health checks
Web Interface: User-friendly Streamlit frontend for easy interaction

🏗️ Architecture
Query → Query Enhancement (LLM) → Semantic Search (FAISS) → 
Balancing (Multi-type) → Re-ranking → Top-K Recommendations
Key Components

Web Scraper: Extracts 377+ Individual Test Solutions from SHL's catalog
Embedding Engine: Creates semantic embeddings using Sentence Transformers
Recommender: Implements hybrid retrieval with LLM-powered enhancement
API: FastAPI server exposing recommendation endpoints
Frontend: Streamlit web application for easy testing

📦 Installation
Prerequisites

Python 3.8+
pip
Virtual environment (recommended)

Setup Steps

Clone the repository

bashgit clone <your-repo-url>
cd shl-recommendation-system

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt
playwright install chromium

Set up environment variables

bash# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env

Run the scraper

bashpython src/scraper/shl_scraper.py

Generate embeddings

bashpython src/recommendation/embedder.py

🚀 Usage
Start the API Server
bashcd src/api
python app.py
API will be available at http://localhost:8000

Health Check: GET http://localhost:8000/health
Recommendations: POST http://localhost:8000/recommend
API Docs: http://localhost:8000/docs

Start the Frontend
bashstreamlit run src/frontend/app.py
Frontend will open at http://localhost:8501
API Usage Example
pythonimport requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "query": "I need Java developers who can collaborate with business teams",
        "top_k": 10
    }
)

results = response.json()
for rec in results['recommendations']:
    print(f"{rec['assessment_name']}: {rec['relevance_score']:.2%}")
Command Line Testing
bash# Test API health
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Need Java and Python developers", "top_k": 5}'
📊 Evaluation
Run Evaluation on Train Set
bashpython src/evaluation/metrics.py
This will:

Load the labeled train data
Generate predictions using the recommender
Calculate Mean Recall@5 and Mean Recall@10
Show per-query performance analysis

Generate Test Predictions
bashpython generate_predictions.py
This creates predictions.csv in the required submission format:
query,assessment_url
"Query 1","https://www.shl.com/..."
"Query 1","https://www.shl.com/..."
...

🧪 Testing
Test with Sample Queries
pythonfrom src.recommendation.recommender import SHLRecommender

recommender = SHLRecommender()

# Test query
query = "I need Java developers who can collaborate well"
recommendations = recommender.recommend(query, top_k=10)

for rec in recommendations:
    print(f"{rec['assessment_name']} ({', '.join(rec['test_type'])})")

📁 Project Structure
shl-recommendation-system/
├── data/
│   ├── raw/                    # Scraped data
│   │   └── shl_catalog.csv
│   ├── processed/              # Embeddings & metadata
│   │   ├── embeddings.npy
│   │   └── metadata.csv
│   └── datasets/               # Train/test sets
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── scraper/
│   │   └── shl_scraper.py     # Web scraper
│   ├── recommendation/
│   │   ├── embedder.py        # Embedding generation
│   │   └── recommender.py     # Recommendation engine
│   ├── evaluation/
│   │   └── metrics.py         # Evaluation metrics
│   ├── api/
│   │   └── app.py             # FastAPI server
│   └── frontend/
│       └── app.py             # Streamlit UI
├── config.py                   # Configuration
├── requirements.txt
└── README.md

🔑 Key Features
1. Smart Query Enhancement
Uses LLM to expand queries with relevant keywords:

"Java developer" → "Java, programming, software development, coding skills"

2. Balanced Recommendations
Automatically detects when queries require multiple skill types:

Technical skills (K) + Soft skills (P) → Returns balanced mix
Example: "Java + collaboration" → 50% K-type, 50% P-type assessments

3. Semantic Matching
Goes beyond keyword matching:

"team player" matches with "collaboration" and "interpersonal skills"
"coding" matches with "programming" and "software development"

4. Fast Retrieval

FAISS index enables sub-second search across 377+ assessments
Normalized cosine similarity for accurate matching

📈 Performance Metrics
The system is evaluated using Mean Recall@K:
Recall@K = (Relevant assessments in top-K) / (Total relevant assessments)
Mean Recall@K = Average of Recall@K across all queries
Expected performance:

Mean Recall@5: > 0.50
Mean Recall@10: > 0.60

🛠️ Technology Stack

Web Scraping: Playwright
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Vector Search: FAISS
LLM: Google Gemini Pro
API: FastAPI + Uvicorn
Frontend: Streamlit
Data: Pandas, NumPy

🐛 Troubleshooting
Common Issues

API not responding

Check if server is running: curl http://localhost:8000/health
Verify port 8000 is not in use
Check logs for errors


Embeddings not found

Run: python src/recommendation/embedder.py
Verify data/processed/embeddings.npy exists


Low recommendation quality

Ensure GOOGLE_API_KEY is set for query enhancement
Check if embeddings are up to date
Verify scraped data has 377+ assessments


Scraper fails

Install Playwright browsers: playwright install chromium
Check internet connection
Try running with headless=False for debugging



📝 Development
Adding New Features

Custom Scoring: Modify recommender.py → semantic_search()
New Test Types: Update TEST_TYPE_INFO in config
Additional Filters: Add to balance_recommendations()

Running Tests
bash# Unit tests (if implemented)
pytest tests/

# Integration test
python -m src.recommendation.recommender

🚢 Deployment
Deploy to Cloud
Recommended platforms:

API: Railway, Render, Fly.io
Frontend: Streamlit Cloud
Storage: Cloud storage for embeddings

Environment Variables
bashGOOGLE_API_KEY=your_gemini_api_key
API_URL=https://your-api-url.com

📄 License
MIT License - See LICENSE file for details

🙏 Acknowledgments
SHL for providing the assessment catalog
Sentence Transformers for embedding models
FastAPI and Streamlit communities
