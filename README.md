# 🎯 SHL Assessment Recommendation System

A production-ready hybrid semantic + lexical retrieval system that intelligently recommends SHL assessments based on job descriptions and natural language queries.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This system combines the power of semantic understanding with traditional keyword matching to deliver accurate assessment recommendations. Built with modern NLP techniques and deployed as a lightweight API, it helps match the right SHL assessments to specific job requirements and skill profiles.

### Key Technologies

- **Semantic Search**: SentenceTransformers + FAISS for vector similarity
- **Lexical Matching**: BM25 for keyword relevance
- **API Framework**: FastAPI for high-performance endpoints
- **Hybrid Fusion**: Balanced scoring across multiple retrieval methods

---

## ✨ Features

- 🔍 **Natural Language Queries** — Input job descriptions or free-form text
- 🎯 **Hybrid Retrieval** — Combines semantic embeddings with lexical ranking
- ⚡ **Fast Search** — FAISS-powered nearest-neighbor lookup
- 🎭 **Balanced Results** — Recommends across technical, behavioral, and cognitive assessments
- 📊 **Structured Output** — Detailed assessment metadata with confidence scores
- 🧪 **Tested & Validated** — Includes comprehensive test suite
- 🚀 **Production Ready** — Fully documented and deployable

---

## 📁 Project Structure

```
SHL_RECOMMENDATION_SYSTEM/
├── data/
│   ├── raw/
│   │   └── shl_catalog.csv           # Raw SHL assessment data
│   ├── processed/
│   │   ├── metadata.csv              # Processed assessment metadata
│   │   ├── model_info.pkl            # Model configuration
│   │   └── embeddings.npy            # Vector embeddings
│   └── datasets/
│       ├── test.csv                  # Test dataset
│       └── train.csv                 # Training dataset
├── src/
│   ├── evaluation/
│   │   └── metrics.py                # Evaluation metrics
│   ├── api/
│   │   └── app.py                    # FastAPI server
│   ├── recommendation/
│   │   ├── recommender.py            # Recommendation engine
│   │   ├── retriever.py              # Hybrid retrieval logic
│   │   └── embedder.py               # Embedding generation
│   ├── frontend/
│   │   └── app.py                    # Streamlit UI (optional)
│   └── scraper/
│       └── shl_scraper.py            # Data scraper
├── notebooks/
│   ├── 01_scraping.ipynb             # Data collection notebook
│   ├── 02_embedding.ipynb            # Embedding exploration
│   └── 03_evaluation.ipynb           # Model evaluation
├── .streamlit/
│   └── secrets.toml                  # Streamlit secrets
├── venv/                             # Virtual environment
├── .env                              # Environment variables
├── .gitignore
├── config.py                         # Configuration settings
├── test_system.py                    # System validation tests
├── predictions.csv                   # Model predictions
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/SHL-Assessment-Recommendation-System.git
   cd SHL-Assessment-Recommendation-System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Generate Embeddings

Before running the system, generate the semantic embeddings:

```bash
python src/recommendation/embedder.py
```

This creates three files in `data/processed/`:
- `embeddings.npy` — Vector representations
- `metadata.csv` — Assessment details
- `model_info.pkl` — Model configuration

### Run Tests

Validate the system setup:

```bash
python test_system.py
```

Expected output:
```
✓ Embeddings: PASSED
✓ Recommender: PASSED
✓ API: PASSED
```

### Start the API

Launch the FastAPI server:

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Access the API:
- **Health Check**: http://localhost:8000/health
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Optional: Launch Streamlit UI

For a web-based interface:

```bash
streamlit run src/frontend/app.py
```

---

## 📖 API Usage

### Endpoints

#### `GET /health`
Health check endpoint to verify the service is running.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /recommend`
Get assessment recommendations based on a query.

**Request:**
```json
{
  "query": "Python developer with data analysis skills",
  "top_k": 5
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Python Coding Assessment",
      "url": "https://shl.com/assessments/python",
      "description": "Evaluates Python programming proficiency",
      "test_type": "Technical",
      "duration": "45 minutes",
      "adaptive_support": true,
      "remote_support": true,
      "relevance_score": 0.92
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Assessment name |
| `url` | string | SHL product page link |
| `description` | string | Brief assessment description |
| `test_type` | string | Category (Technical, Behavioral, Cognitive) |
| `duration` | string | Estimated completion time |
| `adaptive_support` | boolean | Adaptive testing capabilities |
| `remote_support` | boolean | Remote proctoring availability |
| `relevance_score` | float | Confidence score (0-1) |

---

## 🧠 How It Works

### 1. Semantic Embeddings

The system uses **all-MiniLM-L6-v2**, a transformer-based model, to generate 384-dimensional vector embeddings for each assessment. These embeddings capture the semantic meaning of assessment descriptions.

### 2. Hybrid Retrieval

Combines two complementary approaches:

- **FAISS Vector Search**: Finds semantically similar assessments using cosine similarity
- **BM25 Lexical Ranking**: Matches important keywords and phrases

Scores from both methods are normalized and fused using weighted averaging.

### 3. Query Analysis & Categorization

The system automatically detects query intent:
- **Technical Skills**: Programming languages, tools, frameworks
- **Behavioral Traits**: Leadership, communication, teamwork
- **Cognitive Abilities**: Problem-solving, reasoning, aptitude

### 4. Score Calibration

Results are balanced across categories to ensure diverse recommendations that cover all relevant assessment types.

---

## 🛠️ Deployment

### Environment Variables

```bash
export PORT=8000
export HOST=0.0.0.0
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Platform Deployment

Deploy to your preferred platform:

- **Railway**: Connect GitHub repo and deploy
- **Render**: Use `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
- **AWS/GCP**: Deploy as containerized service
- **Heroku**: Add `Procfile` with web command

---

## 🧪 Testing

### Manual Testing

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "software engineer with leadership skills", "top_k": 3}'
```

### Automated Tests

Run the test suite:
```bash
python test_system.py
```

Tests cover:
- Embedding generation and loading
- Retriever functionality
- API endpoint responses
- Output format validation

### Jupyter Notebooks

Explore the development process:
```bash
jupyter notebook
```

- `01_scraping.ipynb` — Data collection from SHL
- `02_embedding.ipynb` — Embedding model exploration
- `03_evaluation.ipynb` — Performance metrics and analysis

---

## 📈 Future Enhancements

- [ ] Add evaluation metrics with held-out test dataset
- [ ] Multi-language support for international queries
- [ ] LLM-based reranking for improved semantic matching
- [ ] Analytics dashboard for recommendation insights
- [ ] User feedback loop for continuous improvement
- [ ] Caching layer for frequently queried recommendations

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is available for academic and demonstration purposes. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **SHL** for assessment dataset guidelines and evaluation metrics
- **Sentence-Transformers** for pre-trained embedding models
- **FAISS** by Meta AI for efficient similarity search
- **FastAPI** for the modern Python web framework

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">
  Made with ❤️ for better talent assessment
</div>