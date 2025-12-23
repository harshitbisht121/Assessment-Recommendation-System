"""
SHL Assessment Recommendation Engine
Uses semantic search with FAISS and LLM-based re-ranking with calibrated scoring
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from pathlib import Path
import google.generativeai as genai
import os
from typing import List, Dict, Optional
import re

class SHLRecommender:
    def __init__(self, data_dir="data/processed", model_name='all-MiniLM-L6-v2'):
        """
        Initialize recommendation system
        
        Args:
            data_dir: Directory with embeddings and metadata
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.data_dir = Path(data_dir)
        
        # Load embeddings and metadata
        self.embeddings = np.load(self.data_dir / "embeddings.npy")
        self.metadata = pd.read_csv(self.data_dir / "metadata.csv")
        
        # Debug: Show available columns
        print(f"Metadata columns: {self.metadata.columns.tolist()}")
        
        # Build FAISS index for fast similarity search
        self.index = self._build_faiss_index()
        
        print("Building BM25 index...")

        texts = (
            self.metadata["name"].fillna("") + " " +
            self.metadata["description"].fillna("")
        ).tolist()

        self.corpus_tokens = [word_tokenize(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        print("✓ BM25 Ready")

        # Initialize Gemini for query enhancement and re-ranking
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use the latest stable model - try multiple options
                try:
                    self.llm = genai.GenerativeModel('gemini-2.5-flash')
                except:
                    try:
                        self.llm = genai.GenerativeModel('gemini-2.0-flash')
                    except:
                        try:
                            self.llm = genai.GenerativeModel('gemini-1.5-flash')
                        except:
                            print("⚠ Warning: No Gemini model available. LLM features disabled.")
                            self.llm = None
            except Exception as e:
                print(f"⚠ Warning: Failed to initialize Gemini: {e}. LLM features disabled.")
                self.llm = None
        else:
            print("⚠ Warning: GOOGLE_API_KEY not set. LLM features disabled.")
            self.llm = None
        
        print(f"✓ Loaded {len(self.metadata)} assessments")
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        index.add(embeddings_norm.astype('float32'))
        
        return index
    
    def enhance_query(self, query: str) -> str:
        """
        Use LLM to enhance query with relevant terms, with fallback to keyword expansion
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query string
        """
        if not self.llm:
            # Fallback: Use keyword-based expansion
            return self._keyword_based_expansion(query)
        
        try:
            prompt = f"""Given this job description or query about hiring:
"{query}"

Extract and list the key skills, competencies, and assessment types needed.
Focus on: technical skills, soft skills, personality traits, cognitive abilities.
Output format: comma-separated list of keywords.
Be concise."""

            response = self.llm.generate_content(prompt)
            enhanced = response.text.strip()
            
            # Combine original query with enhanced terms
            return f"{query} {enhanced}"
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            # Fallback to keyword expansion
            return self._keyword_based_expansion(query)
    
    def _keyword_based_expansion(self, query: str) -> str:
        """
        Fallback keyword-based query expansion when LLM is unavailable
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with related terms
        """
        query_lower = query.lower()
        expansions = []
        
        # Technical skill expansions
        tech_map = {
            'java': 'java programming coding software development',
            'python': 'python programming coding scripting',
            'sql': 'sql database query data',
            'javascript': 'javascript js programming web development',
            'developer': 'developer programmer engineer coder',
            'coding': 'coding programming software development',
        }
        
        # Soft skill expansions
        soft_map = {
            'collaborate': 'collaborate collaboration teamwork interpersonal communication',
            'leadership': 'leadership management supervisor leading team',
            'communication': 'communication interpersonal verbal written',
            'teamwork': 'teamwork collaboration team player cooperative',
        }
        
        # Add expansions
        for keyword, expansion in {**tech_map, **soft_map}.items():
            if keyword in query_lower:
                expansions.append(expansion)
        
        if expansions:
            return f"{query} {' '.join(expansions)}"
        
        return query
    
    def extract_requirements(self, query: str) -> Dict[str, List[str]]:
        """
        Extract skill types and requirements from query
        
        Returns:
            Dict with technical_skills, soft_skills, test_types
        """
        query_lower = query.lower()
        
        requirements = {
            'technical_skills': [],
            'soft_skills': [],
            'test_types': []
        }
        
        # Technical keywords
        tech_keywords = ['java', 'python', 'sql', 'javascript', 'c++', 'react', 'node',
                        'programming', 'coding', 'technical', 'developer', 'engineer']
        
        # Soft skill keywords
        soft_keywords = ['collaborate', 'communication', 'leadership', 'teamwork', 
                        'personality', 'behavior', 'interpersonal', 'management']
        
        # Detect requirements
        for keyword in tech_keywords:
            if keyword in query_lower:
                requirements['technical_skills'].append(keyword)
                requirements['test_types'].append('K')
        
        for keyword in soft_keywords:
            if keyword in query_lower:
                requirements['soft_skills'].append(keyword)
                requirements['test_types'].append('P')
        
        # Deduplicate
        requirements['test_types'] = list(set(requirements['test_types']))
        
        return requirements
    
    def semantic_search(self, query: str, top_k: int = 50):
        """Perform semantic search using FAISS with larger pool"""
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )

        return indices[0].tolist(), scores[0].tolist()
    
    
    def hybrid_search(self, query, semantic_k=200, bm25_k=80):
        """
        Hybrid retrieval:
        - Semantic via FAISS
        - Lexical via BM25
        - Normalize + merge scores
        """

        # ---------- Semantic Retrieval ----------
        sem_idx, sem_scores = self.semantic_search(query, top_k=semantic_k)
        semantic_results = {idx: float(score) for idx, score in zip(sem_idx, sem_scores)}

        # ---------- BM25 Retrieval ----------
        query_tokens = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_ranked = np.argsort(bm25_scores)[::-1][:bm25_k]
        bm25_results = {int(i): float(bm25_scores[i]) for i in bm25_ranked}

        # ---------- Normalize ----------
        if semantic_results:
            max_sem = max(semantic_results.values())
            for k in semantic_results:
                semantic_results[k] /= (max_sem + 1e-8)

        if bm25_results:
            max_bm = max(bm25_results.values())
            for k in bm25_results:
                bm25_results[k] /= (max_bm + 1e-8)

        # ---------- Merge (Weighted) ----------
        final = {}

        # semantic weight higher → keeps good code/query behavior
        for k, v in semantic_results.items():
            final[k] = final.get(k, 0) + v * 0.65

        # bm25 helps business/role language
        for k, v in bm25_results.items():
            final[k] = final.get(k, 0) + v * 0.35

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)

        indices = [x[0] for x in ranked]
        scores = [x[1] for x in ranked]

        return indices, scores


    def smart_rerank(self, indices, scores, query, requirements):
        """
        Weighted reranking to prioritize relevant SHL categories
        Returns raw boosted scores without normalization
        """
        boosted = []

        q = query.lower()

        for idx, score in zip(indices, scores):
            row = self.metadata.iloc[idx]
            test_types = eval(row['test_type']) if isinstance(row['test_type'], str) else row['test_type']

            boost = 0.0
            
            # Category relevance boosting
            if 'K' in requirements['test_types'] and 'K' in test_types:
                boost += 0.15
            if 'P' in requirements['test_types'] and 'P' in test_types:
                boost += 0.15
            
            # Leadership / Manager roles bias
            if any(k in q for k in ["leader", "manager", "coo", "head", "director", "management"]):
                if 'P' in test_types or 'A' in test_types:
                    boost += 0.12

            # Graduate / campus hiring
            if any(k in q for k in ["graduate", "freshers", "entry level", "campus"]):
                if 'P' in test_types or 'A' in test_types:
                    boost += 0.10

            # Cognitive
            if "cognitive" in q or "aptitude" in q:
                if 'A' in test_types:
                    boost += 0.12
            
            # Keyword match boost
            name = str(row['name']).lower()
            desc = str(row.get('description', '')).lower()
            keywords = ["java", "python", "sql", "javascript", "sales", "marketing", "analyst"]

            for k in keywords:
                if k in q and (k in name or k in desc):
                    boost += 0.12
            
            boosted.append((idx, float(score + boost)))

        boosted = sorted(boosted, key=lambda x: x[1], reverse=True)

        indices_sorted = [x[0] for x in boosted]
        scores_sorted = [x[1] for x in boosted]
        
        # Return raw boosted scores (no normalization)
        return indices_sorted, scores_sorted

    def calibrate_score(self, raw_score: float) -> float:
        """
        Calibrate raw scores to meaningful confidence percentages
        
        Args:
            raw_score: Raw boosted score from reranking
            
        Returns:
            Calibrated score between 0 and 1 (0-100%)
        """
        # Based on typical score ranges from hybrid search + boosts
        # These thresholds should be tuned based on your data
        
        if raw_score > 1.5:
            # Excellent match: high semantic similarity + category + keyword matches
            return min(0.85 + (raw_score - 1.5) * 0.10, 1.0)  # 85-100%
        elif raw_score > 1.2:
            # Very good match: good semantic + some boosts
            return 0.70 + (raw_score - 1.2) * 0.50  # 70-85%
        elif raw_score > 0.9:
            # Good match: decent semantic similarity
            return 0.55 + (raw_score - 0.9) * 0.50  # 55-70%
        elif raw_score > 0.6:
            # Moderate match: some relevance
            return 0.35 + (raw_score - 0.6) * 0.67  # 35-55%
        else:
            # Weak match
            return max(raw_score * 0.50, 0.0)  # 0-35%

    def balance_recommendations(
        self,
        indices: List[int],
        scores: List[float],
        requirements: Dict,
        target_count: int = 10
    ) -> List[int]:
        """
        Balance recommendations based on detected requirements
        
        Args:
            indices: Initial recommendation indices
            scores: Similarity scores
            requirements: Dict with technical/soft skill requirements
            target_count: Final number of recommendations
            
        Returns:
            Balanced list of indices
        """
        if not requirements['test_types'] or len(requirements['test_types']) == 1:
            return indices[:target_count]
        
        # Separate by test type
        k_indices = []  # Knowledge & Skills
        p_indices = []  # Personality & Behavior
        other_indices = []
        
        for idx in indices:
            test_types = eval(self.metadata.iloc[idx]['test_type']) \
                        if isinstance(self.metadata.iloc[idx]['test_type'], str) \
                        else self.metadata.iloc[idx]['test_type']
            
            if 'K' in test_types:
                k_indices.append(idx)
            elif 'P' in test_types:
                p_indices.append(idx)
            else:
                other_indices.append(idx)
        
        # Balance if both K and P are required
        if 'K' in requirements['test_types'] and 'P' in requirements['test_types']:
            # Split 60-40 or 50-50 depending on query emphasis
            k_count = target_count // 2
            p_count = target_count - k_count
            
            balanced = k_indices[:k_count] + p_indices[:p_count]
            
            # Fill remaining with best matches
            if len(balanced) < target_count:
                remaining = [idx for idx in indices if idx not in balanced]
                balanced.extend(remaining[:target_count - len(balanced)])
            
            return balanced[:target_count]
        
        return indices[:target_count]
    
    def recommend(
        self,
        query: str,
        top_k: int = 10,
        enhance_query: bool = True,
        balance: bool = True,
        min_relevance: float = 0.30  # Minimum relevance threshold (30%)
    ):
        """
        Generate recommendations with calibrated relevance scores
        
        Args:
            query: User query or job description
            top_k: Number of recommendations to return
            enhance_query: Whether to use LLM query enhancement
            balance: Whether to balance recommendations by category
            min_relevance: Minimum relevance score to include (0-1)
            
        Returns:
            List of recommendations with calibrated relevance scores
        """
        top_k = max(1, min(10, top_k))

        # Enhance query (LLM + fallback)
        search_query = self.enhance_query(query) if enhance_query else query

        # Extract intent
        requirements = self.extract_requirements(query)
    
        # Retrieve larger candidate pool
        indices, scores = self.hybrid_search(search_query, semantic_k=200, bm25_k=80)

        # Convert to list of tuples so we can safely manipulate
        candidates = list(zip(indices, scores))

        # Deduplicate indices
        seen = set()
        unique_candidates = []
        for idx, score in candidates:
            if idx not in seen:
                unique_candidates.append((idx, score))
                seen.add(idx)

        indices = [c[0] for c in unique_candidates]
        scores = [c[1] for c in unique_candidates]
       
        # Balance Categories (Only if test types detected)
        if balance and requirements['test_types']:
            indices = self.balance_recommendations(
                indices,
                scores,
                requirements,
                target_count=40  # Larger pool for fairness
            )

        # Smart Intent Reranking (returns raw boosted scores)
        indices, raw_scores = self.smart_rerank(indices, scores, query, requirements)

        # Calibrate scores to meaningful percentages
        calibrated_scores = [self.calibrate_score(score) for score in raw_scores]

        # Filter by minimum relevance and trim to top_k
        filtered_results = []
        for idx, cal_score, raw_score in zip(indices, calibrated_scores, raw_scores):
            if cal_score >= min_relevance:
                filtered_results.append((idx, cal_score, raw_score))
        
        # Take top_k results
        filtered_results = filtered_results[:top_k]

        # Build Response
        recommendations = []
        for idx, cal_score, raw_score in filtered_results:
            row = self.metadata.iloc[idx]

            test_types = eval(row['test_type']) \
                if isinstance(row['test_type'], str) \
                else row['test_type']
            
            recommendations.append({
                'assessment_name': row['name'],
                'assessment_url': row['url'],
                'description': row.get('description', ''),
                'test_type': test_types,
                'relevance_score': round(float(cal_score), 4),  # Calibrated score
                'raw_score': round(float(raw_score), 4)  # Optional: for debugging
            })

        return recommendations
    
    def recommend_from_url(self, jd_url: str, top_k: int = 10) -> List[Dict]:
        """
        Recommend assessments from job description URL
        
        Args:
            jd_url: URL containing job description
            top_k: Number of recommendations
            
        Returns:
            List of recommendations
        """
        # Fetch content from URL
        try:
            import requests
            response = requests.get(jd_url, timeout=10)
            response.raise_for_status()
            
            # Extract text (simple version - could use BeautifulSoup for better parsing)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            jd_text = soup.get_text(separator=' ', strip=True)
            
            # Truncate if too long
            jd_text = jd_text[:2000]
            
            return self.recommend(jd_text, top_k=top_k)
        
        except Exception as e:
            print(f"Error fetching URL: {e}")
            return []


def main():
    """Test recommendation system"""
    recommender = SHLRecommender()
    
    # Test queries
    test_queries = [
        "I need Java developers who can also collaborate effectively with my business teams.",
        "Looking for mid-level professionals proficient in Python, SQL and JavaScript.",
        "Need cognitive and personality tests for analyst position",
        "Entry level sales representative with communication skills"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        recommendations = recommender.recommend(query, top_k=5)
        
        if not recommendations:
            print("No recommendations found above minimum relevance threshold.")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   Types: {', '.join(rec['test_type'])}")
            print(f"   Relevance: {rec['relevance_score']*100:.2f}%")
            print(f"   Raw Score: {rec['raw_score']:.4f}")
            print(f"   URL: {rec['assessment_url']}")


if __name__ == "__main__":
    main()