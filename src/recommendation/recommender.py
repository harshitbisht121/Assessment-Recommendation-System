"""
SHL Assessment Recommendation Engine
Uses semantic search with FAISS and LLM-based re-ranking
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
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
                        self.llm = genai.GenerativeModel('gemini-2.5-flash')
                    except:
                        try:
                            self.llm = genai.GenerativeModel('gemini-2.5-flash')
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
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[int]:
        """
        Perform semantic search using FAISS
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of indices of top matches
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )
        
        return indices[0].tolist(), scores[0].tolist()
    
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
        balance: bool = True
    ) -> List[Dict]:
        """
        Main recommendation function
        
        Args:
            query: User query or job description
            top_k: Number of recommendations (1-10)
            enhance_query: Whether to use LLM for query enhancement
            balance: Whether to balance results across test types
            
        Returns:
            List of recommendation dictionaries
        """
        # Validate top_k
        top_k = max(1, min(10, top_k))
        
        # Enhance query if enabled
        search_query = self.enhance_query(query) if enhance_query else query
        
        # Extract requirements for balancing
        requirements = self.extract_requirements(query)
        
        # Semantic search
        indices, scores = self.semantic_search(search_query, top_k=20)
        
        # Balance recommendations if needed
        if balance and requirements['test_types']:
            indices = self.balance_recommendations(indices, scores, requirements, top_k)
        else:
            indices = indices[:top_k]
        
        # Format results
        recommendations = []
        for idx in indices:
            row = self.metadata.iloc[idx]
            
            # Handle test_type column
            test_types = eval(row['test_type']) \
                        if isinstance(row['test_type'], str) \
                        else row['test_type']
            
            recommendations.append({
                'assessment_name': row['name'],  # Use 'name' column
                'assessment_url': row['url'],    # Use 'url' column
                'description': row.get('description', ''),
                'test_type': test_types,
                'relevance_score': float(scores[indices.index(idx)]) if idx in indices[:len(scores)] else 0.0
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
        "Need cognitive and personality tests for analyst position"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        recommendations = recommender.recommend(query, top_k=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   Types: {', '.join(rec['test_type'])}")
            print(f"   Score: {rec['relevance_score']:.4f}")
            print(f"   URL: {rec['assessment_url']}")


if __name__ == "__main__":
    main()