"""
Embedding Generation for SHL Assessments
Uses Sentence Transformers to create embeddings for semantic search
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm

class AssessmentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedder with sentence transformer model
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.metadata = None
    
    def create_assessment_text(self, row):
        """
        Create rich text representation of assessment for embedding
        
        Args:
            row: DataFrame row with assessment data
            
        Returns:
            Combined text representation
        """
        parts = []
        
        # Assessment name (most important) - handle both column names
        name_col = 'name' if 'name' in row.index else 'assessment_name'
        if pd.notna(row.get(name_col)):
            parts.append(f"Assessment: {row[name_col]}")
        
        # Description
        if pd.notna(row.get('description')) and row['description']:
            parts.append(f"Description: {row['description']}")
        
        # Test types (expanded with full names)
        test_type_map = {
            'A': 'Ability and Aptitude',
            'B': 'Biodata and Situational Judgement',
            'C': 'Competencies',
            'D': 'Development and 360',
            'E': 'Assessment Exercises',
            'K': 'Knowledge and Skills',
            'P': 'Personality and Behavior',
            'S': 'Simulations'
        }
        
        if pd.notna(row.get('test_type')):
            test_types = eval(row['test_type']) if isinstance(row['test_type'], str) else row['test_type']
            expanded_types = [test_type_map.get(t, t) for t in test_types]
            parts.append(f"Test Types: {', '.join(expanded_types)}")
        
        # Duration
        if pd.notna(row.get('duration')):
            parts.append(f"Duration: {row['duration']} minutes")
        
        return " | ".join(parts)
    
    def generate_embeddings(self, df, text_column=None):
        """
        Generate embeddings for all assessments
        
        Args:
            df: DataFrame with assessment data
            text_column: Column to embed (if None, creates combined text)
            
        Returns:
            numpy array of embeddings
        """
        print("\nGenerating embeddings...")
        
        # Create text representations
        if text_column is None:
            texts = [self.create_assessment_text(row) for _, row in df.iterrows()]
        else:
            texts = df[text_column].tolist()
        
        # Generate embeddings with progress bar
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings)
        self.metadata = df
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
        
        return self.embeddings
    
    def save_embeddings(self, output_dir="data/processed"):
        """Save embeddings and metadata"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_path = Path(output_dir) / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        
        # Save metadata
        metadata_path = Path(output_dir) / "metadata.csv"
        self.metadata.to_csv(metadata_path, index=False)
        
        # Save model info
        model_info = {
            'model_name': self.model.get_sentence_embedding_dimension(),
            'num_assessments': len(self.embeddings)
        }
        
        info_path = Path(output_dir) / "model_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"\n✓ Saved embeddings to: {embeddings_path}")
        print(f"✓ Saved metadata to: {metadata_path}")
    
    @classmethod
    def load_embeddings(cls, model_name='all-MiniLM-L6-v2', data_dir="data/processed"):
        """Load pre-computed embeddings"""
        embedder = cls(model_name=model_name)
        
        embeddings_path = Path(data_dir) / "embeddings.npy"
        metadata_path = Path(data_dir) / "metadata.csv"
        
        embedder.embeddings = np.load(embeddings_path)
        embedder.metadata = pd.read_csv(metadata_path)
        
        print(f"✓ Loaded {len(embedder.embeddings)} embeddings")
        
        return embedder


def main():
    """Generate embeddings from scraped data"""
    # Load scraped data
    data_path = "data/raw/shl_catalog.csv"
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} assessments")
    
    # Create embedder and generate embeddings
    embedder = AssessmentEmbedder(model_name='all-MiniLM-L6-v2')
    embeddings = embedder.generate_embeddings(df)
    
    # Save embeddings
    embedder.save_embeddings()
    
    print("\n" + "="*70)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()