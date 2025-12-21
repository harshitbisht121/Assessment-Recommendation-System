"""
Streamlit Frontend for SHL Assessment Recommendation System
"""
import streamlit as st
import requests
import pandas as pd
from typing import List, Dict
import sys
from pathlib import Path

# Page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .test-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .type-K { background-color: #e3f2fd; color: #1976d2; }
    .type-P { background-color: #f3e5f5; color: #7b1fa2; }
    .type-A { background-color: #e8f5e9; color: #388e3c; }
    .type-S { background-color: #fff3e0; color: #f57c00; }
    .type-B { background-color: #fce4ec; color: #c2185b; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Test type descriptions
TEST_TYPE_INFO = {
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'A': 'Ability & Aptitude',
    'S': 'Simulations',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises'
}


def check_api_health() -> bool:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_recommendations(query: str, top_k: int = 10) -> Dict:
    """Get recommendations from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None


def display_recommendation(rec: Dict, index: int):
    """Display a single recommendation"""
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {index}. {rec['assessment_name']}")
            
            # Test type badges
            test_types_html = ""
            for test_type in rec.get('test_type', []):
                full_name = TEST_TYPE_INFO.get(test_type, test_type)
                test_types_html += f'<span class="test-type-badge type-{test_type}">{full_name}</span>'
            st.markdown(test_types_html, unsafe_allow_html=True)
            
            # Description
            if rec.get('description'):
                st.markdown(f"**Description:** {rec['description'][:300]}...")
            
            # URL
            st.markdown(f"🔗 [View Assessment]({rec['assessment_url']})")
        
        with col2:
            # Relevance score
            score = rec.get('relevance_score', 0)
            st.metric("Relevance", f"{score:.2%}")
        
        st.markdown("---")


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🎯 SHL Assessment Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find the perfect assessments for your hiring needs</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 About")
        st.markdown("""
        This system helps you find relevant SHL assessments based on:
        - Natural language queries
        - Job descriptions
        - Role requirements
        
        **Features:**
        - Semantic search across 377+ assessments
        - Intelligent balancing of technical & soft skills
        - LLM-powered query enhancement
        """)
        
        st.markdown("---")
        
        # API Status
        st.markdown("## 🔌 API Status")
        if check_api_health():
            st.success("✓ Connected")
        else:
            st.error("✗ API Unavailable")
            st.info(f"Expected API at: {API_BASE_URL}")
        
        st.markdown("---")
        
        # Settings
        st.markdown("## ⚙️ Settings")
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=10
        )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📝 Examples", "ℹ️ Help"])
    
    with tab1:
        # Query input
        st.markdown("### Enter your query or job description")
        
        input_method = st.radio(
            "Input method:",
            ["Text Query", "Job Description Text", "URL (Coming Soon)"],
            horizontal=True
        )
        
        if input_method == "Text Query":
            query = st.text_area(
                "Enter your query:",
                placeholder="e.g., I need Java developers who can collaborate with business teams",
                height=100
            )
        elif input_method == "Job Description Text":
            query = st.text_area(
                "Paste job description:",
                placeholder="Paste the complete job description here...",
                height=200
            )
        else:
            st.info("URL input feature coming soon!")
            query = ""
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        # Search and display results
        if search_button and query:
            with st.spinner("🤖 Finding best assessments..."):
                results = get_recommendations(query, num_recommendations)
            
            if results and results.get('recommendations'):
                st.success(f"✓ Found {len(results['recommendations'])} recommendations")
                
                # Display recommendations
                st.markdown("---")
                st.markdown("## 📋 Recommendations")
                
                for i, rec in enumerate(results['recommendations'], 1):
                    display_recommendation(rec, i)
                
                # Export options
                st.markdown("---")
                st.markdown("### 💾 Export Results")
                
                # Create DataFrame for export
                export_data = []
                for rec in results['recommendations']:
                    export_data.append({
                        'Assessment Name': rec['assessment_name'],
                        'URL': rec['assessment_url'],
                        'Test Types': ', '.join(rec.get('test_type', [])),
                        'Relevance Score': rec.get('relevance_score', 0)
                    })
                
                df = pd.DataFrame(export_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "shl_recommendations.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col2:
                    st.dataframe(df, use_container_width=True)
            
            elif results:
                st.warning("No recommendations found. Try a different query.")
            else:
                st.error("Failed to get recommendations. Please check API connection.")
    
    with tab2:
        st.markdown("### 📝 Example Queries")
        st.markdown("""
        Try these sample queries to see the system in action:
        """)
        
        examples = [
            "I need Java developers who can collaborate with business teams",
            "Looking for mid-level professionals proficient in Python, SQL and JavaScript",
            "Need cognitive and personality tests for analyst position",
            "Hiring customer service representatives with strong communication skills",
            "Looking for leadership assessments for manager-level candidates"
        ]
        
        for example in examples:
            if st.button(f"📌 {example}", use_container_width=True):
                st.session_state['example_query'] = example
                st.rerun()
    
    with tab3:
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. **Enter your query** in natural language or paste a job description
        2. **Click Search** to get recommendations
        3. **Review results** ranked by relevance
        4. **Export** recommendations as CSV if needed
        
        ### 🎯 Tips for Better Results
        - Be specific about technical skills (e.g., "Java", "Python")
        - Mention soft skills (e.g., "collaboration", "leadership")
        - Include role level (e.g., "senior", "mid-level", "entry-level")
        - Specify assessment types if needed (e.g., "cognitive", "personality")
        
        ### 🏷️ Test Type Legend
        - **K**: Knowledge & Skills - Technical assessments
        - **P**: Personality & Behavior - Soft skills & traits
        - **A**: Ability & Aptitude - Cognitive tests
        - **S**: Simulations - Job simulations
        - **B**: Biodata & SJT - Situational judgment
        """)


if __name__ == "__main__":
    main()