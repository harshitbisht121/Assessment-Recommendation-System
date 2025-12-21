"""
FastAPI Application for SHL Assessment Recommendations
Implements required endpoints as per specification
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommendation.recommender import SHLRecommender

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender (singleton)
recommender = None

@app.on_event("startup")
async def startup_event():
    """Load recommender on startup"""
    global recommender
    try:
        print("Loading recommender system...")
        recommender = SHLRecommender(
            data_dir="data/processed",
            model_name='all-MiniLM-L6-v2'
        )
        print("✓ Recommender system loaded successfully")
        
        # Test recommendation to catch any issues early
        try:
            test_rec = recommender.recommend("test query", top_k=1, enhance_query=False)
            print(f"✓ Test recommendation successful: {len(test_rec)} results")
        except Exception as e:
            print(f"⚠ Warning: Test recommendation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Error loading recommender: {e}")
        import traceback
        traceback.print_exc()
        raise


# Request/Response Models
class RecommendRequest(BaseModel):
    """Request model for recommendation endpoint"""
    query: str = Field(..., description="Job description or natural language query", min_length=1)
    top_k: Optional[int] = Field(10, description="Number of recommendations (1-10)", ge=1, le=10)
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class AssessmentRecommendation(BaseModel):
    """Single assessment recommendation"""
    assessment_name: str
    assessment_url: str
    description: Optional[str] = ""
    test_type: List[str]
    relevance_score: float


class RecommendResponse(BaseModel):
    """Response model for recommendation endpoint"""
    query: str
    top_k: int
    recommendations: List[AssessmentRecommendation]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running
    
    Returns:
        Status indicating API health
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend_assessments(request: RecommendRequest):
    """
    Recommend SHL assessments based on job description or query
    
    Args:
        request: RecommendRequest with query and top_k
        
    Returns:
        RecommendResponse with list of recommended assessments
    
    Example:
        ```json
        {
            "query": "I need Java developers who collaborate well",
            "top_k": 10
        }
        ```
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender system not initialized"
        )
    
    try:
        # Get recommendations
        recommendations = recommender.recommend(
            query=request.query,
            top_k=request.top_k,
            enhance_query=False,  # Disable LLM enhancement to avoid errors
            balance=True
        )
        
        # Format response
        response = {
            "query": request.query,
            "top_k": request.top_k,
            "recommendations": recommendations
        }
        
        return response
    
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"Error in recommend endpoint:")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }


# Error handlers
from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": ["/health", "/recommend", "/docs"]
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )