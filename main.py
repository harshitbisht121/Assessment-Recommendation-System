#!/usr/bin/env python3
"""
Railway deployment entry point for SHL Recommendation System
"""
import os
import uvicorn
from src.api.app import app

if __name__ == "__main__":
    # Get port from environment (Railway provides PORT)
    port = int(os.environ.get("PORT", 8080))

    # Run the FastAPI app
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
