"""
Test the complete recommendation system before deployment
Compatible with new embeddings structure (npy + csv)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_embeddings():
    print("\n" + "="*70)
    print("TEST 1: Loading Embeddings")
    print("="*70)

    try:
        import numpy as np
        import pandas as pd

        emb_path = "data/processed/embeddings.npy"
        meta_path = "data/processed/metadata.csv"

        embeddings = np.load(emb_path)
        metadata = pd.read_csv(meta_path)

        print("✓ Embeddings loaded successfully")
        print(f"  - {len(embeddings)} assessments")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print(f"  - Metadata columns: {list(metadata.columns)}")

        required_cols = ["name", "url", "test_type"]

        for col in required_cols:
            if col in metadata.columns:
                print(f"  ✓ Column '{col}' present")
            else:
                print(f"  ✗ Column '{col}' MISSING")
                return False

        return True

    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        return False


def test_recommender():
    print("\n" + "="*70)
    print("TEST 2: Recommender System")
    print("="*70)

    try:
        from recommendation.recommender import SHLRecommender

        # IMPORTANT — pass folder, not file
        recommender = SHLRecommender("data/processed")

        query = "I need a Java developer with problem-solving skills"
        print(f"\nTest query: '{query}'")

        recommendations = recommender.recommend(query, top_k=3)

        if not recommendations:
            print("✗ No recommendations returned")
            return False

        print(f"\n✓ Got {len(recommendations)} recommendations:")

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   Score: {rec['relevance_score']}")
            print(f"   Types: {rec['test_type']}")
            print(f"   URL: {rec['assessment_url'][:60]}...")

        return True

    except Exception as e:
        print(f"✗ Error in recommender: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    print("\n" + "="*70)
    print("TEST 3: API Server")
    print("="*70)

    try:
        from api.app import app
    except ImportError:
        print("✗ Could not import API app")
        return False

    try:
        from httpx import AsyncClient, ASGITransport
        from asgi_lifespan import LifespanManager
        import asyncio

        print("\nTesting /health endpoint...")

        async def test_health():
            async with LifespanManager(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(base_url="http://test", transport=transport) as client:
                    return await client.get("/health")

        response = asyncio.run(test_health())
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")

        if response.status_code != 200:
            print("✗ Health check failed")
            return False

        print("\nTesting /recommend endpoint...")
        test_request = {
            "query": "Python developer with data analysis skills",
            "top_k": 5
        }

        async def test_recommend():
            async with LifespanManager(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(base_url="http://test", transport=transport) as client:
                    return await client.post("/recommend", json=test_request)

        response = asyncio.run(test_recommend())
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Got {len(data['recommendations'])} recommendations")
            print(f"  First recommendation: {data['recommendations'][0]['assessment_name']}")
            return True
        else:
            print(f"✗ API request failed: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error testing API: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("SHL RECOMMENDATION SYSTEM - PRE-DEPLOYMENT TESTS")
    print("="*70)

    results = {
        "Embeddings": test_embeddings(),
        "Recommender": test_recommender(),
        "API": test_api()
    }

    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for deployment!")
        print("="*70)
    else:
        print("❌ SOME TESTS FAILED - Fix issues before deploying")
        print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
