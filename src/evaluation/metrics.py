"""
Evaluation Metrics for SHL Recommendation System
Implements Mean Recall@K and other evaluation metrics
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.recommendation.recommender import SHLRecommender


# ===============================
# 🔥 KEY FIX: Compare by SLUG
# ===============================
def get_slug(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.strip().lower()
    if url.endswith("/"):
        url = url[:-1]
    return url.split("/")[-1]


def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    top_k = [get_slug(u) for u in top_k]
    relevant = [get_slug(u) for u in relevant]

    matches = len(set(top_k) & set(relevant))
    return matches / len(relevant)


def mean_recall_at_k(predictions: Dict[str, List[str]],
                     ground_truth: Dict[str, List[str]],
                     k: int) -> float:
    recalls = []
    for query in ground_truth.keys():
        if query in predictions:
            recall = recall_at_k(predictions[query], ground_truth[query], k)
            recalls.append(recall)
        else:
            recalls.append(0.0)
    return np.mean(recalls)


def evaluate_recommender(recommender: SHLRecommender,
                         test_data_path: str,
                         k_values: List[int] = [5, 10]) -> Dict[str, float]:

    df = pd.read_csv(test_data_path)
    url_col = 'url' if 'url' in df.columns else 'assessment_url'

    # Ground truth
    ground_truth = {}
    for query, group in df.groupby('query'):
        ground_truth[query] = group[url_col].tolist()

    predictions = {}
    print(f"\nGenerating predictions for {len(ground_truth)} queries...")

    for i, query in enumerate(ground_truth.keys(), 1):
        print(f"  [{i}/{len(ground_truth)}] {query[:60]}...")

        # ✅ FORCE enhanced recommender
        recs = recommender.recommend(
            query=query,
            top_k=max(k_values),
            enhance_query=True,
            balance=True
        )

        predictions[query] = [r['assessment_url'] for r in recs]

    # ============================
    # RESULTS
    # ============================
    results = {}
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")

    for k in k_values:
        mrk = mean_recall_at_k(predictions, ground_truth, k)
        results[f'mean_recall@{k}'] = mrk
        print(f"Mean Recall@{k}: {mrk:.4f}")

    # ============================
    # PER QUERY ANALYSIS
    # ============================
    per_query_recalls = {}
    for query in ground_truth.keys():
        per_query_recalls[query] = recall_at_k(
            predictions[query],
            ground_truth[query],
            10
        )

    sorted_queries = sorted(
        per_query_recalls.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'='*70}")
    print("PER-QUERY ANALYSIS")
    print(f"{'='*70}")

    print("\nBest performing queries:")
    for query, recall in sorted_queries[:3]:
        print(f"  Recall@10: {recall:.4f} - {query[:60]}...")

    print("\nWorst performing queries:")
    for query, recall in sorted_queries[-3:]:
        print(f"  Recall@10: {recall:.4f} - {query[:60]}...")

    results['per_query_recalls'] = per_query_recalls
    return results


def generate_submission_csv(recommender: SHLRecommender,
                             test_queries_path: str,
                             output_path: str = "predictions.csv"):

    df = pd.read_csv(test_queries_path)
    queries = df['query'].tolist()

    print(f"\nGenerating predictions for {len(queries)} test queries...")

    submission_data = []

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:60]}...")

        recs = recommender.recommend(
            query=query,
            top_k=10,
            enhance_query=True,
            balance=True
        )

        for rec in recs:
            submission_data.append({
                'query': query,
                'assessment_url': rec['assessment_url']
            })

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved predictions to: {output_path}")
    print(f"  Total rows: {len(submission_df)}")
    print("  Format: query, assessment_url")

    return submission_df


def main():
    print("Loading recommender system...")
    recommender = SHLRecommender(
        data_dir="data/processed",
        model_name='all-MiniLM-L6-v2'
    )

    train_path = "data/datasets/train.csv"
    if Path(train_path).exists():
        df_check = pd.read_csv(train_path)
        if len(df_check) > 0:
            print(f"\n{'='*70}")
            print("EVALUATING ON TRAIN SET")
            print(f"{'='*70}")
            evaluate_recommender(recommender, train_path, k_values=[5, 10])

    test_path = "data/datasets/test.csv"
    if Path(test_path).exists():
        df_check = pd.read_csv(test_path)
        if len(df_check) > 0:
            print(f"\n{'='*70}")
            print("GENERATING TEST PREDICTIONS")
            print(f"{'='*70}")
            generate_submission_csv(recommender, test_path, "predictions.csv")


if __name__ == "__main__":
    main()
