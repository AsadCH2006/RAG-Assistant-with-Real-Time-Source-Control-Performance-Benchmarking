from typing import List, Dict, Any

class PerformanceEvaluator:
    @staticmethod
    def evaluate_retrieval_quality(sources: List[Dict[str, Any]]) -> Dict[str, float]:
        if not sources:
            return {"avg_distance": 0.0, "best_match_score": 0.0}
        
        scores = [src["score"] for src in sources]
        avg_score = sum(scores) / len(scores)
        best_score = min(scores)  # In ChromaDB, lower distance = higher similarity
        
        return {
            "avg_distance": round(avg_score, 4),
            "best_match_score": round(best_score, 4)
        }