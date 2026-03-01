from typing import Dict, Any, List
from rompiche.core.evaluator import Evaluator

def evaluate_all_results(
    results: List[Dict[str, Any]], evaluator: Evaluator
) -> Dict[str, Dict[str, float]]:
    """Evaluate all results and return average metrics per field.

    Returns: {field_name: {metric_name: avg_score}}
    """
    totals: Dict[str, Dict[str, float]] = {}
    count = len(results)

    for result in results:
        per_field = evaluator.evaluate(result["prediction"], result["ground_truth"])
        for field_name, field_metrics in per_field.items():
            if field_name not in totals:
                totals[field_name] = {}
            for metric_name, value in field_metrics.items():
                totals[field_name][metric_name] = (
                    totals[field_name].get(metric_name, 0.0) + value
                )

    averages: Dict[str, Dict[str, float]] = {}
    for field_name, field_totals in totals.items():
        averages[field_name] = {
            metric: round(total / count, 4) for metric, total in field_totals.items()
        }

    return averages

def collect_mismatch_examples(
    results: List[Dict[str, Any]], evaluator: Evaluator, max_examples: int = 10
) -> List[Dict[str, Any]]:
    """Collect examples where prediction differs from ground truth or has low scores."""
    # First, collect exact mismatches (up to max_examples)
    exact_mismatches = []
    scored_results = []

    for result in results:
        if result["prediction"] != result["ground_truth"]:
            exact_mismatches.append(
                {
                    "input": result["input"],
                    "prediction": result["prediction"],
                    "ground_truth": result["ground_truth"],
                    "type": "exact_mismatch",
                }
            )
            if len(exact_mismatches) >= max_examples:
                return exact_mismatches

        # Also collect all results with their scores for potential low-scoring examples
        evaluation = evaluator.evaluate(result["prediction"], result["ground_truth"])
        scored_results.append(
            {
                "input": result["input"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"],
                "scores": evaluation,
                "type": "low_score",
            }
        )

    # If we have exact mismatches, return them
    if exact_mismatches:
        return exact_mismatches[:max_examples]

    # Otherwise, find the worst scoring examples
    # Calculate overall score for each result (average of all metric scores)
    for item in scored_results:
        all_scores = []
        for field_scores in item["scores"].values():
            all_scores.extend(field_scores.values())
        item["overall_score"] = sum(all_scores) / len(all_scores) if all_scores else 0

    # Sort by overall score (ascending - worst first) and return top max_examples
    scored_results.sort(key=lambda x: x["overall_score"])
    return scored_results[:max_examples]