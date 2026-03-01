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
    results: List[Dict[str, Any]], evaluator: Evaluator, max_examples_per_field: int = 5
) -> List[Dict[str, Any]]:
    """Collect examples where prediction differs from ground truth or has low scores.
    
    Args:
        results: List of processing results with prediction and ground_truth
        evaluator: Evaluator instance
        max_examples_per_field: Maximum number of mismatch examples to collect per field
        
    Returns:
        List of mismatch examples (up to max_examples_per_field * number_of_fields)
    """
    # Collect all mismatches per field
    field_mismatches = {}
    
    for result in results:
        prediction = result["prediction"]
        ground_truth = result["ground_truth"]
        
        # Check each field for mismatches
        for field_name in ground_truth.keys():
            if field_name not in field_mismatches:
                field_mismatches[field_name] = []
            
            # Check if this field has a mismatch
            if prediction.get(field_name) != ground_truth.get(field_name):
                # Evaluate this specific field
                field_evaluation = evaluator.evaluate(prediction, ground_truth)
                field_score = field_evaluation[field_name]
                
                # Add to field mismatches if we haven't reached the limit
                if len(field_mismatches[field_name]) < max_examples_per_field:
                    field_mismatches[field_name].append(
                        {
                            "input": result["input"],
                            "prediction": prediction,
                            "ground_truth": ground_truth,
                            "field": field_name,
                            "field_score": field_score,
                            "type": "field_mismatch",
                        }
                    )
        
        # Early exit if we've collected enough examples
        total_mismatches = sum(len(m) for m in field_mismatches.values())
        if total_mismatches >= 15:
            break

    # Flatten the field mismatches into a single list
    all_mismatches = []
    for field_name, mismatches in field_mismatches.items():
        all_mismatches.extend(mismatches)
    
    return all_mismatches