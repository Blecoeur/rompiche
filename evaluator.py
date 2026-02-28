import json
import difflib
from typing import Dict, Any, Optional, List


class Evaluator:
    """
    Evaluator that computes metrics per field, comparing prediction to ground truth.
    
    Supported metrics:
    - exact_match: Binary 0 or 1 — does the field value match ground truth exactly?
    - string_distance: Similarity ratio (0-1) between predicted and expected field values.
    
    Returns a nested dict: {field_name: {metric_name: score}}
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = self.config.get('metrics', ['exact_match', 'string_distance'])
        self.field_metrics = self.config.get('field_metrics', {})
        self.success_thresholds = self.config.get('success_thresholds', {})
    
    def evaluate(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate prediction against ground truth, per field.
        
        Returns:
            {field_name: {metric_name: score}} for every field in ground_truth.
        """
        results: Dict[str, Dict[str, float]] = {}
        
        for field_name, truth_value in ground_truth.items():
            pred_value = prediction.get(field_name)
            field_results: Dict[str, float] = {}
            
            # Use field-specific metrics if defined, otherwise use default metrics
            metrics_to_use = self.field_metrics.get(field_name, self.metrics)
            
            for metric in metrics_to_use:
                if metric == 'exact_match':
                    field_results[metric] = 1.0 if pred_value == truth_value else 0.0
                elif metric == 'string_distance':
                    field_results[metric] = self._string_distance(
                        str(pred_value) if pred_value is not None else "",
                        str(truth_value)
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            
            results[field_name] = field_results
        
        return results
    
    @staticmethod
    def _string_distance(pred: str, truth: str) -> float:
        return round(difflib.SequenceMatcher(None, pred, truth).ratio(), 4)

    def is_success(self, evaluation_results: Dict[str, Dict[str, float]]) -> bool:
        """
        Determine if the evaluation meets success criteria.
        
        Args:
            evaluation_results: Results from evaluate() method
            
        Returns:
            True if all fields meet their success thresholds, False otherwise
        """
        for field_name, metrics in evaluation_results.items():
            # Get thresholds for this field (default: exact_match=1.0, string_distance=1.0)
            field_thresholds = self.success_thresholds.get(field_name, {})
            
            for metric_name, score in metrics.items():
                # Get threshold for this metric (default to 1.0 for exact_match, 0.8 for string_distance)
                if metric_name == 'exact_match':
                    threshold = field_thresholds.get(metric_name, 1.0)
                else:  # string_distance
                    threshold = field_thresholds.get(metric_name, 0.8)
                
                if score < threshold:
                    return False
        
        return True


if __name__ == "__main__":
    # Example with field-specific metrics and success thresholds
    evaluator = Evaluator({
        "metrics": ["exact_match", "string_distance"],
        "field_metrics": {
            "date": ["exact_match"],
            "time": ["exact_match"],
            "title": ["string_distance"]
        },
        "success_thresholds": {
            "date": {"exact_match": 1.0},
            "time": {"exact_match": 1.0},
            "title": {"string_distance": 0.7}
        }
    })
    
    prediction = {
        "date": "2026-03-11",
        "time": "09:52",
        "title": "Code Review"
    }
    ground_truth = {
        "date": "2026-03-11",
        "time": "09:52:00", 
        "title": "Code Review Session"
    }
    
    results = evaluator.evaluate(prediction, ground_truth)
    print("Per-field evaluation with field-specific metrics:")
    for field, metrics in results.items():
        print(f"  {field}: {metrics}")
    
    success = evaluator.is_success(results)
    print(f"\nOverall success: {success}")