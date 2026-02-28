import json
import difflib
import os
from typing import Dict, Any, Optional, Union
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Evaluator:
    """
    Evaluator class to compare JSON output to ground truth using various metrics.
    
    Supported metrics:
    - exact_match: Binary 0 or 1 for exact match
    - string_distance: Computed string distance (Levenshtein ratio)
    - llm_judge: LLM-based evaluation with customizable prompt and model
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator with optional config.
        
        Args:
            config: Configuration dictionary containing:
                - metrics: List of metrics to use
                - llm_config: Configuration for LLM judge (model, prompt_template, etc.)
        """
        self.config = config or {}
        self.metrics = self.config.get('metrics', ['exact_match'])
        self.llm_config = self.config.get('llm_config', {})
    
    def evaluate(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate prediction against ground truth using configured metrics.
        
        Args:
            prediction: Predicted JSON output
            ground_truth: Ground truth JSON output
            
        Returns:
            Dictionary with evaluation results for each metric
        """
        results = {}
        
        for metric in self.metrics:
            if metric == 'exact_match':
                results[metric] = self._exact_match(prediction, ground_truth)
            elif metric == 'string_distance':
                results[metric] = self._string_distance(prediction, ground_truth)
            elif metric == 'llm_judge':
                results[metric] = self._llm_judge(prediction, ground_truth)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
        return results
    
    def _exact_match(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> int:
        """
        Exact match metric: 1 if identical, 0 otherwise.
        """
        return 1 if prediction == ground_truth else 0
    
    def _string_distance(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        String distance metric using Levenshtein ratio.
        Returns similarity score between 0 and 1.
        """
        pred_str = json.dumps(prediction, sort_keys=True)
        truth_str = json.dumps(ground_truth, sort_keys=True)
        
        similarity = difflib.SequenceMatcher(None, pred_str, truth_str).ratio()
        return round(similarity, 4)
    
    def _llm_judge(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Union[float, str]:
        """
        LLM judge metric using Mistral API.
        
        Requires llm_config with:
        - model: LLM model to use
        - prompt_template: Template for evaluation prompt
        - api_key: Mistral API key (or set via MISTRAL_API_KEY environment variable)
        """
        if not self.llm_config:
            raise ValueError("LLM judge requires llm_config in evaluator configuration")
            
        model = self.llm_config.get('model', 'mistral-tiny')
        api_key = self.llm_config.get('api_key', os.getenv('MISTRAL_API_KEY'))
        
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in config or environment")
            
        prompt_template = self.llm_config.get('prompt_template', 
            "Evaluate the following prediction against ground truth:\n\n"
            "Prediction: {prediction}\n"
            "Ground Truth: {ground_truth}\n"
            "Respond with only a score between 0 and 1 where 1 means perfect match.")
            
        # Format prompt
        prompt = prompt_template.format(
            prediction=json.dumps(prediction, indent=2),
            ground_truth=json.dumps(ground_truth, indent=2)
        )
        
        try:
            # Initialize Mistral client
            client = MistralClient(api_key=api_key)
            
            # Create chat completion
            chat_response = client.chat(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)]
            )
            
            # Extract and parse the score from LLM response
            response_text = chat_response.choices[0].message.content
            
            # Try to extract a numerical score from the response
            # This is a simple approach - you might need more sophisticated parsing
            try:
                score = float(response_text.strip())
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                # If we can't parse as float, return the raw response
                return response_text
            
        except Exception as e:
            raise RuntimeError(f"LLM evaluation failed: {str(e)}")

# Example configuration
def get_example_config() -> Dict[str, Any]:
    """Return example evaluator configuration."""
    return {
        "metrics": ["exact_match", "string_distance"],
        "llm_config": {
            "model": "mistral-tiny",
            "prompt_template": "Evaluate prediction vs ground truth:\nPrediction: {prediction}\nGround Truth: {ground_truth}\nScore (0-1):",
            "api_key": os.getenv('MISTRAL_API_KEY')  # Will use .env file
        }
    }

if __name__ == "__main__":
    # Example usage
    config = get_example_config()
    evaluator = Evaluator(config)
    
    # Example data
    prediction = {
        "date": "2026-03-11",
        "time": "09:52:00",
        "title": "Code Review"
    }
    
    ground_truth = {
        "date": "2026-03-11",
        "time": "09:52:00", 
        "title": "Code Review Session"
    }
    
    results = evaluator.evaluate(prediction, ground_truth)
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score}")