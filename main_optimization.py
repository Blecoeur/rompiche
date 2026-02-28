#!/usr/bin/env python3
"""
Main optimization workflow that ties everything together:
1. Processes all input files using example_processor.py
2. Evaluates results using evaluator.py
3. Uses brain.py to decide whether to continue or stop
4. Applies suggestions using applier.py
5. Repeats until success criteria are met or max iterations reached
"""

import json
import os
from typing import Dict, Any, List
from evaluator import Evaluator
from example_processor import process
from brain import get_brain_decision
from tqdm import tqdm

def load_data(file_path: str = "example_data/example.jsonl") -> List[Dict[str, Any]]:
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_script_on_all_files(prompt: str, schema: Dict[str, Any], data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the processor on all input data"""
    results = []
    for index, item in tqdm(enumerate(data), desc="Running the processor on all files"):
        if index > 10:
            break
        input_text = item["input"]["text"]
        prediction = process(input_text, prompt, schema)
        ground_truth = item["results"]
        results.append({
            "input": input_text,
            "prediction": prediction,
            "ground_truth": ground_truth
        })
    return results

def collect_mismatch_examples(results: List[Dict[str, Any]], max_examples: int = 5) -> List[Dict[str, Any]]:
    """Collect examples where prediction differs from ground truth."""
    mismatches = []
    for result in results:
        if result["prediction"] != result["ground_truth"]:
            mismatches.append({
                "input": result["input"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"]
            })
        if len(mismatches) >= max_examples:
            break
    return mismatches

def evaluate_all_results(results: List[Dict[str, Any]], evaluator: Evaluator) -> Dict[str, Dict[str, float]]:
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
                totals[field_name][metric_name] = totals[field_name].get(metric_name, 0.0) + value
    
    averages: Dict[str, Dict[str, float]] = {}
    for field_name, field_totals in totals.items():
        averages[field_name] = {
            metric: round(total / count, 4) for metric, total in field_totals.items()
        }
    
    return averages


def check_success_criteria(
    metrics: Dict[str, Dict[str, float]],
    criteria: Dict[str, Dict[str, float]]
) -> bool:
    """Check if all per-field metrics meet their success criteria."""
    for field_name, field_criteria in criteria.items():
        field_metrics = metrics.get(field_name, {})
        for metric_name, target in field_criteria.items():
            if field_metrics.get(metric_name, 0.0) < target:
                return False
    return True

def run_full_optimization_loop(max_iterations: int = 5):
    """Main optimization loop"""
    
    # Initialize
    current_prompt = "Extract the date, time, and event title from the text."
    current_schema = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "The date of the event"},
            "time": {"type": "string", "description": "The time of the event"},
            "title": {"type": "string", "description": "The title of the event"}
        },
        "required": ["date", "time", "title"]
    }
    # Per-field success criteria (now handled by evaluator)
    success_criteria = {
        "date": {"exact_match": 0.95, "string_distance": 0.95},
        "time": {"exact_match": 0.95, "string_distance": 0.95},
        "title": {"exact_match": 0.90, "string_distance": 0.90},
    }
    
    # Load evaluation data
    data = load_data()
    
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
            "title": {"string_distance": 0.8}
        }
    })
    
    print("Starting optimization loop...")
    print(f"Initial prompt: {current_prompt}")
    print(f"Initial schema: {json.dumps(current_schema, indent=2)}")
    print(f"Success criteria: {success_criteria}")
    
    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print(f"{'='*50}")
        
        # 1. Run processor on all data
        print("Running processor on all files...")
        processing_results = run_script_on_all_files(current_prompt, current_schema, data)
        
        # 2. Evaluate results
        print("Evaluating results...")
        metrics = evaluate_all_results(processing_results, evaluator)
        print("Metrics:")
        for field, field_metrics in metrics.items():
            print(f"  {field}: {field_metrics}")
        
        # 3. Check if we meet success criteria using evaluator
        criteria_met = evaluator.is_success(metrics)
        
        if criteria_met:
            print("🎉 Success criteria met! Stopping optimization.")
            break
        
        # 4. Collect mismatch examples for the brain
        mismatch_examples = collect_mismatch_examples(processing_results)
        print(f"Found {len(mismatch_examples)} mismatch examples")
        
        # 5. Ask the brain for decision with mismatch context
        print("Consulting optimization brain...")
        try:
            decision = get_brain_decision(
                metrics, current_prompt, current_schema,
                success_criteria, mismatch_examples
            )
            print(f"Brain decision: {json.dumps(decision, indent=2)}")
            
            if decision["decision"] == "stop":
                print("Brain decided to stop optimization.")
                break
            
            # 6. Use the brain's updated prompt and schema directly
            if "updated_prompt" in decision:
                current_prompt = decision["updated_prompt"]
            if "updated_schema" in decision:
                current_schema = decision["updated_schema"]
            print(f"Updated prompt: {current_prompt}")
            print(f"Updated schema: {json.dumps(current_schema, indent=2)}")
            
        except Exception as e:
            print(f"Error consulting brain: {e}")
            print("Continuing with current prompt and schema...")
    
    print(f"\n{'='*50}")
    print("Optimization complete!")
    print("Final metrics:")
    for field, field_metrics in metrics.items():
        print(f"  {field}: {field_metrics}")
    print(f"Final prompt: {current_prompt}")
    print(f"Final schema: {json.dumps(current_schema, indent=2)}")
    
    # Return final results
    return {
        "metrics": metrics,
        "prompt": current_prompt,
        "schema": current_schema,
        "iteration": iteration + 1
    }

if __name__ == "__main__":
    # Run the optimization with a maximum of 3 iterations
    final_results = run_full_optimization_loop(max_iterations=3)
    
    # Save results to a file
    with open("optimization_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to optimization_results.json")