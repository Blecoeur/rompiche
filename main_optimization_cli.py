#!/usr/bin/env python3
"""
CLI version of the main optimization workflow.
Accepts a config file to initialize prompt, schema, success criteria, evaluator, and number of loops.
"""

import json
import os
import argparse
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
#        if index > 10:
#            break
        input_text = item["input"]["text"]
        prediction = process(input_text, prompt, schema)
        ground_truth = item["results"]
        results.append({
            "input": input_text,
            "prediction": prediction,
            "ground_truth": ground_truth
        })
    return results

def collect_mismatch_examples(results: List[Dict[str, Any]], evaluator: Evaluator, max_examples: int = 10) -> List[Dict[str, Any]]:
    """Collect examples where prediction differs from ground truth or has low scores."""
    # First, collect exact mismatches (up to max_examples)
    exact_mismatches = []
    scored_results = []
    
    for result in results:
        if result["prediction"] != result["ground_truth"]:
            exact_mismatches.append({
                "input": result["input"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"],
                "type": "exact_mismatch"
            })
            if len(exact_mismatches) >= max_examples:
                return exact_mismatches
        
        # Also collect all results with their scores for potential low-scoring examples
        evaluation = evaluator.evaluate(result["prediction"], result["ground_truth"])
        scored_results.append({
            "input": result["input"],
            "prediction": result["prediction"],
            "ground_truth": result["ground_truth"],
            "scores": evaluation,
            "type": "low_score"
        })
    
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

def run_full_optimization_loop(
    initial_prompt: str,
    initial_schema: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    max_iterations: int = 5,
    data_file: str = "example_data/example.jsonl"
) -> Dict[str, Any]:
    """Main optimization loop with configurable parameters"""
    
    current_prompt = initial_prompt
    current_schema = initial_schema
    
    # Load evaluation data
    data = load_data(data_file)
    
    evaluator = Evaluator(evaluator_config)
    
    print("Starting optimization loop...")
    print(f"Initial prompt: {current_prompt}")
    print(f"Initial schema: {json.dumps(current_schema, indent=2)}")

    print(f"Evaluator config: {json.dumps(evaluator_config, indent=2)}")
    
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
        
        # 3. Check if we meet success thresholds using evaluator
        criteria_met = evaluator.is_success(metrics)
        
        if criteria_met:
            print("🎉 Success thresholds met! Stopping optimization.")
            break
        
        # 4. Collect mismatch examples for the brain
        mismatch_examples = collect_mismatch_examples(processing_results, evaluator)
        print(f"Found {len(mismatch_examples)} mismatch examples")
        
        # 5. Ask the brain for decision with mismatch context
        print("Consulting optimization brain...")
        try:
            decision = get_brain_decision(
                metrics, current_prompt, current_schema,
                evaluator_config, mismatch_examples
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

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run optimization workflow with configurable parameters')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--output', type=str, default='optimization_results.json', help='Output file path')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    for key in ['prompt', 'schema', 'evaluator', 'max_iterations', 'data_file']:
        if key not in config:
            raise ValueError(f"Error: Missing required configuration parameter: {key}")
    
    # Run optimization with config parameters
    final_results = run_full_optimization_loop(
        initial_prompt=config.get('prompt'),
        initial_schema=config.get('schema'),
        evaluator_config=config.get('evaluator'),
        max_iterations=config.get('max_iterations'),
        data_file=config.get('data_file')
    )
    
    # Save results to a file
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()