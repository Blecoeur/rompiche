#!/usr/bin/env python3
"""
CLI version of the main optimization workflow.
Accepts a config file to initialize prompt, schema, success criteria, evaluator, and number of loops.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import importlib.util
import threading
import time
from typing import Dict, Any, List, Callable

# Ensure the project root is on sys.path so intra-package imports work
# regardless of how the script is invoked.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import after path modification
from rompiche.core.evaluator import Evaluator  # noqa: E402
from rompiche.core.brain import get_brain_decision  # noqa: E402

# Try to import TUI components
try:
    from rompiche.tui.dashboard import (
        MetricsTracker,
        LiveDashboard,
        check_tui_available,
    )

    TUI_AVAILABLE = check_tui_available()
except ImportError:
    TUI_AVAILABLE = False


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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


def run_full_optimization_loop(
    initial_prompt: str,
    initial_schema: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    data_file: str,
    processor_func: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
    max_iterations: int,
    max_samples: int | None = None,
    tracker: MetricsTracker = None,
    use_tui: bool = False,
) -> Dict[str, Any]:
    """Main optimization loop with configurable parameters and optional TUI support"""

    current_prompt = initial_prompt
    current_schema = initial_schema
    if tracker:
        tracker.set_active_configuration(current_prompt, current_schema)

    # Reset token counters for processor and brain
    if hasattr(processor_func, "tokens_used"):
        processor_func.tokens_used = 0
    if hasattr(get_brain_decision, "tokens_used"):
        get_brain_decision.tokens_used = 0

    # Load evaluation data
    data = load_data(data_file)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer when provided.")
        data = data[:max_samples]

    evaluator = Evaluator(evaluator_config)

    if use_tui and tracker:
        tracker.evaluator = evaluator  # Store evaluator for mismatch analysis
        tracker.update_status("💤 Starting optimization loop...")
    else:
        print("Starting optimization loop...")
        print(f"Initial prompt: {current_prompt}")
        print(f"Initial schema: {json.dumps(current_schema, indent=2)}")
        print(f"Evaluator config: {json.dumps(evaluator_config, indent=2)}")

    for iteration in range(max_iterations):
        if use_tui and tracker and tracker.stopped:
            break

        # Wait if paused
        while use_tui and tracker and tracker.paused and not tracker.stopped:
            time.sleep(0.1)

        if use_tui and tracker:
            tracker.update_status(f"🔄 Iteration {iteration + 1}/{max_iterations}")
        else:
            print(f"\n{'=' * 50}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'=' * 50}")

        # 1. Run processor on all data
        if use_tui and tracker:
            tracker.update_status("💻 Running processor on all files...")
        else:
            print("💻 Running processor on all files...")

        processing_results = []
        total_items = len(data)
        for index, item in enumerate(data, start=1):
            input_text = item["input"]["text"]
            prediction = processor_func(input_text, current_prompt, current_schema)
            ground_truth = item["results"]
            processing_results.append(
                {
                    "input": input_text,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                }
            )
            if use_tui and tracker:
                tracker.update_progress(index, total_items)

        if use_tui and tracker:
            tracker.update_progress(total_items, total_items)

        # Track processor tokens
        if hasattr(processor_func, "tokens_used") and tracker:
            tracker.add_tokens(processor_func.tokens_used)

        # 2. Evaluate results
        if use_tui and tracker:
            tracker.update_status("Evaluating results...")
        else:
            print("Evaluating results...")

        metrics = evaluate_all_results(processing_results, evaluator)

        if use_tui and tracker:
            tracker.add_iteration_metrics(metrics)
        else:
            # Even when not using TUI, track iteration progress
            if tracker:
                tracker.add_iteration_metrics(metrics)
            print("Metrics:")
            for field, field_metrics in metrics.items():
                print(f"  {field}: {field_metrics}")

        # 3. Check if we meet success thresholds using evaluator
        criteria_met = evaluator.is_success(metrics)

        if criteria_met:
            if use_tui and tracker:
                tracker.update_status(
                    "🎉 Success thresholds met! Stopping optimization."
                )
            else:
                print("🎉 Success thresholds met! Stopping optimization.")
            break

        # 4. Collect mismatch examples for the brain
        mismatch_examples = collect_mismatch_examples(processing_results, evaluator)

        if use_tui and tracker:
            # Add top 3 mismatches to tracker
            for example in mismatch_examples[:10]:
                tracker.add_mismatch(example)
            tracker.update_status(f"Found {len(mismatch_examples)} mismatch examples")
        else:
            print(f"Found {len(mismatch_examples)} mismatch examples")

        # 5. Ask the brain for decision with mismatch context
        if use_tui and tracker:
            tracker.update_status("🤖 Consulting optimization brain...")
        else:
            print("🤖 Consulting optimization brain...")

        try:
            hints = tracker.user_hints if tracker else None
            decision = get_brain_decision(
                metrics,
                current_prompt,
                current_schema,
                evaluator_config,
                mismatch_examples,
                hints,
            )

            # Track brain tokens
            if hasattr(get_brain_decision, "tokens_used") and tracker:
                tracker.add_tokens(get_brain_decision.tokens_used)

            if use_tui and tracker:
                tracker.update_status("🤖 Brain decision received")
            else:
                print(f"🤖 Brain decision: {json.dumps(decision, indent=2)}")

            if decision["decision"] == "stop":
                if tracker:
                    tracker.add_brain_update(
                        {
                            "iteration": iteration + 1,
                            "decision": "stop",
                            "summary": decision.get(
                                "reason", "Brain decided to stop optimization."
                            ),
                        }
                    )
                if use_tui and tracker:
                    tracker.update_status("🤖 Brain decided to stop optimization.")
                else:
                    print("🤖 Brain decided to stop optimization.")
                break

            # 6. Use the brain's updated prompt and schema directly
            prompt_was_updated = (
                "updated_prompt" in decision
                and decision["updated_prompt"] != current_prompt
            )
            schema_was_updated = (
                "updated_schema" in decision
                and decision["updated_schema"] != current_schema
            )
            if "updated_prompt" in decision:
                current_prompt = decision["updated_prompt"]
            if "updated_schema" in decision:
                current_schema = decision["updated_schema"]

            if tracker:
                tracker.set_active_configuration(current_prompt, current_schema)
                summary = decision.get("update_summary")
                if not summary:
                    changed_parts = []
                    if prompt_was_updated:
                        changed_parts.append("prompt")
                    if schema_was_updated:
                        changed_parts.append("schema")
                    if changed_parts:
                        summary = f"Updated {', '.join(changed_parts)}."
                    else:
                        summary = "No prompt/schema changes in this decision."
                    reason = decision.get("reason")
                    if reason:
                        summary = f"{summary} Reason: {reason}"
                tracker.add_brain_update(
                    {
                        "iteration": iteration + 1,
                        "decision": decision.get("decision", "continue"),
                        "summary": summary,
                    }
                )

            if use_tui and tracker:
                tracker.update_status("🤖 Applying brain updates")
            else:
                print(f"Updated prompt: {current_prompt}")
                print(f"Updated schema: {json.dumps(current_schema, indent=2)}")

        except Exception as e:
            if tracker:
                tracker.add_brain_update(
                    {
                        "iteration": iteration + 1,
                        "decision": "error",
                        "summary": f"Brain consultation failed: {e}",
                    }
                )
            if use_tui and tracker:
                tracker.update_status(f"Error consulting brain: {e}")
            else:
                print(f"Error consulting brain: {e}")
                print("Continuing with current prompt and schema...")

    if use_tui and tracker:
        tracker.set_active_configuration(current_prompt, current_schema)
        tracker.update_status("🎉 Optimization complete!")
    else:
        # Ensure tracker is up to date even when not using TUI
        if tracker:
            tracker.current_iteration = iteration + 1
        print(f"\n{'=' * 50}")
        print("Optimization complete!")
        print("Final metrics:")
        for field, field_metrics in metrics.items():
            print(f"  {field}: {field_metrics}")
        print(f"Final prompt: {current_prompt}")
        print(f"Final schema: {json.dumps(current_schema, indent=2)}")

    # Return final results
    result = {
        "metrics": metrics,
        "prompt": current_prompt,
        "schema": current_schema,
        "iteration": iteration + 1,
    }

    # Add token usage information
    processor_tokens = getattr(processor_func, "tokens_used", 0)
    brain_tokens = getattr(get_brain_decision, "tokens_used", 0)
    total_tokens = processor_tokens + brain_tokens

    result["token_usage"] = {
        "processor_tokens": processor_tokens,
        "brain_tokens": brain_tokens,
        "total_tokens": total_tokens,
    }

    return result


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_file, "r") as f:
        return json.load(f)


def load_processor_module(
    module_path: str,
) -> Callable[[str, str, Dict[str, Any]], Dict[str, Any]]:
    """Dynamically load a processor module and return its process function"""
    try:
        # Check if file exists
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Processor module not found: {module_path}")

        # Create module spec from file location
        spec = importlib.util.spec_from_file_location("custom_processor", module_path)
        if spec is None:
            raise ImportError(f"Could not load module from {module_path}")

        # Create module from spec
        module = importlib.util.module_from_spec(spec)

        # Execute the module to load its contents
        spec.loader.exec_module(module)

        # Get the process function
        if not hasattr(module, "process"):
            raise AttributeError(
                f"Module {module_path} does not have a 'process' function. Expected signature: process(input: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]"
            )

        return module.process

    except Exception as e:
        raise type(e)(
            f"Error loading processor module '{module_path}': {str(e)}"
        ) from e


def main():
    parser = argparse.ArgumentParser(
        description="Run optimization workflow with configurable parameters"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--processor", type=str, required=True, help="Path to custom processor module"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/optimization_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of dataset samples processed per iteration",
    )
    parser.add_argument("--tui", action="store_true", help="Enable live TUI dashboard")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    for key in ["prompt", "schema", "evaluator", "max_iterations", "data_file"]:
        if key not in config:
            raise ValueError(f"Error: Missing required configuration parameter: {key}")

    max_samples = (
        args.max_samples if args.max_samples is not None else config.get("max_samples")
    )

    # Load processor module
    processor_func = load_processor_module(args.processor)
    print(f"Using custom processor from: {args.processor}")

    # Check if TUI is available and requested
    use_tui = args.tui and TUI_AVAILABLE

    if use_tui:
        print("Starting TUI dashboard...")
        tracker = MetricsTracker()
        tracker.total_iterations = config.get("max_iterations", 5)

        # Run optimization in a background thread since TUI needs main thread
        import queue

        result_queue = queue.Queue()

        def run_optimization():
            try:
                final_results = run_full_optimization_loop(
                    initial_prompt=config.get("prompt"),
                    initial_schema=config.get("schema"),
                    evaluator_config=config.get("evaluator"),
                    max_iterations=config.get("max_iterations"),
                    max_samples=max_samples,
                    data_file=config.get("data_file"),
                    processor_func=processor_func,
                    tracker=tracker,
                    use_tui=True,
                )
                result_queue.put(final_results)
            except Exception as e:
                result_queue.put({"status": "error", "error": str(e)})

        # Start optimization thread
        optimization_thread = threading.Thread(target=run_optimization, daemon=True)
        optimization_thread.start()

        try:
            # Run TUI in main thread
            tui_app = LiveDashboard(tracker)
            tui_app.run()

            # Get results from queue
            final_results = result_queue.get(timeout=5)
        except Exception as e:
            print(f"TUI error: {e}")
            final_results = {"status": "error", "error": str(e)}
    else:
        # Create tracker even when not using TUI to track progress
        tracker = MetricsTracker()
        tracker.total_iterations = config.get("max_iterations", 5)

        # Run without TUI (original behavior)
        try:
            final_results = run_full_optimization_loop(
                initial_prompt=config.get("prompt"),
                initial_schema=config.get("schema"),
                evaluator_config=config.get("evaluator"),
                max_iterations=config.get("max_iterations"),
                max_samples=max_samples,
                data_file=config.get("data_file"),
                processor_func=processor_func,
                tracker=tracker,
                use_tui=False,
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            final_results = {"status": "interrupted"}
        except Exception as e:
            print(f"\nError during optimization: {e}")
            final_results = {"status": "error", "error": str(e)}

    # Save results to a file
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
