from typing import Dict, Any, Callable
from rompiche.core.metrics import MetricsTracker
from rompiche.core.evaluator import Evaluator
from rompiche.core.brain import get_brain_decision
from rompiche.utils.utils import load_data
from rompiche.utils.evaluate_utils import evaluate_all_results
from rompiche.utils.evaluate_utils import collect_mismatch_examples

import json
import time


def _initialize_loop(
    initial_prompt: str,
    initial_schema: Dict[str, Any],
    data_file: str,
    max_samples: int | None,
    tracker: MetricsTracker,
    evaluator_config: Dict[str, Any],
    use_tui: bool,
) -> tuple[Dict, Evaluator]:
    """Initialize loop state and load data."""
    current_prompt = initial_prompt
    current_schema = initial_schema
    
    if tracker:
        tracker.set_active_configuration(current_prompt, current_schema)
    
    # Reset token counters
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
        tracker.evaluator = evaluator
        tracker.update_status("💤 Starting optimization loop...")
    else:
        print("Starting optimization loop...")
        print(f"Initial prompt: {current_prompt}")
        print(f"Initial schema: {json.dumps(current_schema, indent=2)}")
        print(f"Evaluator config: {json.dumps(evaluator_config, indent=2)}")
    
    return data, evaluator


def _run_processor(
    data: Dict,
    processor_func: Callable,
    current_prompt: str,
    current_schema: Dict[str, Any],
    tracker: MetricsTracker,
    use_tui: bool,
) -> list[Dict]:
    """Run processor on all data and return results."""
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
        processing_results.append({
            "input": input_text,
            "prediction": prediction,
            "ground_truth": ground_truth,
        })
        if use_tui and tracker:
            tracker.update_progress(index, total_items)
    
    if use_tui and tracker:
        tracker.update_progress(total_items, total_items)
    
    return processing_results


def _evaluate_results(
    processing_results: list[Dict],
    evaluator: Evaluator,
    tracker: MetricsTracker,
    use_tui: bool,
) -> Dict:
    """Evaluate processing results and return metrics."""
    if use_tui and tracker:
        tracker.update_status("Evaluating results...")
    else:
        print("Evaluating results...")
    
    metrics = evaluate_all_results(processing_results, evaluator)
    
    if use_tui and tracker:
        tracker.add_iteration_metrics(metrics)
    else:
        if tracker:
            tracker.add_iteration_metrics(metrics)
        print("Metrics:")
        for field, field_metrics in metrics.items():
            print(f"  {field}: {field_metrics}")
    
    return metrics


def _check_success(
    metrics: Dict,
    evaluator: Evaluator,
    tracker: MetricsTracker,
    use_tui: bool,
) -> bool:
    """Check if success criteria are met."""
    criteria_met = evaluator.is_success(metrics)
    
    if criteria_met:
        if use_tui and tracker:
            tracker.update_status("🎉 Success thresholds met! Stopping optimization.")
        else:
            print("🎉 Success thresholds met! Stopping optimization.")
    
    return criteria_met


def _consult_brain(
    metrics: Dict,
    current_prompt: str,
    current_schema: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    mismatch_examples: list[Dict],
    tracker: MetricsTracker,
    use_tui: bool,
    iteration: int,
) -> Dict | None:
    """Consult the brain for optimization decisions."""
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
        
        if hasattr(get_brain_decision, "tokens_used") and tracker:
            tracker.add_tokens(get_brain_decision.tokens_used)
        
        if use_tui and tracker:
            tracker.update_status("🤖 Brain decision received")
        else:
            print(f"🤖 Brain decision: {json.dumps(decision, indent=2)}")
        
        return decision
    except Exception as e:
        if tracker:
            tracker.add_brain_update({
                "iteration": iteration + 1,
                "decision": "error",
                "summary": f"Brain consultation failed: {e}",
            })
        if use_tui and tracker:
            tracker.update_status(f"Error consulting brain: {e}")
        else:
            print(f"Error consulting brain: {e}")
            print("Continuing with current prompt and schema...")
        
        return None


def _apply_brain_decision(
    decision: Dict,
    current_prompt: str,
    current_schema: Dict[str, Any],
    iteration: int,
    tracker: MetricsTracker,
    use_tui: bool,
) -> tuple[str, Dict[str, Any], bool]:
    """Apply brain decision and return updated prompt/schema."""
    if decision["decision"] == "stop":
        if tracker:
            tracker.add_brain_update({
                "iteration": iteration + 1,
                "decision": "stop",
                "summary": decision.get("reason", "Brain decided to stop optimization."),
            })
        if use_tui and tracker:
            tracker.update_status("🤖 Brain decided to stop optimization.")
        else:
            print("🤖 Brain decided to stop optimization.")
        
        return current_prompt, current_schema, True
    
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
        tracker.add_brain_update({
            "iteration": iteration + 1,
            "decision": decision.get("decision", "continue"),
            "summary": summary,
        })
    
    if use_tui and tracker:
        tracker.update_status("🤖 Applying brain updates")
    else:
        print(f"Updated prompt: {current_prompt}")
        print(f"Updated schema: {json.dumps(current_schema, indent=2)}")
    
    return current_prompt, current_schema, False


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
    
    # Initialize loop state
    data, evaluator = _initialize_loop(
        initial_prompt,
        initial_schema,
        data_file,
        max_samples,
        tracker,
        evaluator_config,
        use_tui,
    )
    
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
        
        # 1. Run processor
        processing_results = _run_processor(
            data,
            processor_func,
            current_prompt,
            current_schema,
            tracker,
            use_tui,
        )
        
        # 2. Evaluate results
        metrics = _evaluate_results(
            processing_results,
            evaluator,
            tracker,
            use_tui,
        )
        
        # 3. Check success criteria
        if _check_success(metrics, evaluator, tracker, use_tui):
            break
        
        # 4. Collect mismatch examples
        mismatch_examples = collect_mismatch_examples(processing_results, evaluator)
        
        if use_tui and tracker:
            for example in mismatch_examples[:10]:
                tracker.add_mismatch(example)
            tracker.update_status(f"Found {len(mismatch_examples)} mismatch examples")
        else:
            print(f"Found {len(mismatch_examples)} mismatch examples")
        
        # 5. Consult brain
        decision = _consult_brain(
            metrics,
            current_prompt,
            current_schema,
            evaluator_config,
            mismatch_examples,
            tracker,
            use_tui,
            iteration,
        )
        
        if decision is None:
            continue
        
        # 6. Apply brain decision
        current_prompt, current_schema, should_stop = _apply_brain_decision(
            decision,
            current_prompt,
            current_schema,
            iteration,
            tracker,
            use_tui,
        )
        
        if should_stop:
            break
    
    # Finalize
    if use_tui and tracker:
        tracker.set_active_configuration(current_prompt, current_schema)
        tracker.update_status("🎉 Optimization complete!")
        tracker.freeze_elapsed_time()
    else:
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