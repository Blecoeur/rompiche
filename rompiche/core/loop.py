from typing import Dict, Any, Callable
from rompiche.core.metrics import MetricsTracker
from rompiche.core.evaluator import Evaluator
from rompiche.core.brain import get_brain_decision
from rompiche.utils.utils import load_data, split_data
from rompiche.utils.evaluate_utils import evaluate_all_results
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import random
import time


def _normalize_update_changes(changes: Any) -> list[str]:
    """Normalize update changes into a non-empty list of strings."""
    if isinstance(changes, list):
        normalized = [str(item).strip() for item in changes if str(item).strip()]
        if normalized:
            return normalized
    elif isinstance(changes, str) and changes.strip():
        return [changes.strip()]
    return []


def _has_meaningful_prompt_update(updated_prompt: Any, current_prompt: str) -> bool:
    """Return True when the brain provided a non-empty changed prompt."""
    return (
        isinstance(updated_prompt, str)
        and updated_prompt.strip() != ""
        and updated_prompt != current_prompt
    )


def _has_meaningful_schema_update(updated_schema: Any, current_schema: Dict[str, Any]) -> bool:
    """Return True when the brain provided a non-empty changed schema."""
    return (
        isinstance(updated_schema, dict)
        and bool(updated_schema)
        and updated_schema != current_schema
    )


def _initialize_loop(
    initial_prompt: str,
    initial_schema: Dict[str, Any],
    data_file: str,
    max_samples: int | None,
    test_size: float,
    tracker: MetricsTracker,
    evaluator_config: Dict[str, Any],
    use_tui: bool,
) -> tuple[Dict, Dict, Evaluator]:
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
    
    # Split into train/test sets
    train_data, test_data = split_data(data, test_size=test_size)
    
    evaluator = Evaluator(evaluator_config)
    
    if use_tui and tracker:
        tracker.evaluator = evaluator
        tracker.update_status("💤 Starting optimization loop...")
    else:
        print("Starting optimization loop...")
        print(f"Initial prompt: {current_prompt}")
        print(f"Initial schema: {json.dumps(current_schema, indent=2)}")
        print(f"Evaluator config: {json.dumps(evaluator_config, indent=2)}")
        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    return train_data, test_data, evaluator


def _run_processor(
    data: list[Dict],
    processor_func: Callable,
    current_prompt: str,
    current_schema: Dict[str, Any],
    tracker: MetricsTracker,
    use_tui: bool,
    status_message: str | None = None,
    evaluator: Evaluator | None = None,
    early_stop_per_field: int | None = None,
    batch_size: int | None = None,
) -> tuple[list[Dict], int, list[Dict], bool]:
    """Run processor and optionally early-stop when every mismatched field is saturated.

    When *early_stop_per_field* is set, the processor keeps running until
    every field that has at least one mismatch has collected that many
    examples. This guarantees the brain receives balanced coverage across
    all problematic fields.
    
    Args:
        batch_size: Number of items to process in parallel. If None, processes sequentially.
    """
    message = status_message or "💻 Running processor on all files..."
    if use_tui and tracker:
        tracker.update_status(message)
    else:
        print(message)

    processing_results = []
    processor_tokens_used = 0
    total_items = len(data)
    processed_items = 0
    early_stopped_on_mismatches = False
    field_mismatches: dict[str, list[Dict[str, Any]]] = {}

    # Process data in batches if batch_size is specified
    if batch_size and batch_size > 1:
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]
            
            before_batch_tokens = _get_processor_tokens(processor_func)
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_index = {}
                for batch_index, item in enumerate(batch):
                    index = batch_start + batch_index + 1
                    future = executor.submit(
                        processor_func, 
                        item["input"], 
                        current_prompt, 
                        current_schema
                    )
                    future_to_index[future] = index
                
                results_in_batch = {}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        prediction = future.result()
                        results_in_batch[index] = ("success", prediction)
                    except Exception as e:
                        print(f"Error processing item {index}: {e}")
                        results_in_batch[index] = ("error", {})
            
            after_batch_tokens = _get_processor_tokens(processor_func)
            processor_tokens_used += max(0, after_batch_tokens - before_batch_tokens)
            
            for batch_index, item in enumerate(batch):
                index = batch_start + batch_index + 1
                result_type, prediction = results_in_batch.get(index, ("error", {}))
                ground_truth = item["results"]
                processing_results.append({
                    "input": item["input"],
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                })
                processed_items = index
                
                if evaluator and early_stop_per_field and result_type == "success":
                    field_evaluation = evaluator.evaluate(prediction, ground_truth)
                    for field_name in ground_truth.keys():
                        if field_name not in field_mismatches:
                            field_mismatches[field_name] = []

                        field_score = field_evaluation.get(field_name, {})
                        field_thresholds = evaluator.success_thresholds.get(field_name, {})
                        field_failed = any(
                            field_score.get(metric, 1.0) < field_thresholds.get(metric, 1.0)
                            for metric in field_score
                        )

                        if field_failed and len(field_mismatches[field_name]) < early_stop_per_field:
                            field_mismatches[field_name].append(
                                {
                                    "input": item["input"],
                                    "prediction": prediction,
                                    "ground_truth": ground_truth,
                                    "field": field_name,
                                    "field_score": field_score,
                                    "type": "field_mismatch",
                                }
                            )

                    fields_with_mismatches = [
                        f for f, m in field_mismatches.items() if len(m) > 0
                    ]
                    if fields_with_mismatches and all(
                        len(field_mismatches[f]) >= early_stop_per_field
                        for f in fields_with_mismatches
                    ):
                        early_stopped_on_mismatches = True
                        break

            if use_tui and tracker:
                tracker.update_progress(processed_items, total_items)
                
            if early_stopped_on_mismatches:
                break
    else:
        # Process sequentially (original behavior)
        for index, item in enumerate(data, start=1):
            input_data = item["input"]
            before_tokens = _get_processor_tokens(processor_func)
            prediction = processor_func(input_data, current_prompt, current_schema)
            after_tokens = _get_processor_tokens(processor_func)
            processor_tokens_used += max(0, after_tokens - before_tokens)
            ground_truth = item["results"]
            processing_results.append({
                "input": input_data,
                "prediction": prediction,
                "ground_truth": ground_truth,
            })
            processed_items = index

            if evaluator and early_stop_per_field:
                field_evaluation = evaluator.evaluate(prediction, ground_truth)
                for field_name in ground_truth.keys():
                    if field_name not in field_mismatches:
                        field_mismatches[field_name] = []

                    field_score = field_evaluation.get(field_name, {})
                    field_thresholds = evaluator.success_thresholds.get(field_name, {})
                    field_failed = any(
                        field_score.get(metric, 1.0) < field_thresholds.get(metric, 1.0)
                        for metric in field_score
                    )

                    if field_failed and len(field_mismatches[field_name]) < early_stop_per_field:
                        field_mismatches[field_name].append(
                            {
                                "input": input_data,
                                "prediction": prediction,
                                "ground_truth": ground_truth,
                                "field": field_name,
                                "field_score": field_score,
                                "type": "field_mismatch",
                            }
                        )

                fields_with_mismatches = [
                    f for f, m in field_mismatches.items() if len(m) > 0
                ]
                if fields_with_mismatches and all(
                    len(field_mismatches[f]) >= early_stop_per_field
                    for f in fields_with_mismatches
                ):
                    early_stopped_on_mismatches = True
                    break

            if use_tui and tracker:
                tracker.update_progress(index, total_items)

    if use_tui and tracker:
        tracker.update_progress(processed_items, total_items)

    mismatch_examples: list[Dict[str, Any]] = []
    for mismatches in field_mismatches.values():
        mismatch_examples.extend(mismatches)

    if early_stopped_on_mismatches:
        field_counts = ", ".join(
            f"{f}: {len(m)}" for f, m in field_mismatches.items() if len(m) > 0
        )
        early_stop_message = (
            f"🎯 All mismatched fields saturated at {early_stop_per_field}/field "
            f"({field_counts}). "
            f"Stopped training early at {processed_items}/{total_items} samples."
        )
        if use_tui and tracker:
            tracker.update_status(early_stop_message)
        else:
            print(early_stop_message)

    return (
        processing_results,
        processor_tokens_used,
        mismatch_examples,
        early_stopped_on_mismatches,
    )


def _evaluate_results(
    processing_results: list[Dict],
    evaluator: Evaluator,
    tracker: MetricsTracker,
    use_tui: bool,
    dataset_type: str = "train",
    status_message: str | None = None,
) -> Dict:
    """Evaluate processing results and return metrics."""
    status_msg = status_message or f"Evaluating {dataset_type} results..."
    if use_tui and tracker:
        tracker.update_status(status_msg)
    else:
        print(status_msg)
    
    metrics = evaluate_all_results(processing_results, evaluator)
    
    if tracker:
        tracker.add_iteration_metrics(metrics, dataset_type=dataset_type)
    if not use_tui:
        print(f"{dataset_type.upper()} Metrics:")
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
    performance_worsened: bool = False,
    status_message: str | None = None,
) -> Dict | None:
    """Consult the brain for optimization decisions."""
    message = status_message or "🤖 Consulting optimization brain..."
    if use_tui and tracker:
        tracker.update_status(message)
    else:
        print(message)
    
    try:
        hints = tracker.user_hints if tracker else None
        before_tokens = getattr(get_brain_decision, "tokens_used", 0)
        decision = get_brain_decision(
            metrics,
            current_prompt,
            current_schema,
            evaluator_config,
            mismatch_examples,
            hints,
            performance_worsened,
        )
        after_tokens = getattr(get_brain_decision, "tokens_used", 0)
        brain_tokens_used = max(0, after_tokens - before_tokens)
        if tracker and brain_tokens_used:
            tracker.add_tokens(brain_tokens_used)
        
        if use_tui and tracker:
            tracker.update_status("🤖 Brain decision received")
        else:
            print(f"🤖 Brain decision: {json.dumps(decision, indent=2)}")
        
        return decision
    except Exception as e:
        if tracker:
            error_message = f"Brain consultation failed: {e}"
            tracker.add_brain_update({
                "iteration": iteration + 1,
                "decision": "error",
                "summary": error_message,
                "changes": [error_message],
            })
        if use_tui and tracker:
            tracker.update_status(f"Error consulting brain: {e}")
        else:
            print(f"Error consulting brain: {e}")
            print("Continuing with current prompt and schema...")
        
        return None


def _get_processor_tokens(
    processor_func: Callable[[str, str, Dict[str, Any]], Dict[str, Any]]
) -> int:
    """Read processor token usage from function or bound instance."""
    direct_tokens = getattr(processor_func, "tokens_used", None)
    if isinstance(direct_tokens, (int, float)):
        return int(direct_tokens)

    if hasattr(processor_func, "get_token_usage") and callable(
        getattr(processor_func, "get_token_usage")
    ):
        try:
            usage = processor_func.get_token_usage()
            if isinstance(usage, (int, float)):
                return int(usage)
        except Exception:
            pass

    instance = getattr(processor_func, "__self__", None)
    if instance is not None:
        instance_tokens = getattr(instance, "tokens_used", None)
        if isinstance(instance_tokens, (int, float)):
            return int(instance_tokens)
        if hasattr(instance, "get_token_usage") and callable(
            getattr(instance, "get_token_usage")
        ):
            try:
                usage = instance.get_token_usage()
                if isinstance(usage, (int, float)):
                    return int(usage)
            except Exception:
                pass

    return 0


def _apply_brain_decision(
    decision: Dict,
    current_prompt: str,
    current_schema: Dict[str, Any],
    iteration: int,
    tracker: MetricsTracker,
    use_tui: bool,
    previous_metrics: Dict | None = None,
    status_message: str | None = None,
) -> tuple[str, Dict[str, Any], bool]:
    """Apply brain decision and return updated prompt/schema."""
    if decision["decision"] == "stop":
        reason = decision.get("reason", "").strip()
        stop_summary = reason or "Brain decided to stop optimization."
        stop_changes = _normalize_update_changes(decision.get("changes"))
        if not stop_changes:
            stop_changes = [stop_summary]

        if tracker:
            tracker.add_brain_update({
                "iteration": iteration + 1,
                "decision": "stop",
                "summary": stop_summary,
                "changes": stop_changes,
            })
        if use_tui and tracker:
            tracker.update_status("🤖 Brain decided to stop optimization.")
        else:
            print("🤖 Brain decided to stop optimization.")
        
        return current_prompt, current_schema, True
    
    updated_prompt = decision.get("updated_prompt")
    updated_schema = decision.get("updated_schema")

    prompt_was_updated = _has_meaningful_prompt_update(updated_prompt, current_prompt)
    schema_was_updated = _has_meaningful_schema_update(updated_schema, current_schema)

    if prompt_was_updated:
        current_prompt = updated_prompt
    if schema_was_updated:
        current_schema = updated_schema

    if tracker:
        tracker.set_active_configuration(current_prompt, current_schema)
        reason = decision.get("reason", "").strip()
        summary = decision.get("update_summary")
        change_items = _normalize_update_changes(decision.get("changes"))

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
            if reason:
                summary = f"{summary} Reason: {reason}"

        if not change_items:
            if reason:
                change_items.append(reason)
            elif prompt_was_updated or schema_was_updated:
                change_items.append("No detailed changes were returned by the brain.")
        if not change_items:
            change_items = ["No prompt/schema changes in this decision."]

        tracker.add_brain_update({
            "iteration": iteration + 1,
            "decision": decision.get("decision", "continue"),
            "summary": summary,
            "changes": change_items,
        })
    
    update_message = status_message or "🤖 Applying brain updates"
    if use_tui and tracker:
        tracker.update_status(update_message)
    else:
        print(f"Updated prompt: {current_prompt}")
        print(f"Updated schema: {json.dumps(current_schema, indent=2)}")
    
    return current_prompt, current_schema, False


def _compare_metrics(previous: Dict, current: Dict) -> bool:
    """Compare metrics to determine if performance worsened."""
    # Simple comparison: check if any metric value decreased
    for field, field_metrics in current.items():
        if field in previous:
            prev_value = previous[field].get("value", 0)
            curr_value = field_metrics.get("value", 0)
            # For accuracy/precision metrics, lower is worse
            if field in ["accuracy", "precision", "recall", "f1"]:
                if curr_value < prev_value:
                    return True
            # For error/loss metrics, higher is worse
            elif field in ["error_rate", "loss"]:
                if curr_value > prev_value:
                    return True
    return False


def run_full_optimization_loop(
    initial_prompt: str,
    initial_schema: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    data_file: str,
    processor_func: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
    max_iterations: int,
    max_samples: int | None = None,
    test_size: float = 0.0,
    early_stop_mismatches_per_field: int | None = 5,
    batch_size: int | None = None,
    tracker: MetricsTracker = None,
    use_tui: bool = False,
) -> Dict[str, Any]:
    """Main optimization loop with configurable parameters and optional TUI support"""
    current_prompt = initial_prompt
    current_schema = initial_schema
    previous_metrics = None
    total_processor_tokens = 0
    
    # Initialize loop state
    train_data, test_data, evaluator = _initialize_loop(
        initial_prompt,
        initial_schema,
        data_file,
        max_samples,
        test_size,
        tracker,
        evaluator_config,
        use_tui,
    )
    
    for iteration in range(max_iterations):
        has_test_set = bool(test_data)
        total_steps = 7 if has_test_set else 5

        if use_tui and tracker and tracker.stopped:
            break
        
        # Wait if paused
        while use_tui and tracker and tracker.paused and not tracker.stopped:
            time.sleep(0.1)

        if tracker:
            tracker.current_iteration = iteration + 1
        
        if use_tui and tracker:
            tracker.update_status(
                f"🔄 Iteration {iteration + 1}/{max_iterations} - preparing step 1/{total_steps}"
            )
        else:
            print(f"\n{'=' * 50}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'=' * 50}")
        
        # 1. Evaluate on test set first (if available)
        test_metrics = None
        test_processing_results = None
        if test_data:
            (
                test_processing_results,
                test_processor_tokens,
                _,
                _,
            ) = _run_processor(
                test_data,
                processor_func,
                current_prompt,
                current_schema,
                tracker,
                use_tui,
                status_message=f"Step 1/{total_steps}: 💻 Running processor on test set...",
                batch_size=batch_size,
            )
            total_processor_tokens += test_processor_tokens
            if tracker and test_processor_tokens:
                tracker.add_tokens(test_processor_tokens)
            
            test_metrics = _evaluate_results(
                test_processing_results,
                evaluator,
                tracker,
                use_tui,
                dataset_type="test",
                status_message=f"Step 2/{total_steps}: 📊 Evaluating test results...",
            )
            
            # Check success criteria on test set
            if _check_success(test_metrics, evaluator, tracker, use_tui):
                # Store test metrics as final results
                metrics = test_metrics
                break
        
        # 2. Run processor on training set (shuffled to avoid order bias)
        random.shuffle(train_data)
        (
            train_processing_results,
            train_processor_tokens,
            train_mismatch_examples,
            train_early_stopped_on_mismatch_limit,
        ) = _run_processor(
            train_data,
            processor_func,
            current_prompt,
            current_schema,
            tracker,
            use_tui,
            status_message=(
                f"Step 3/{total_steps}: 💻 Running processor on training set..."
                if has_test_set
                else f"Step 1/{total_steps}: 💻 Running processor on training set..."
            ),
            evaluator=evaluator,
            early_stop_per_field=early_stop_mismatches_per_field,
            batch_size=batch_size,
        )
        total_processor_tokens += train_processor_tokens
        if tracker and train_processor_tokens:
            tracker.add_tokens(train_processor_tokens)
        
        # 3. Evaluate training results
        train_metrics = _evaluate_results(
            train_processing_results,
            evaluator,
            tracker,
            use_tui,
            dataset_type="train",
            status_message=(
                f"Step 4/{total_steps}: 📊 Evaluating training results..."
                if has_test_set
                else f"Step 2/{total_steps}: 📊 Evaluating training results..."
            ),
        )
        
        # Use training metrics for optimization decisions
        metrics = train_metrics
        
        # 4. Collect mismatch examples from training set
        mismatch_examples = train_mismatch_examples
        
        if use_tui and tracker:
            # Show all mismatch examples collected for this iteration in the UI.
            tracker.mismatch_examples = list(mismatch_examples)
            tracker.update_status(
                (
                    f"Step 5/{total_steps}: Found {len(mismatch_examples)} mismatch examples from training set"
                    if has_test_set
                    else f"Step 3/{total_steps}: Found {len(mismatch_examples)} mismatch examples"
                )
            )
        else:
            print(f"Found {len(mismatch_examples)} mismatch examples from training set")
        
        if train_early_stopped_on_mismatch_limit:
            if use_tui and tracker:
                tracker.update_status("🎯 Enough mismatch examples collected. Moving to brain consultation...")
            else:
                print("🎯 Enough mismatch examples collected. Moving to brain consultation...")
        
        # 5. Check if performance worsened
        performance_worsened = False
        if previous_metrics and _compare_metrics(previous_metrics, metrics):
            performance_worsened = True
            if use_tui and tracker:
                tracker.update_status("⚠️ Performance worsened! Reverting to previous configuration...")
            else:
                print("⚠️ Performance worsened! Reverting to previous configuration...")
            # Revert to previous prompt/schema
            current_prompt, current_schema = previous_metrics.get("prompt", current_prompt), previous_metrics.get("schema", current_schema)
            if tracker:
                tracker.set_active_configuration(current_prompt, current_schema)
        
        # 6. Consult brain
        decision = _consult_brain(
            metrics,
            current_prompt,
            current_schema,
            evaluator_config,
            mismatch_examples,
            tracker,
            use_tui,
            iteration,
            performance_worsened,
            status_message=(
                f"Step 6/{total_steps}: 🤖 Consulting optimization brain..."
                if has_test_set
                else f"Step 4/{total_steps}: 🤖 Consulting optimization brain..."
            ),
        )
        
        if decision is None:
            # Store current state for next iteration
            previous_metrics = {
                "metrics": metrics,
                "prompt": current_prompt,
                "schema": current_schema,
            }
            continue
        
        # 7. Apply brain decision
        current_prompt, current_schema, should_stop = _apply_brain_decision(
            decision,
            current_prompt,
            current_schema,
            iteration,
            tracker,
            use_tui,
            previous_metrics,
            status_message=(
                f"Step 7/{total_steps}: 🤖 Applying brain updates..."
                if has_test_set
                else f"Step 5/{total_steps}: 🤖 Applying brain updates..."
            ),
        )
        
        if should_stop:
            break
        
        # Store current state for next iteration
        previous_metrics = {
            "metrics": metrics,
            "prompt": current_prompt,
            "schema": current_schema,
        }
    
    # Finalize - evaluate on test set if available
    final_metrics = metrics
    if test_data:
        (
            final_processing_results,
            final_test_processor_tokens,
            _,
            _,
        ) = _run_processor(
            test_data,
            processor_func,
            current_prompt,
            current_schema,
            tracker,
            use_tui,
            status_message="Final test evaluation: 💻 Running processor on test set...",
            batch_size=batch_size,
        )
        total_processor_tokens += final_test_processor_tokens
        if tracker and final_test_processor_tokens:
            tracker.add_tokens(final_test_processor_tokens)
        final_metrics = _evaluate_results(
            final_processing_results,
            evaluator,
            tracker,
            use_tui,
            dataset_type="test",
            status_message="Final test evaluation: 📊 Evaluating test results...",
        )
    
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
        for field, field_metrics in final_metrics.items():
            print(f"  {field}: {field_metrics}")
        print(f"Final prompt: {current_prompt}")
        print(f"Final schema: {json.dumps(current_schema, indent=2)}")
    
    # Return final results
    result = {
        "metrics": final_metrics,
        "prompt": current_prompt,
        "schema": current_schema,
        "iteration": iteration + 1,
    }
    
    # Add token usage information
    processor_tokens = total_processor_tokens
    brain_tokens = getattr(get_brain_decision, "tokens_used", 0)
    total_tokens = processor_tokens + brain_tokens
    
    result["token_usage"] = {
        "processor_tokens": processor_tokens,
        "brain_tokens": brain_tokens,
        "total_tokens": total_tokens,
    }
    
    return result
