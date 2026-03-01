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
import threading

# Ensure the project root is on sys.path so intra-package imports work
# regardless of how the script is invoked.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import after path modification
from rompiche.utils.utils import load_config, load_processor_module
from rompiche.core.loop import run_full_optimization_loop

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
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Proportion of data to use for test set (0.0 to 1.0)",
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
    
    # Get test_size from CLI or config
    test_size = args.test_size if args.test_size is not None else config.get("test_size", 0.0)

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
                    test_size=test_size,
                    data_file=config.get("data_file"),
                    processor_func=processor_func,
                    tracker=tracker,
                    use_tui=True,
                )
                result_queue.put(final_results)
            except Exception as e:
                tracker.freeze_elapsed_time()
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
                test_size=test_size,
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
