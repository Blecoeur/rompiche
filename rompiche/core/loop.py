from brain import get_brain_decision
from applier import apply_suggestions
import json


def run_optimization_loop(max_iterations=5):
    # Initialize
    current_prompt = "Extract the title and date from the text."
    current_schema = {"title": "str", "date": "YYYY-MM-DD"}
    success_criteria = {"date_exact_match": 1.0, "title_similarity": 0.8}

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # 1. Run your script and evaluator (replace with your functions)
        metrics = run_script_and_evaluate(current_prompt, current_schema)
        print("Metrics:", metrics)

        # 2. Ask the brain
        decision = get_brain_decision(
            metrics, current_prompt, current_schema, success_criteria
        )
        print("Brain decision:", json.dumps(decision, indent=2))

        # 3. Stop or improve
        if decision["decision"] == "stop":
            print("🎉 Target met!")
            break
        else:
            current_prompt, current_schema = apply_suggestions(
                current_prompt, current_schema, decision["suggestions"]
            )
            print("Updated prompt:", current_prompt)
            print("Updated schema:", current_schema)


def run_script_and_evaluate(prompt, schema):
    # TODO: Replace with your actual script/evaluator
    # This is a mock that improves with each iteration
    iteration = (
        run_script_and_evaluate.iteration
        if hasattr(run_script_and_evaluate, "iteration")
        else 0
    )
    run_script_and_evaluate.iteration = iteration + 1
    return {
        "date_exact_match": 0.85 + iteration * 0.05,
        "title_similarity": 0.60 + iteration * 0.05,
    }


if __name__ == "__main__":
    run_optimization_loop()
