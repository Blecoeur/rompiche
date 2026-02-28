def apply_suggestions(current_prompt, current_schema, suggestions):
    """
    Applies LLM suggestions to prompt/schema.
    Returns: updated_prompt, updated_schema
    """
    updated_prompt = current_prompt
    updated_schema = current_schema

    for suggestion in suggestions:
        if suggestion is None:
            continue

        suggestion_lower = str(suggestion).lower()

        # Handle prompt-related suggestions
        if "prompt" in suggestion_lower:
            # Extract formatting instructions and integrate them into the main prompt
            if any(
                keyword in suggestion_lower
                for keyword in ["format", "example", "iso 8601", "yyyy-mm-dd", "hh:mm"]
            ):
                # Add formatting instructions directly to the prompt
                if "date" in suggestion_lower and "time" in suggestion_lower:
                    updated_prompt = f"{current_prompt} Use ISO 8601 format for dates (YYYY-MM-DD) and 24-hour format for times (HH:MM)."
                elif "date" in suggestion_lower:
                    updated_prompt = (
                        f"{current_prompt} Use ISO 8601 format for dates (YYYY-MM-DD)."
                    )
                elif "time" in suggestion_lower:
                    updated_prompt = (
                        f"{current_prompt} Use 24-hour format for times (HH:MM)."
                    )

            # Add examples directly to the prompt
            if "example" in suggestion_lower:
                updated_prompt = f'{updated_prompt} Example: {{"date": "2023-10-05", "time": "14:30", "title": "Team Meeting"}}'

            # Add edge case handling
            if any(
                keyword in suggestion_lower
                for keyword in ["edge-case", "ambiguous", "no time", "all-day"]
            ):
                updated_prompt = f"{updated_prompt} For events with no time, return null for the time field."

        # Handle schema-related suggestions
        if "schema" in suggestion_lower:
            # Add format constraints to schema
            if "date" in updated_schema.get("properties", {}):
                if (
                    "format" in suggestion_lower
                    or "iso 8601" in suggestion_lower
                    or "yyyy-mm-dd" in suggestion_lower
                ):
                    updated_schema["properties"]["date"]["format"] = "date"
                    updated_schema["properties"]["date"]["pattern"] = (
                        "^\\d{4}-\\d{2}-\\d{2}$"
                    )

            if "time" in updated_schema.get("properties", {}):
                if (
                    "format" in suggestion_lower
                    or "hh:mm" in suggestion_lower
                    or "24-hour" in suggestion_lower
                ):
                    updated_schema["properties"]["time"]["format"] = "time"
                    updated_schema["properties"]["time"]["pattern"] = (
                        "^([01]\\d|2[0-3]):[0-5]\\d$"
                    )

            # Make time optional if suggested
            if any(
                keyword in suggestion_lower
                for keyword in ["optional", "no time", "all-day"]
            ):
                if "time" in updated_schema.get("required", []):
                    updated_schema["required"] = [
                        field for field in updated_schema["required"] if field != "time"
                    ]

    return updated_prompt, updated_schema


# Example usage
if __name__ == "__main__":
    current_prompt = "Extract the title and date from the text."
    current_schema = {"title": "str", "date": "YYYY-MM-DD"}
    suggestions = [
        "Add examples of titles with suffixes (e.g., 'Workshop').",
        "Clarify date format in prompt (e.g., 'YYYY-MM-DD').",
    ]

    new_prompt, new_schema = apply_suggestions(
        current_prompt, current_schema, suggestions
    )
    print("Updated prompt:", new_prompt)
    print("Updated schema:", new_schema)
