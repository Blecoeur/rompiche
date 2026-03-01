---
name: generate-config
description: Generate a config file for the optimization process
license: MIT
compatibility: Python 3.12+
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - ask_user_question
---

# Generate Config Skill

This skill helps generate a config file for the optimization process.

## Pre-requisites

The user needs to provide a dataset file. Ask where it is located.

## Steps

1. Ask the user where the dataset file is located.
2. Read the dataset file, at least the first 10 lines.
3. Analyze the data structure to identify fields and their types.
4. Suggest a config file for the optimization process.

## Tips for the generation of the config file

- Start simple for the prompt and the schema. Most of the time ALL fields are required.
- For the evaluator:
    - Fields like date, time, numbers should be evaluated with `exact_match`.
    - Fields like title, description, names should be evaluated with `string_distance`.
    - You can also use `regex_match` for pattern-based validation.
- For the max iterations, start with 3-5.
- For the data file, use the dataset file path.
- Consider using a test_size (e.g., 0.2 for 20%) to evaluate on a held-out test set.
- Set `early_stop_mismatches_per_field` (default: 5) to stop training early when enough mismatch examples are collected.

## Config File Structure

```json
{
  "prompt": "Extract the date, time, and event title from the text.",
  "schema": {
    "type": "object",
    "properties": {
      "date": {"type": "string", "description": "The date of the event"},
      "time": {"type": "string", "description": "The time of the event"},
      "title": {"type": "string", "description": "The title of the event"}
    },
    "required": ["date", "time", "title"]
  },
  "evaluator": {
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
  },
  "max_iterations": 3,
  "data_file": "path/to/your/data.jsonl",
  "test_size": 0.2,
  "early_stop_mismatches_per_field": 5
}
```

## Data File Format

The data file should be in JSONL format (JSON Lines):

```json
{"input": {"text": "The event is on 2026-03-11 at 9:52 AM"}, "results": {"date": "2026-03-11", "time": "09:52:00", "title": "Event"}}
{"input": {"text": "Meeting scheduled for 2026-03-12 at 2:30 PM"}, "results": {"date": "2026-03-12", "time": "14:30:00", "title": "Meeting"}}
```