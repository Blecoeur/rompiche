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
3. Suggest a config file for the optimization process.

## Tips for the generation of the config file

- Start simple for the prompt and the schema. Most of the time ALL fields are required.
- For the evaluator
    - fields like date, time, ...should be evaluated with exact_match.
    - fields like title, description, ...should be evaluated with string_distance.
- For the max iterations, start with 10.
- For the data file, use the dataset file.