# 😴 Rompiche

A Python tool for optimizing LLM-based information extraction workflows through iterative prompt and schema refinement.

**All you need is a benchmark and an API key to get started!**

Built with ❤️ and Mistral Vibe during the Mistral AI Worldwide Hackathon

## Features

- **Iterative Optimization**: Automatically refines prompts and schemas
- **Train/Test Split**: Evaluate on held-out test set while optimizing on training data
- **Custom Evaluation**: Define success criteria and metrics
- **CLI Interface**: Command line tool with configurable parameters
- **TUI Dashboard**: Live dashboard for monitoring progress
- **Mistral AI Integration**: Uses Mistral's API for structured output extraction

## Installation

```bash
pip install rompiche
```

## Quick Start

1. **Set up environment**:
```bash
echo "MISTRAL_API_KEY=your_api_key_here" > .env
```

2. **Run optimization**:
```bash
rompiche --config example_input/example_config.json \
         --processor rompiche.processors.text_to_json_processor \
         --output output/results.json
```

## Usage

### CLI Options
```bash
# Basic usage
rompiche --config path/to/config.json --processor path/to/processor.py --output results.json

# With TUI dashboard
rompiche --config path/to/config.json --processor path/to/processor.py --output results.json --tui

# With train/test split
rompiche --config path/to/config.json --processor path/to/processor.py --output results.json --test-size 0.2
```

### Python API
```python
from rompiche.core.main_optimization_cli import main
main()
```

## Configuration

### Config File Structure
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
  "data_file": "example_input/example.jsonl",
  "test_size": 0.2
}
```

### Available Processors
- `rompiche.processors.text_to_json_processor` - Text processing
- `rompiche.processors.vlm_document_processor` - Image/document processing (VLM only)
- `rompiche.processors.ocr_vlm_document_processor` - OCR + VLM processing

## Example Files

- `example_input/example_config.json` - Sample configuration
- `example_input/example.jsonl` - Sample data
- `output/results.json` - Output results

## Testing

```bash
# Run tests
uv run python -m pytest

# Linting
uv run ruff check .
uv run ruff format .
```

## Contribute

We welcome contributions! Please feel free to:

- **Report issues**: Submit bug reports or feature requests
- **Fix bugs**: Send pull requests for bug fixes
- **Add features**: Propose and implement new functionality
- **Improve documentation**: Help make the docs better

To contribute:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

## License

MIT