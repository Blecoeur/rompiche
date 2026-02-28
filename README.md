# Rompiche

A Python tool for optimizing LLM-based information extraction workflows through iterative prompt and schema refinement.

## Features

- **Iterative Optimization**: Automatically refines prompts and schemas to improve extraction accuracy
- **Custom Evaluation**: Define success criteria and metrics for your specific use case
- **CLI Interface**: Run optimizations via command line with configurable parameters
- **TUI Dashboard**: Optional live dashboard for monitoring optimization progress
- **Mistral AI Integration**: Uses Mistral's API for structured output extraction

## Installation

(Not yet released)
```bash
pip install rompiche 
```

## Usage

### Command Line Interface

#### Development (current setup)

```bash
# Run the CLI optimization tool
uv run python -m rompiche.core.main_optimization_cli \
  --config example_input/example_config.json \
  --processor rompiche.processors.text_to_json_processor \
  --output output/results.json

# With TUI dashboard
uv run python -m rompiche.core.main_optimization_cli \
  --config example_input/example_config.json \
  --processor rompiche.processors.text_to_json_processor \
  --output output/results.json \
  --tui

# Help
uv run python -m rompiche.core.main_optimization_cli --help
```

#### After Installation

```bash
# Run the CLI optimization tool
rompiche --config path/to/config.json --processor path/to/processor.py --output results.json

# Run the TUI dashboard (requires optional TUI dependencies)
rompiche-tui

# Install TUI dependencies if needed
pip install rompiche[tui]
```

### Python API

You can also use rompiche programmatically:

```python
from rompiche.core.main_optimization_cli import main

# Run optimization from Python
main()

# Or use the TUI dashboard
from rompiche.tui.__main__ import main as tui_main
tui_main()
```

## Quick Start

1. **Set up your environment**: Create a `.env` file with your Mistral API key:

```bash
MISTRAL_API_KEY=your_api_key_here
```

2. **Prepare your data**: Create a JSONL file with input text and expected results:

```json
{"input": {"text": "The event is on 2026-03-11 at 9:52 AM"}, "results": {"date": "2026-03-11", "time": "09:52:00", "title": "Event"}}
```

3. **Generate a config file**: Use the `skill.md` file to generate a config file for the optimization process. This will guide you through the setup and suggest a config file based on your dataset.

4. **Create a processor module** (see `example_processor.py` for reference):

```python
def process(input: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    # Your custom processing logic using Mistral API
    pass
```

5. **Run the optimization**:

```bash
python -m rompiche.core.main_optimization_cli \
  --config path/to/config.json \
  --processor path/to/processor.py \
  --output results.json
```

## Configuration

### Config File Structure

The config file is a JSON file that defines your optimization parameters:

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
  "data_file": "example_input/example.jsonl"
}
```

### Available Processors

You can use any of these built-in processors:

- **`rompiche.processors.text_to_json_processor`** - For text processing (recommended for most use cases)
- **`rompiche.processors.vlm_document_processor`** - For image/document processing using VLM only
- **`rompiche.processors.ocr_vlm_document_processor`** - For OCR + VLM processing

### Configuration Options

- **`prompt`**: Initial extraction prompt for the LLM
- **`schema`**: JSON schema defining expected output structure
- **`evaluator`**: Evaluation metrics and success thresholds
- **`max_iterations`**: Maximum number of optimization iterations
- **`data_file`**: Path to JSONL file with training data
- **`max_samples`**: Optional limit on samples processed per iteration

### Evaluator Metrics

- **`exact_match`**: Exact string matching
- **`string_distance`**: Similarity score (0-1)
- **`regex_match`**: Regex pattern matching
- **`type_match`**: Type validation

### Example Data Format

The data file should be in JSONL format (JSON Lines):

```json
{"input": {"text": "The event is on 2026-03-11 at 9:52 AM"}, "results": {"date": "2026-03-11", "time": "09:52:00", "title": "Event"}}
{"input": {"text": "Meeting scheduled for 2026-03-12 at 2:30 PM"}, "results": {"date": "2026-03-12", "time": "14:30:00", "title": "Meeting"}}
```

## Example

### Running the Example

The project includes a complete working example:

```bash
# 1. Set up API key
echo "MISTRAL_API_KEY=your_api_key_here" > .env

# 2. Run the optimization with the example config
uv run python -m rompiche.core.main_optimization_cli \
  --config example_input/example_config.json \
  --processor rompiche.processors.text_to_json_processor \
  --output output/results.json \
  --tui

# 3. View the results
cat output/results.json
```

### Example Files

- **`example_input/example_config.json`**: Sample configuration file
- **`example_input/example.jsonl`**: Sample data file with test cases
- **`output/results.json`**: Output file with optimization results

### Custom Example

To create your own example:

1. **Create a config file** based on `example_input/example_config.json`
2. **Prepare your data** in JSONL format
3. **Run the optimization** with your config and data
4. **Analyze the results** in the output file

### Built-in Processors

All built-in processors support configuration injection:

```python
from rompiche.processors import TextToJsonProcessor

# With custom configuration
config = {
    "model": "mistral-large-latest",
    "temperature": 0.3
}
processor = TextToJsonProcessor(config=config)
```

## Dependencies

- Python 3.11+
- Mistral AI SDK
- DeepDiff for comparison metrics
- Textual for TUI (optional)
- tqdm for progress tracking

## Testing

### Running Tests

The project includes a comprehensive test suite:

```bash
# Run all tests
uv run python -m pytest

# Run specific test files
uv run python -m pytest tests/unit/test_entrypoints.py
uv run python -m pytest tests/integration/test_processors_integration.py

# Run with verbose output
uv run python -m pytest -v

# Run only unit tests
uv run python -m pytest -m unit

# Run only integration tests
uv run python -m pytest -m integration
```

### Test Results

```
23 passed, 8 skipped
```

All core functionality is tested and working correctly. Skipped tests are related to optional dependencies.

### Linting

The project uses `ruff` for linting and formatting:

```bash
# Check for linting issues
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Check formatting
uv run ruff format --check .

# Apply formatting
uv run ruff format .
```

## License

MIT
