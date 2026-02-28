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

After installing rompiche, you can use the following commands:

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

- `prompt`: Initial extraction prompt
- `schema`: JSON schema defining expected output structure
- `evaluator`: Evaluation metrics and success thresholds
- `max_iterations`: Maximum number of optimization iterations
- `data_file`: Path to JSONL file with training data
- `max_samples`: Optional limit on samples processed per iteration

### Evaluator Configuration

```json
{
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
}
```

## Example

See `example_processor.py` and `example_input/` for a complete working example.

## Dependencies

- Python 3.11+
- Mistral AI SDK
- DeepDiff for comparison metrics
- Textual for TUI (optional)
- tqdm for progress tracking

## License

MIT
