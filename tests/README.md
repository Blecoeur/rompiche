# Tests for Rompiche

This directory contains the test suite for the Rompiche project, organized into unit and integration tests.

## Test Structure

```
tests/
├── conftest.py          # Pytest configuration and common fixtures
├── __init__.py          # Package initialization
├── unit/                # Unit tests (fast, isolated)
│   └── test_entrypoints.py  # Tests for entry points and configuration
└── integration/         # Integration tests (broader scope)
    └── test_processors_integration.py  # Tests for processor functionality
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test files
```bash
pytest tests/unit/test_entrypoints.py
pytest tests/integration/test_processors_integration.py
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests with coverage
```bash
pytest --cov=rompiche --cov-report=term-missing
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Coverage

### Unit Tests (`tests/unit/test_entrypoints.py`)

- **CLI Entry Point**: Tests for the main CLI entry point
- **TUI Entry Point**: Tests for the TUI dashboard entry point
- **Main.py**: Tests for the main entry point script
- **PyProject.toml**: Tests for entry point configuration
- **Configuration Injection**: Tests for configuration injection in processors

### Integration Tests (`tests/integration/test_processors_integration.py`)

- **Processor Integration**: Tests for processor initialization and configuration
- **Token Tracking**: Tests for token usage tracking
- **Processor Exports**: Tests for proper module exports
- **Configuration Inheritance**: Tests for configuration inheritance in processor hierarchy
- **Error Handling**: Tests for error handling and edge cases

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_config`: Sample configuration dictionary
- `sample_schema`: Sample JSON schema
- `sample_prompt`: Sample prompt text
- `temp_image_file`: Temporary image file for testing

## Skipped Tests

Some tests are marked as skipped:

- **OCR-related tests**: Skipped when OCR dependencies are not available
- **Example processor tests**: Skipped when example processor is not in package

These tests can be enabled by installing the required dependencies:

```bash
# Install OCR dependencies
pip install rompiche[ocr]

# Install TUI dependencies  
pip install rompiche[tui]
```

## Test Results Summary

```
23 passed, 8 skipped
```

All core functionality is tested and working correctly. Skipped tests are related to optional dependencies that are not installed in the test environment.