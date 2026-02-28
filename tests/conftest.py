"""
Pytest configuration file for rompiche tests.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path so we can import rompiche
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def pytest_configure(config):
    """Pytest configuration hook"""
    # Set a dummy API key for tests that need it
    os.environ["MISTRAL_API_KEY"] = "test-api-key-for-tests-only"


def pytest_collection_modifyitems(items):
    """Add markers to tests based on their location"""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker("unit")
        elif "integration" in str(item.fspath):
            item.add_marker("integration")


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary"""
    return {
        "model": "mistral-large-latest",
        "temperature": 0.3,
        "api_key": "test-api-key",
    }


@pytest.fixture
def sample_schema():
    """Return a sample JSON schema"""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["name", "email"],
    }


@pytest.fixture
def sample_prompt():
    """Return a sample prompt"""
    return "Extract the person's name, age, and email from the text."


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary image file for testing"""
    # Create a simple 1x1 PNG image
    image_path = tmp_path / "test_image.png"
    png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    image_path.write_bytes(png_data)
    return str(image_path)
