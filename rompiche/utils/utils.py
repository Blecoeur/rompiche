from typing import Dict, Any, List
import json
import os
import importlib
from typing import Callable

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_file, "r") as f:
        return json.load(f)


def load_processor_module(
    module_path: str,
) -> Callable[[str, str, Dict[str, Any]], Dict[str, Any]]:
    """Load processor from Python module path or file path."""
    try:
        module = None

        # If it's an existing file path, load from file location.
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("custom_processor", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Otherwise treat it as a dotted Python module path.
            module = importlib.import_module(module_path)

        # Get the process function
        if not hasattr(module, "process"):
            raise AttributeError(
                f"Module {module_path} does not have a 'process' function. Expected signature: process(input: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]"
            )

        return module.process

    except Exception as e:
        raise type(e)(
            f"Error loading processor module '{module_path}': {str(e)}"
        ) from e
