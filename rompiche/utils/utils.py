from typing import Dict, Any, List, Tuple
import json
import os
import importlib
from typing import Callable
import random

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def split_data(data: List[Dict[str, Any]], test_size: float = 0.0, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training and test sets.
    
    Args:
        data: List of data samples
        test_size: Proportion of data to use for test set (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_data, test_data)
    """
    if test_size <= 0 or test_size >= 1.0:
        return data, []
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split index
    split_idx = int(len(shuffled_data) * (1 - test_size))
    
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data

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
                f"Module {module_path} does not have a 'process' function. Expected signature: process(input_data: Dict[str, Any], prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]"
            )

        return module.process

    except Exception as e:
        raise type(e)(
            f"Error loading processor module '{module_path}': {str(e)}"
        ) from e
