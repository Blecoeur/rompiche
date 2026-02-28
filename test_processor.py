from typing import Dict, Any
import time

def process(input: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Simple test processor for TUI testing"""
    # Simulate some processing time
    time.sleep(0.01)
    
    # Create mock result based on schema properties
    result = {}
    for field_name, field_config in schema.get("properties", {}).items():
        if field_config.get("type") == "string":
            result[field_name] = f"mock_{field_name}"
        elif field_config.get("type") == "number":
            result[field_name] = 42
        elif field_config.get("type") == "boolean":
            result[field_name] = True
        else:
            result[field_name] = None
    
    return result

if __name__ == "__main__":
    print("Test processor ready!")