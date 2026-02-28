from typing import Dict, Any, List, Tuple


def create_function_calling_tools(
    schema: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create function calling tools for a given function name and schema.
    """
    function_name = "extract_information"
    function_description = (
        "Extract structured information from text. Arguments cannot be null."
    )

    parameters = {"type": "object", "properties": {}, "required": []}

    for field_name, field_config in schema.get("properties", {}).items():
        parameters["properties"][field_name] = field_config
        if field_name in schema.get("required", []):
            parameters["required"].append(field_name)

    tools = [
        {
            "type": "function",
            "function": {
                "name": function_name,
                "description": function_description,
                "parameters": parameters,
            },
        }
    ]
    return function_name, tools
