import json
import os
from typing import Dict, Any
import mistralai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process(input: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")

    client = mistralai.Mistral(api_key=api_key)

    function_name = "extract_information"
    function_description = "Extract structured information from text. Arguments cannot be null."

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

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
                "parameters": parameters
            }
        }
    ]

    try:
        response = client.chat.complete(
            model="ministral-3b-latest",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Text to process: {input}"
                }
            ],
            tools=tools,
            tool_choice="any",
            temperature=0.1
        )

        # Track token usage from response
        tokens_used = 0
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'total_tokens'):
                tokens_used = response.usage.total_tokens
            elif hasattr(response.usage, 'prompt_tokens') and hasattr(response.usage, 'completion_tokens'):
                tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
        
        # Store tokens in a global variable or return them
        if 'tokens_used' not in process.__dict__:
            process.tokens_used = 0
        process.tokens_used += tokens_used

        if response.choices and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == function_name:
                args_dict = json.loads(tool_call.function.arguments)
                return args_dict

        return {}

    except Exception as e:
        print(f"Error calling Mistral API: {e}")
        return {}

# Example usage
if __name__ == "__main__":
    example_input = "The code review is on 2026-03-11 at 9:52 AM. Coffee provided."
    example_prompt = "Extract the date, time, and event title from the text. All fields are required and cannot be null."
    example_schema = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "The date of the event"},
            "time": {"type": "string", "description": "The time of the event"},
            "title": {"type": "string", "description": "The title of the event"}
        },
        "required": ["date", "time", "title"]
    }

    result = process(example_input, example_prompt, example_schema)
    print(json.dumps(result, indent=2))
