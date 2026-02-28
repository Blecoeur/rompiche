"""
Text to JSON processor using Mistral AI.
Extracts structured information from text using function calling.
"""
import json
import os
from typing import Dict, Any, Optional
import mistralai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextToJsonProcessor:
    """Processor for extracting structured data from text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with optional configuration.
        
        Args:
            config: Dictionary containing configuration parameters:
                - model: Model name to use (default: "ministral-3b-latest")
                - temperature: Temperature for generation (default: 0.1)
                - api_key: Optional API key override
        """
        self.config = config or {}
        
        # Set defaults
        self.model = self.config.get("model", "ministral-3b-latest")
        self.temperature = self.config.get("temperature", 0.1)
        
        # Use provided API key or fall back to environment
        api_key = self.config.get("api_key") or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables or config")
        
        self.client = mistralai.Mistral(api_key=api_key)
        self.tokens_used = 0
    
    def process(self, input_text: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text and extract structured information based on the provided schema.
        
        Args:
            input_text: The text to process
            prompt: Instructions for the extraction
            schema: JSON schema defining the output structure
            
        Returns:
            Dictionary containing the extracted information
        """
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
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"Text to process: {input_text}"
                    }
                ],
                tools=tools,
                tool_choice="any",
                temperature=self.temperature
            )

            # Track token usage
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'total_tokens'):
                    tokens_used = response.usage.total_tokens
                elif hasattr(response.usage, 'prompt_tokens') and hasattr(response.usage, 'completion_tokens'):
                    tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
            
            self.tokens_used += tokens_used

            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == function_name:
                    args_dict = json.loads(tool_call.function.arguments)
                    return args_dict

            return {}

        except Exception as e:
            print(f"Error calling Mistral API: {e}")
            return {}
    
    def get_token_usage(self) -> int:
        """Get total tokens used by this processor."""
        return self.tokens_used


def process(input_text: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Creates a processor instance and calls process.
    """
    processor = TextToJsonProcessor()
    return processor.process(input_text, prompt, schema)