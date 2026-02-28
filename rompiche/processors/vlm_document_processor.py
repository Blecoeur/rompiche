"""
VLM (Vision-Language Model) document/image to JSON processor using Mistral AI.
"""
import json
import os
from typing import Dict, Any, Optional
import mistralai
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()


class BaseDocumentProcessor:
    """Base class for document/image to JSON processors."""
    
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
    
    def _track_tokens(self, response) -> int:
        """Track token usage from API response."""
        tokens_used = 0
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'total_tokens'):
                tokens_used = response.usage.total_tokens
            elif hasattr(response.usage, 'prompt_tokens') and hasattr(response.usage, 'completion_tokens'):
                tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
        
        self.tokens_used += tokens_used
        return tokens_used
    
    def get_token_usage(self) -> int:
        """Get total tokens used by this processor."""
        return self.tokens_used


class VLMOnlyDocumentProcessor(BaseDocumentProcessor):
    """Processor for document/image to JSON using VLM (Vision-Language Model) only."""
    
    def process(self, image_path: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image/document and extract structured information using VLM.
        
        Args:
            image_path: Path to the image/document file
            prompt: Instructions for the extraction
            schema: JSON schema defining the output structure
            
        Returns:
            Dictionary containing the extracted information
        """
        # Open and encode the image
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Convert to base64 for API
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
        except Exception as e:
            print(f"Error reading image file: {e}")
            return {}
        
        function_name = "extract_information"
        function_description = "Extract structured information from document/image. Arguments cannot be null."

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
                        "content": [
                            {
                                "type": "text",
                                "text": "Document to process:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                tools=tools,
                tool_choice="any",
                temperature=self.temperature
            )

            self._track_tokens(response)

            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == function_name:
                    args_dict = json.loads(tool_call.function.arguments)
                    return args_dict

            return {}

        except Exception as e:
            print(f"Error calling Mistral API: {e}")
            return {}


def process_vlm_only(image_path: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy function for VLM-only processing (backward compatibility).
    """
    processor = VLMOnlyDocumentProcessor()
    return processor.process(image_path, prompt, schema)