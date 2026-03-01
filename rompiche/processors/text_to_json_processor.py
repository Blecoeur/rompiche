"""
Text to JSON processor using Mistral AI.
Extracts structured information from text using function calling.
"""

import json
import os
from typing import Dict, Any, Optional
import mistralai
from dotenv import load_dotenv
from rompiche.utils.processor_utils import create_function_calling_tools

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
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables or config"
            )

        self.client = mistralai.Mistral(api_key=api_key)
        self.tokens_used = 0

    def process(
        self, input_data: Dict[str, Any], prompt: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process text and extract structured information based on the provided schema.

        Args:
            input_data: Dict containing at least a "text" key with the text to process
            prompt: Instructions for the extraction
            schema: JSON schema defining the output structure

        Returns:
            Dictionary containing the extracted information
        """
        input_text = input_data["text"]
        function_name, tools = create_function_calling_tools(schema)

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Text to process: {input_text}"},
                ],
                tools=tools,
                tool_choice="any",
                temperature=self.temperature,
            )

            # Track token usage
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                if hasattr(response.usage, "total_tokens"):
                    tokens_used = response.usage.total_tokens
                elif hasattr(response.usage, "prompt_tokens") and hasattr(
                    response.usage, "completion_tokens"
                ):
                    tokens_used = (
                        response.usage.prompt_tokens + response.usage.completion_tokens
                    )

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


def process(input_data: Dict[str, Any], prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level process function expected by the processor loader.
    """
    processor = TextToJsonProcessor()
    result = processor.process(input_data, prompt, schema)
    if not hasattr(process, "tokens_used"):
        process.tokens_used = 0
    process.tokens_used += processor.tokens_used
    return result


def build_mismatch_explanation_messages(
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    mismatch: Dict[str, Any],
) -> list:
    """
    Build explanation messages for a mismatch from a text extraction task.
    Returns a single user message with the source text and field delta as plain text.
    """
    field = mismatch.get("field", "unknown")
    field_score = mismatch.get("field_score", {})
    input_text = input_data.get("text", "") if isinstance(input_data, dict) else str(input_data)

    pred_value = prediction.get(field) if isinstance(prediction, dict) else prediction
    gt_value = ground_truth.get(field) if isinstance(ground_truth, dict) else ground_truth

    content = (
        f"Field with mismatch: {field}\n"
        f"Score: {json.dumps(field_score)}\n\n"
        f"Source text:\n{input_text}\n\n"
        f"Predicted value: {json.dumps(pred_value)}\n"
        f"Expected value:  {json.dumps(gt_value)}"
    )
    return [{"role": "user", "content": content}]
