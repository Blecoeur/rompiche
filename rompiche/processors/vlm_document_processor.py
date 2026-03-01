"""
VLM (Vision-Language Model) document/image to JSON processor using Mistral AI.
"""

import json
import os
from typing import Dict, Any, Optional
import mistralai
from dotenv import load_dotenv
import base64
from rompiche.utils.processor_utils import create_function_calling_tools

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
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables or config"
            )

        self.client = mistralai.Mistral(api_key=api_key)
        self.tokens_used = 0

    def _track_tokens(self, response) -> int:
        """Track token usage from API response."""
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
        return tokens_used

    def get_token_usage(self) -> int:
        """Get total tokens used by this processor."""
        return self.tokens_used


class VLMOnlyDocumentProcessor(BaseDocumentProcessor):
    """Processor for document/image to JSON using VLM (Vision-Language Model) only."""

    def process(
        self, input_data: Dict[str, Any], prompt: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an image/document and extract structured information using VLM.

        Args:
            input_data: Dict containing at least an "image_path" key
            prompt: Instructions for the extraction
            schema: JSON schema defining the output structure

        Returns:
            Dictionary containing the extracted information
        """
        image_path = input_data.get("image_path")
        if not image_path:
            raise ValueError("image_path is required in input_data")

        # Open and encode the image
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            # Convert to base64 for API
            image_base64 = base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            print(f"Error reading image file: {e}")
            return {}

        function_name, tools = create_function_calling_tools(schema)

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Document to process:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ],
                tools=tools,
                tool_choice="any",
                temperature=self.temperature,
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


def process(input_data: Dict[str, Any], prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level process function expected by the processor loader.
    """
    processor = VLMOnlyDocumentProcessor()
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
    Build explanation messages for a mismatch from a VLM (image) extraction task.
    Embeds the source image as a base64 data URI so the LLM can visually verify
    what the document looks like alongside the prediction/ground-truth delta.
    Falls back to a text-only message if the image cannot be read.
    """
    import json as _json

    field = mismatch.get("field", "unknown")
    field_score = mismatch.get("field_score", {})
    image_path = input_data.get("image_path") if isinstance(input_data, dict) else None

    pred_value = prediction.get(field) if isinstance(prediction, dict) else prediction
    gt_value = ground_truth.get(field) if isinstance(ground_truth, dict) else ground_truth

    text_part = {
        "type": "text",
        "text": (
            f"Field with mismatch: {field}\n"
            f"Score: {_json.dumps(field_score)}\n\n"
            f"Predicted value: {_json.dumps(pred_value)}\n"
            f"Expected value:  {_json.dumps(gt_value)}\n\n"
            "The document image is attached below. Explain what caused the mismatch."
        ),
    }

    user_content: list = [text_part]

    if image_path:
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
        except Exception:
            # If the image can't be read, fall back to mentioning the path
            text_part["text"] += f"\n(Image path: {image_path} — could not be loaded)"

    return [{"role": "user", "content": user_content}]
