"""
OCR+VLM document/image to JSON processor using Mistral AI.
Uses OCR for text extraction followed by VLM for structured information extraction.
"""

import json
import os
from typing import Dict, Any, Optional
import mistralai
from dotenv import load_dotenv
from PIL import Image
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


class OCRVLMDocumentProcessor(BaseDocumentProcessor):
    """Processor for document/image to JSON using OCR + VLM pipeline."""

    def __init__(
        self, ocr_engine: str = "tesseract", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OCR+VLM processor.

        Args:
            ocr_engine: OCR engine to use ('tesseract', 'easyocr')
            config: Dictionary containing configuration parameters:
                - model: Model name to use (default: "ministral-3b-latest")
                - temperature: Temperature for generation (default: 0.1)
                - api_key: Optional API key override
                - ocr_languages: List of languages for OCR (default: ['en'])
        """
        # Extract OCR-specific config before calling parent
        ocr_config = config.copy() if config else {}
        self.ocr_languages = ocr_config.pop("ocr_languages", ["en"])

        super().__init__(config)
        self.ocr_engine = ocr_engine

        # Import OCR engine dynamically
        try:
            if self.ocr_engine == "tesseract":
                import pytesseract

                self.ocr = pytesseract
            elif self.ocr_engine == "easyocr":
                import easyocr

                self.ocr = easyocr.Reader(self.ocr_languages)
            else:
                raise ValueError(f"Unsupported OCR engine: {ocr_engine}")
        except ImportError as e:
            raise ImportError(f"OCR engine '{ocr_engine}' not available: {e}")

    def _extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from image using OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text
        """
        try:
            if self.ocr_engine == "tesseract":
                text = self.ocr.image_to_string(Image.open(image_path))
            elif self.ocr_engine == "easyocr":
                results = self.ocr.readtext(image_path)
                text = " ".join([result[1] for result in results])

            return text.strip()
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    def process(
        self, image_path: str, prompt: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an image/document using OCR+VLM pipeline and extract structured information.

        Args:
            image_path: Path to the image/document file
            prompt: Instructions for the extraction
            schema: JSON schema defining the output structure

        Returns:
            Dictionary containing the extracted information
        """
        # Step 1: Extract text using OCR
        extracted_text = self._extract_text_with_ocr(image_path)
        if not extracted_text:
            print("No text extracted via OCR")
            return {}

        function_name, tools = create_function_calling_tools(schema)

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"OCR extracted text to process:\n\n{extracted_text}",
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


def process_ocr_vlm(
    image_path: str, prompt: str, schema: Dict[str, Any], ocr_engine: str = "tesseract"
) -> Dict[str, Any]:
    """
    Legacy function for OCR+VLM processing (backward compatibility).
    """
    processor = OCRVLMDocumentProcessor(ocr_engine=ocr_engine)
    result = processor.process(image_path, prompt, schema)
    if not hasattr(process_ocr_vlm, "tokens_used"):
        process_ocr_vlm.tokens_used = 0
    process_ocr_vlm.tokens_used += processor.tokens_used
    return result
