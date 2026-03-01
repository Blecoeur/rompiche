"""
Example processor design pattern.
This demonstrates how to create a processor that follows the rompiche architecture.
"""

from typing import Dict, Any
from dotenv import load_dotenv
from rompiche.processors.base_processor import BaseProcessor
from rompiche.utils.processor_utils import create_function_calling_tools

# Load environment variables
load_dotenv()


class ExampleProcessor(BaseProcessor):
    """
    Example processor that demonstrates the design pattern.
    This processor extracts structured information from text.
    """

    def process(
        self, input_data: Dict[str, Any], prompt: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process input data and extract structured information based on the provided schema.

        Args:
            input_data: Dict with input fields (e.g. {"text": "..."} or {"image_path": "..."})
            prompt: Instructions for the extraction
            schema: JSON schema defining the output structure

        Returns:
            Dictionary containing the extracted information
        """
        function_name, tools = create_function_calling_tools(schema)

        try:
            response = None  # noqa: F841 # Here we would probably call the LLM provider client

            # Here we would probably track the tokens used

            # Here we would probably extract the structured data from the response

            return {}

        except Exception as e:
            print(f"Error calling the LLM provider client: {e}")
            return {}
