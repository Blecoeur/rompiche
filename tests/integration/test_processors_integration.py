"""
Integration tests for processors.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestProcessorIntegration:
    """Integration tests for processor functionality"""

    def test_vlm_processor_initialization(self, sample_config):
        """Test VLM processor initialization with configuration"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        processor = VLMOnlyDocumentProcessor(config=sample_config)

        # Verify configuration was applied
        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3
        assert processor.tokens_used == 0

        # Verify client was created
        assert hasattr(processor, "client")

    @pytest.mark.skip(reason="OCR dependencies not available in test environment")
    def test_ocr_vlm_processor_initialization(self, sample_config):
        """Test OCR+VLM processor initialization with configuration"""
        from rompiche.processors import OCRVLMDocumentProcessor

        # Add OCR-specific config
        ocr_config = sample_config.copy()
        ocr_config["ocr_languages"] = ["en", "fr"]

        # Mock the OCR import
        with patch("builtins.__import__") as mock_import:

            def mock_import_side_effect(name, *args, **kwargs):
                if name == "pytesseract":
                    return MagicMock()
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_side_effect
            processor = OCRVLMDocumentProcessor(config=ocr_config)

            # Verify configuration was applied
            assert processor.model == "mistral-large-latest"
            assert processor.temperature == 0.3
            assert processor.ocr_languages == ["en", "fr"]
            assert processor.tokens_used == 0

            # Verify client was created
            assert hasattr(processor, "client")

    def test_text_to_json_processor_initialization(self, sample_config):
        """Test TextToJsonProcessor initialization with configuration"""
        from rompiche.processors import TextToJsonProcessor

        processor = TextToJsonProcessor(config=sample_config)

        # Verify configuration was applied
        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3
        assert processor.tokens_used == 0

        # Verify client was created
        assert hasattr(processor, "client")

    @pytest.mark.skip(reason="Example processor not in package")
    def test_example_processor_initialization(self, sample_config):
        """Test ExampleProcessor initialization with configuration"""
        # Import from the actual file path
        import sys
        import os

        sys.path.insert(
            0,
            os.path.dirname(os.path.abspath(__file__)).replace(
                "/tests/integration", ""
            ),
        )
        from example_processor_design import ExampleProcessor

        processor = ExampleProcessor(config=sample_config)

        # Verify configuration was applied
        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3
        assert processor.tokens_used == 0

        # Verify client was created
        assert hasattr(processor, "client")


class TestTokenTracking:
    """Test token tracking functionality"""

    def test_token_tracking_initialization(self):
        """Test that token tracking starts at 0"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        processor = VLMOnlyDocumentProcessor()
        assert processor.get_token_usage() == 0

    @patch("rompiche.processors.vlm_document_processor.mistralai.Mistral")
    def test_token_tracking_with_mock_response(self, mock_mistral):
        """Test token tracking with mocked API response"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        # Create mock response
        mock_response = MagicMock()
        mock_response.usage.total_tokens = 100
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [
            MagicMock(
                function=MagicMock(
                    name="extract_information", arguments='{"test": "value"}'
                )
            )
        ]

        # Mock the client
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response
        mock_mistral.return_value = mock_client

        processor = VLMOnlyDocumentProcessor()

        # Mock the image reading
        with patch("builtins.open", create=True):
            with patch("base64.b64encode", return_value=b"test"):
                processor.process(
                    "dummy_path.png",
                    "test prompt",
                    {"type": "object", "properties": {}},
                )

        # Verify tokens were tracked
        assert processor.get_token_usage() == 100


class TestProcessorExports:
    """Test that processors are properly exported"""

    def test_all_processors_exported(self):
        """Test that all processors can be imported from main module"""
        from rompiche.processors import (
            VLMOnlyDocumentProcessor,
            OCRVLMDocumentProcessor,
            TextToJsonProcessor,
            BaseDocumentProcessor,
            process_vlm_only,
            process_ocr_vlm,
        )

        # Verify all imports work
        assert VLMOnlyDocumentProcessor is not None
        assert OCRVLMDocumentProcessor is not None
        assert TextToJsonProcessor is not None
        assert BaseDocumentProcessor is not None
        assert callable(process_vlm_only)
        assert callable(process_ocr_vlm)

    def test_legacy_functions_work(self):
        """Test that legacy functions still work"""
        from rompiche.processors import process_vlm_only, process_ocr_vlm

        # These should be callable (we won't actually call them to avoid API calls)
        assert callable(process_vlm_only)
        assert callable(process_ocr_vlm)


class TestConfigurationInheritance:
    """Test configuration inheritance in processor hierarchy"""

    def test_base_processor_config_inheritance(self, sample_config):
        """Test that base processor properly handles configuration"""
        from rompiche.processors.vlm_document_processor import BaseDocumentProcessor

        processor = BaseDocumentProcessor(config=sample_config)

        # Verify configuration was applied
        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3
        assert processor.tokens_used == 0

    @pytest.mark.skip(reason="OCR dependencies not available in test environment")
    def test_ocr_vlm_inherits_from_base(self, sample_config):
        """Test that OCRVLMDocumentProcessor inherits from BaseDocumentProcessor"""
        from rompiche.processors import OCRVLMDocumentProcessor, BaseDocumentProcessor

        # Mock the OCR import
        with patch("builtins.__import__") as mock_import:

            def mock_import_side_effect(name, *args, **kwargs):
                if name == "pytesseract":
                    return MagicMock()
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_side_effect
            processor = OCRVLMDocumentProcessor(config=sample_config)

            # Should be instance of both classes
            assert isinstance(processor, OCRVLMDocumentProcessor)
            assert isinstance(processor, BaseDocumentProcessor)


class TestErrorHandling:
    """Test error handling in processors"""

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises appropriate error"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        # Remove API key from environment
        import os

        old_key = os.environ.get("MISTRAL_API_KEY")
        os.environ.pop("MISTRAL_API_KEY", None)

        # Should raise ValueError
        with pytest.raises(ValueError, match="MISTRAL_API_KEY not found"):
            VLMOnlyDocumentProcessor()

        # Restore environment variable
        if old_key:
            os.environ["MISTRAL_API_KEY"] = old_key

    @pytest.mark.skip(reason="OCR dependencies not available in test environment")
    def test_invalid_ocr_engine_raises_error(self):
        """Test that invalid OCR engine raises appropriate error"""
        from rompiche.processors import OCRVLMDocumentProcessor

        # Mock the OCR import to fail
        with patch("builtins.__import__") as mock_import:

            def mock_import_side_effect(name, *args, **kwargs):
                if name == "pytesseract":
                    raise ImportError("Not available")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_side_effect
            with pytest.raises(
                ImportError, match="OCR engine 'tesseract' not available"
            ):
                OCRVLMDocumentProcessor()

    @pytest.mark.skip(reason="OCR dependencies not available in test environment")
    def test_unsupported_ocr_engine_raises_error(self):
        """Test that unsupported OCR engine raises appropriate error"""
        from rompiche.processors import OCRVLMDocumentProcessor

        # Mock the OCR import
        with patch("builtins.__import__") as mock_import:

            def mock_import_side_effect(name, *args, **kwargs):
                if name == "pytesseract":
                    return MagicMock()
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_side_effect
            with pytest.raises(
                ValueError, match="Unsupported OCR engine: invalid_engine"
            ):
                OCRVLMDocumentProcessor(ocr_engine="invalid_engine")
