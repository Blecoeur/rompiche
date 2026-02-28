"""
Unit tests for entry points.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestCLIEntryPoint:
    """Test the CLI entry point"""

    def test_cli_main_import(self):
        """Test that CLI main function can be imported"""
        from rompiche.core.main_optimization_cli import main

        assert callable(main)

    def test_cli_main_function(self):
        """Test that CLI main function exists and is callable"""
        from rompiche.core.main_optimization_cli import main

        # Just verify it's callable, don't actually run it
        assert callable(main)


class TestTUIEntryPoint:
    """Test the TUI entry point"""

    def test_tui_main_import(self):
        """Test that TUI main function can be imported"""
        from rompiche.tui.__main__ import main

        assert callable(main)

    def test_tui_check_dependencies_available(self):
        """Test TUI dependency checking when dependencies are available"""
        from rompiche.tui.dashboard import check_tui_available

        # Should return True since textual is installed
        result = check_tui_available()
        assert result is True

    @patch("rompiche.tui.dashboard.TUI_AVAILABLE", False)
    def test_tui_check_dependencies_unavailable(self):
        """Test TUI dependency checking when dependencies are unavailable"""
        from rompiche.tui.dashboard import check_tui_available

        # Should return False when TUI_AVAILABLE is False
        result = check_tui_available()
        assert result is False

    @patch("rompiche.tui.__main__.check_tui_available")
    def test_tui_main_with_unavailable_dependencies(self, mock_check):
        """Test TUI main function when dependencies are unavailable"""
        from rompiche.tui.__main__ import main

        # Mock the check to return False
        mock_check.return_value = False

        # Capture print output
        with patch("builtins.print") as mock_print:
            result = main()

            # Should return 1 (error code)
            assert result == 1

            # Should print error messages
            mock_print.assert_any_call("Error: TUI dependencies not available.")
            mock_print.assert_any_call(
                "Please install them with: pip install rompiche[tui]"
            )

    @patch("rompiche.tui.__main__.check_tui_available")
    @patch("rompiche.tui.__main__.LiveDashboard")
    def test_tui_main_with_available_dependencies(self, mock_dashboard, mock_check):
        """Test TUI main function when dependencies are available"""
        from rompiche.tui.__main__ import main

        # Mock the check to return True
        mock_check.return_value = True

        # Mock the dashboard
        mock_instance = MagicMock()
        mock_dashboard.return_value = mock_instance

        # Call main
        result = main()

        # Should return None (no error)
        assert result is None

        # Should create and run the dashboard
        mock_dashboard.assert_called_once()
        mock_instance.run.assert_called_once()


class TestMainPy:
    """Test the main.py entry point"""

    def test_main_import(self):
        """Test that main module can be imported"""
        import main

        assert hasattr(main, "main")
        assert callable(main.main)

    def test_main_function(self):
        """Test that main function is callable"""
        import main

        with patch("builtins.print") as mock_print:
            main.main()
            # Should print helpful information
            mock_print.assert_any_call("Rompiche - AI-powered document processing")
            mock_print.assert_any_call("Available commands:")


class TestPyProjectToml:
    """Test pyproject.toml configuration"""

    def test_entry_points_in_pyproject(self):
        """Test that entry points are configured in pyproject.toml"""
        with open("pyproject.toml", "r") as f:
            content = f.read()

        # Check for scripts section
        assert "[project.scripts]" in content

        # Check for rompiche entry point
        assert "rompiche = " in content
        assert "rompiche.core.main_optimization_cli:main" in content

        # Check for rompiche-tui entry point
        assert "rompiche-tui = " in content
        assert "rompiche.tui.__main__:main" in content


class TestConfigurationInjection:
    """Test configuration injection in processors"""

    def test_vlm_processor_config_injection(self, sample_config):
        """Test VLM processor accepts configuration"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        processor = VLMOnlyDocumentProcessor(config=sample_config)

        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3

    @pytest.mark.skip(reason="OCR dependencies not available in test environment")
    def test_ocr_vlm_processor_config_injection(self, sample_config):
        """Test OCR+VLM processor accepts configuration"""
        from rompiche.processors import OCRVLMDocumentProcessor

        # Add OCR-specific config
        ocr_config = sample_config.copy()
        ocr_config["ocr_languages"] = ["en", "fr"]

        # Mock the OCR import to avoid dependency issues
        with patch("builtins.__import__") as mock_import:

            def mock_import_side_effect(name, *args, **kwargs):
                if name == "pytesseract":
                    return MagicMock()
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = mock_import_side_effect
            processor = OCRVLMDocumentProcessor(config=ocr_config)

            assert processor.model == "mistral-large-latest"
            assert processor.temperature == 0.3
            assert processor.ocr_languages == ["en", "fr"]

    def test_text_to_json_processor_config_injection(self, sample_config):
        """Test TextToJsonProcessor accepts configuration"""
        from rompiche.processors import TextToJsonProcessor

        processor = TextToJsonProcessor(config=sample_config)

        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3

    @pytest.mark.skip(reason="Example processor not in package")
    def test_example_processor_config_injection(self, sample_config):
        """Test ExampleProcessor accepts configuration"""
        # Import from the actual file path
        import sys
        import os

        sys.path.insert(
            0, os.path.dirname(os.path.abspath(__file__)).replace("/tests/unit", "")
        )
        from example_processor_design import ExampleProcessor

        processor = ExampleProcessor(config=sample_config)

        assert processor.model == "mistral-large-latest"
        assert processor.temperature == 0.3

    @pytest.mark.skip(reason="Example processor not in package")
    def test_default_config_values(self):
        """Test that processors work with default configuration"""
        from rompiche.processors import VLMOnlyDocumentProcessor, TextToJsonProcessor

        # Import example processor from file path
        import sys
        import os

        sys.path.insert(
            0, os.path.dirname(os.path.abspath(__file__)).replace("/tests/unit", "")
        )
        from example_processor_design import ExampleProcessor

        # Test VLM processor
        vlm_processor = VLMOnlyDocumentProcessor()
        assert vlm_processor.model == "ministral-3b-latest"
        assert vlm_processor.temperature == 0.1

        # Test TextToJsonProcessor
        text_processor = TextToJsonProcessor()
        assert text_processor.model == "ministral-3b-latest"
        assert text_processor.temperature == 0.1

        # Test ExampleProcessor
        example_processor = ExampleProcessor()
        assert example_processor.model == "ministral-3b-latest"
        assert example_processor.temperature == 0.1

    def test_api_key_override(self, sample_config):
        """Test that API key can be overridden in config"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        custom_config = sample_config.copy()
        custom_config["api_key"] = "custom-api-key"

        processor = VLMOnlyDocumentProcessor(config=custom_config)

        # The client should be initialized with the custom API key
        # We can't easily test the actual API key without mocking, but we can verify
        # that the processor was created successfully
        assert processor.model == "mistral-large-latest"

    def test_api_key_from_env(self):
        """Test that API key falls back to environment variable"""
        from rompiche.processors import VLMOnlyDocumentProcessor

        # Set environment variable
        import os

        os.environ["MISTRAL_API_KEY"] = "env-api-key"

        # Create processor without config
        processor = VLMOnlyDocumentProcessor()

        # Should use environment API key
        assert processor.model == "ministral-3b-latest"

    def test_missing_api_key_error(self):
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
