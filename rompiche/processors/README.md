# Processors Module

This module contains different processor implementations for various tasks in the rompiche framework.

## Available Processors

### 1. Text to JSON Processor
**File:** `text_to_json_processor.py`

Extracts structured information from text using Mistral AI's function calling capabilities.

**Usage:**
```python
from rompiche.processors.text_to_json_processor import TextToJsonProcessor

processor = TextToJsonProcessor()
result = processor.process(
    input_text="The event is on 2026-03-11 at 9:52 AM",
    prompt="Extract the date and time",
    schema={
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "time": {"type": "string"}
        },
        "required": ["date", "time"]
    }
)
print(result)
print(f"Tokens used: {processor.get_token_usage()}")
```

### 2. Document/Image to JSON Processors
**File:** `document_to_json_processor.py`

Two variants for processing documents and images:

#### VLM-Only Processor
Uses Vision-Language Model directly on images/documents.

**Usage:**
```python
from rompiche.processors.document_to_json_processor import VLMOnlyDocumentProcessor

processor = VLMOnlyDocumentProcessor()
result = processor.process(
    image_path="invoice.jpg",
    prompt="Extract invoice details",
    schema={
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total": {"type": "string"}
        },
        "required": ["invoice_number", "date", "total"]
    }
)
```

#### OCR+VLM Processor
Uses OCR to extract text first, then processes with VLM.

**Usage:**
```python
from rompiche.processors.document_to_json_processor import OCRVLMDocumentProcessor

# Choose OCR engine: "tesseract" or "easyocr"
processor = OCRVLMDocumentProcessor(ocr_engine="tesseract")
result = processor.process(
    image_path="form_scan.jpg",
    prompt="Extract form fields",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {"type": "string"}
        },
        "required": ["name", "address"]
    }
)
```

## Processor Design Pattern

All processors follow this pattern:

1. **Base Class**: `BaseProcessor` provides common functionality
   - API key management
   - Token tracking
   - Mistral client initialization

2. **Processor Class**: Implement specific processing logic
   - Inherit from `BaseProcessor`
   - Implement `process()` method
   - Track token usage via `_track_tokens()`

3. **Legacy Function**: Optional backward-compatible function
   - Creates processor instance
   - Calls `process()` method

## Creating Your Own Processor

```python
from rompiche.processors import BaseProcessor

class MyCustomProcessor(BaseProcessor):
    def process(self, input_data, prompt, schema):
        # Your processing logic here
        response = self.client.chat.complete(...)
        self._track_tokens(response)
        # Extract and return structured data
        return result
```

## Token Tracking

All processors automatically track token usage:

```python
processor = TextToJsonProcessor()
result = processor.process(...)
print(f"Tokens used: {processor.get_token_usage()}")
```

## Error Handling

Processors handle errors gracefully and return empty dictionaries on failure:

```python
result = processor.process(...)
if not result:
    print("Processing failed")
```
