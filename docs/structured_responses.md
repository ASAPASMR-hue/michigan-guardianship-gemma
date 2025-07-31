# Structured Response Documentation

## Overview

The Michigan Guardianship AI backend now returns structured JSON responses that separate data from presentation, enabling precise testing and better UI features.

## Response Schema

### Main Response Structure

```json
{
  "data": {
    "answer_markdown": "Full prose answer with inline citations...",
    "citations": [...],
    "forms": ["PC 651", "PC 654"],
    "fees": ["$175 filing fee"],
    "steps": [...],
    "risk_flags": ["legal_risk"],
    "debug": {...}
  },
  "metadata": {
    "model": "google/gemma-3-4b-it",
    "retrieval_metadata": {...},
    "validation": {...},
    "processing_time": 1.23
  },
  "timestamp": "2025-01-31T12:00:00",
  "answer": "Full prose answer..."  // Backward compatibility
}
```

### Field Descriptions

#### `data.answer_markdown`
- Type: `string`
- Description: The complete prose answer in markdown format with inline citations
- Example: "To file for guardianship in Genesee County, you'll need to complete Form PC 651 [Source: PC 651 Instructions] and pay the $175 filing fee..."

#### `data.citations`
- Type: `array[Citation]`
- Description: All sources referenced in the answer
- Citation structure:
  ```json
  {
    "source_id": "pc651.pdf#chunk_3",
    "title": "Form PC 651 - Petition to Appoint Guardian",
    "url": null
  }
  ```

#### `data.forms`
- Type: `array[string]`
- Description: All form numbers mentioned in the answer
- Example: `["PC 651", "PC 654", "MC 20"]`

#### `data.fees`
- Type: `array[string]`
- Description: All fees mentioned in the answer
- Example: `["$175 filing fee"]`

#### `data.steps`
- Type: `array[Step]`
- Description: Procedural steps if the answer involves a process
- Step structure:
  ```json
  {
    "text": "Complete Form PC 651 (Petition to Appoint Guardian)",
    "citations": [...]
  }
  ```

#### `data.risk_flags`
- Type: `array[string]`
- Description: Risk indicators for the query/response
- Possible values:
  - `"legal_risk"` - Emergency or urgent situation
  - `"out_of_scope"` - Query outside minor guardianship scope
  - `"icwa_sensitive"` - Involves Indian Child Welfare Act
  - `"emergency"` - Emergency guardianship situation
  - `"cps_involved"` - Child Protective Services mentioned

#### `data.debug`
- Type: `object`
- Description: Debug information for developers
- Fields:
  - `retrieval_hits`: Number of documents retrieved
  - `model`: Model used for generation
  - `processing_time`: Total processing time in seconds
  - `complexity`: Query complexity level
  - `retrieval_latency`: Retrieval time in seconds

## API Endpoints

### `/api/ask` (POST)
- **Description**: Standard endpoint with backward compatibility
- **Request**:
  ```json
  {
    "question": "What is the filing fee for guardianship?"
  }
  ```
- **Response**: Full structured response with `answer` field for compatibility

### `/api/ask/structured` (POST)
- **Description**: Returns only structured data format
- **Request**: Same as above
- **Response**: Structured response without backward compatibility field

## Benefits

1. **Precise Testing**
   ```python
   assert response.data.fees[0] == "$175 filing fee"
   assert "PC 651" in response.data.forms
   ```

2. **Better UI**
   - Display forms as clickable chips
   - Show fees in a highlighted box
   - Present steps as a checklist

3. **Export Ready**
   - Structured data can be easily exported to CSV/Excel
   - Analytics on common forms, fees, and issues

4. **Validation**
   - Validate that all required forms are mentioned
   - Ensure fee amounts are correct
   - Check for proper citations

## Example Usage

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:5000/api/ask/structured",
    json={"question": "How do I file for emergency guardianship?"}
)

data = response.json()
forms_needed = data["data"]["forms"]
filing_fee = data["data"]["fees"][0]
steps = data["data"]["steps"]

print(f"Forms needed: {', '.join(forms_needed)}")
print(f"Filing fee: {filing_fee}")
for i, step in enumerate(steps, 1):
    print(f"{i}. {step['text']}")
```

### Testing
```python
# Run the test script
python test_structured_responses.py

# Run integration tests
python -m pytest integration_tests/test_structured_responses.py
```

## Migration Guide

### For Frontend Developers
1. The `/api/ask` endpoint still returns the `answer` field for backward compatibility
2. New features should use the structured `data` object
3. Forms and fees can be displayed as interactive elements

### For Testing
1. Use specific field assertions instead of string matching
2. Validate structured data completeness
3. Test extraction accuracy for forms, fees, and citations