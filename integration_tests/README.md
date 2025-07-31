# Michigan Guardianship AI - Integration Tests

This directory contains comprehensive integration tests for the Michigan Guardianship AI system.

## Test Structure

### 1. Document-to-Database Pipeline Test (`full_pipeline_test.py`)
- Creates test documents with known content
- Processes documents through chunking pipeline
- Embeds chunks using the embedding model
- Stores in test ChromaDB instance
- Verifies successful storage and retrieval

### 2. Query-to-Response Pipeline Test
Tests the complete flow with "Golden Q&A" pairs:
- **Simple queries**: Filing fees, court location, hearing days
- **Procedural queries**: Required forms, fee waivers, notification requirements
- **Complex scenarios**: ICWA requirements, emergency guardianship
- **Out-of-scope detection**: Adult guardianship, other counties

### 3. End-to-End Test (`end_to_end_test.py`)
- Full pipeline from query to validated response
- Mock LLM integration (can use real OpenAI API if configured)
- Complete validation including:
  - Hallucination detection
  - Citation verification
  - Genesee County specifics
  - Appropriate disclaimers

## Running Tests

### Quick Test (3 sample queries)
```bash
./quick_test.py
```

### Golden Q&A Test Suite
```bash
./run_golden_qa.py
```

### Full Integration Test
```bash
./full_pipeline_test.py
```

### End-to-End Test
```bash
./end_to_end_test.py
```

### Run All Tests
```bash
./run_all_tests.py
```

## Test Data

### Test Documents (`test_documents/`)
- `test_filing_info.txt`: Filing fees, court location, hearing schedule
- `test_icwa_info.txt`: ICWA requirements and procedures
- `test_procedures.txt`: General guardianship procedures

### Golden Q&A Pairs
The test suite includes 10 carefully designed Q&A pairs that test:
1. Factual accuracy (fees, locations, schedules)
2. Procedural correctness (forms, deadlines)
3. Complex scenario handling (ICWA, emergencies)
4. Out-of-scope detection (adult guardianship, other jurisdictions)

## Expected Results

### Pass Criteria
- Document pipeline: All chunks stored and retrievable
- Retrieval: Relevant chunks retrieved for each query type
- Validation: No hallucinations, correct citations, accurate Genesee info
- Latency: Meets budgets (Simple: 800ms, Standard: 1500ms, Complex: 2000ms)

### Known Limitations
- Mock LLM responses are simplified for testing
- Small test corpus (3 documents) vs full knowledge base
- Latency tests depend on hardware performance

## Test Reports

Test results are saved in `test_results/` as JSON files. Run `generate_test_report.py` to create a markdown summary.

## Configuration

The tests use small models by default (`USE_SMALL_MODEL=true`) for faster execution. To test with production models:
```bash
export USE_SMALL_MODEL=false
./run_all_tests.py
```

## Troubleshooting

### ChromaDB Issues
If you see database errors, the test DB may be corrupted:
```bash
rm -rf integration_tests/test_chroma_db/
```

### Model Download Issues
The first run may take longer as models are downloaded. Ensure you have:
- Stable internet connection
- Valid HuggingFace token (if using gated models)
- Sufficient disk space (~500MB for small models)

### Memory Issues
If tests fail with memory errors:
- Use small models (`USE_SMALL_MODEL=true`)
- Run tests individually rather than all at once
- Close other applications