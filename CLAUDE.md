# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Michigan Guardianship AI is a production-ready RAG (Retrieval-Augmented Generation) system that helps families navigate minor guardianship procedures in Genesee County, Michigan. The system emphasizes zero hallucination, maximum actionability, and legal compliance.

## Common Development Commands

### Full Pipeline Execution
```bash
# Run complete Phase 1-3 pipeline
make phase1

# Test with small models (faster)
make test-phase1

# Clean and rebuild everything
make clean-all && make phase1
```

### Individual Component Commands
```bash
# Document embedding
python scripts/embed_kb.py

# Retrieval setup
python scripts/retrieval_setup.py

# Validation configuration
python scripts/validator_setup.py

# Run evaluation rubric
python scripts/eval_rubric.py

# Train complexity classifier
python scripts/train_complexity_classifier.py

# Run Phase 3 tests (when embedding auth is fixed)
python scripts/run_phase3_tests.py
```

### Testing Specific Models
```bash
# Test a specific model
python scripts/run_phase3_tests.py --model "google/gemma-2b-it"

# Run battle mode comparison
python scripts/battle_mode_comparison.py
```

### Utility Commands
```bash
# Extract configurations from documentation
make extract-configs

# Clean ChromaDB
make clean-chroma

# View logs
tail -f logs/phase3_testing_*.log
```

## High-Level Architecture

### Core Pipeline Flow
1. **Document Processing** (`embed_kb.py`)
   - Semantic chunking with legal pattern recognition
   - Metadata extraction (forms, statutes, deadlines)
   - BAAI/bge-m3 embeddings → ChromaDB storage

2. **Adaptive Retrieval** (`adaptive_retrieval.py`)
   - Query complexity classification (simple/standard/complex/crisis)
   - Dynamic parameter adjustment (top_k, rewrites, rerank_k)
   - Hybrid search: 70% vector + 30% BM25
   - Latency budget enforcement (800-2500ms based on complexity)

3. **Response Generation** (`run_phase3_tests.py`)
   - 11 models configured (HuggingFace, local, Google)
   - Dynamic mode switching (STRICT for facts, PERSONALIZED for guidance)
   - Genesee County constants injection
   - Legal disclaimer insertion

4. **Validation** (`validator_setup.py`)
   - Hallucination detection via LettuceDetect
   - Citation verification (every legal fact must cite source)
   - Out-of-scope pattern matching
   - Procedural accuracy checks

### Configuration Management
All configurations are centralized in `/config/`:
- `chunking_config.yaml` - Document processing parameters
- `embedding_config.yaml` - Model and vector DB settings
- `retrieval_config.yaml` - Search pipeline configuration
- `validation_config.yaml` - Quality assurance rules
- `model_configs.yaml` - LLM configurations for all 11 models

### Key Design Principles
1. **Zero Hallucination**: Every legal statement must have inline citation
2. **Genesee County Focus**: All responses include local specifics ($175 fee, Thursday hearings, etc.)
3. **Adaptive Complexity**: Simple questions get fast answers; complex scenarios get thorough analysis
4. **Legal Compliance**: Clear disclaimers per Michigan Rule 7.1
5. **Human-Centered Quality**: Scored on actionability, not just accuracy

### Current Development Status
**Phase 3: Testing & Refinement** (Active)
- Test infrastructure complete
- 400+ test questions prepared
- Battle mode comparison framework ready
- **Blocker**: HuggingFace embedding model authentication needs resolution before full testing

### Critical Files to Understand
1. `docs/Project_Guidance_v2.1.md` - Master technical and evaluation playbook
2. `scripts/run_phase3_tests.py` - Main testing entry point
3. `config/model_configs.yaml` - All model configurations
4. `patterns/out_of_scope_patterns.json` - Scope boundary definitions
5. `rubrics/evaluation_rubric.yaml` - 10-point quality scoring system

### Testing Approach
When implementing new features:
1. Add test cases to `data/test_questions_dataset.csv`
2. Update complexity tiers if needed in `rubrics/question_tiers.yaml`
3. Run isolated tests with `make test-phase1`
4. Validate with full pipeline using `make phase1`

### Performance Targets
- Retrieval precision: >0.85
- Hallucination rate: <0.5%
- P95 latency: <1000ms (simple), <1800ms (standard), <2500ms (complex)
- Procedural accuracy: >98%
- Actionability score: >85%