**Observations**  
I've analyzed the existing Michigan guardianship knowledge base system that currently uses ChromaDB for vector storage. The system has well-structured document sources (kb_files/), existing embedding pipelines (embed_kb.py, embed_kb_cloud.py), and comprehensive retrieval infrastructure (adaptive_retrieval.py). The user wants to migrate to Pinecone with enhanced PDF parsing, local LLM chunk summarization, and rich metadata tagging to improve retrieval quality by 15-25%. The existing system already has good foundations with proper chunking, metadata extraction, and testing patterns that can be leveraged for the migration.  
  
**Approach**  
The migration will create a new scripts/migrate_to_pinecone.py script that reuses existing chunking logic while adding enhanced PDF parsing (PyPDF2/Unstructured), local LLM summarization via llm_handler.py, and rich metadata extraction. The script will populate a Pinecone index with proper namespacing and metadata filtering capabilities. A comprehensive A/B testing component will compare retrieval quality between ChromaDB and Pinecone using existing test patterns. The implementation maintains compatibility with the current system while preparing for future phases that will integrate Pinecone into the main retrieval pipeline.

## sequenceDiagram
    participant User
    participant MigrationScript as migrate_to_pinecone.py
    participant DocumentUtils as document_utils.py
    participant LLMHandler as llm_handler.py
    participant EmbeddingUtils as embedding_utils.py
    participant Pinecone
    participant TestScript as test_pinecone_retrieval.py
    participant ChromaDB

    User->>MigrationScript: Run migration
    MigrationScript->>MigrationScript: Load configs (pinecone.yaml, .env)
    MigrationScript->>Pinecone: Initialize client & create index
    
    loop For each document in kb_files/
        MigrationScript->>DocumentUtils: Load & parse document (PDF/TXT)
        DocumentUtils->>DocumentUtils: Extract rich metadata
        DocumentUtils->>DocumentUtils: Apply semantic chunking
        
        loop For each chunk
            MigrationScript->>LLMHandler: Generate chunk summary
            LLMHandler-->>MigrationScript: Return 1-2 sentence summary
            MigrationScript->>EmbeddingUtils: Generate embedding
            EmbeddingUtils-->>MigrationScript: Return vector embedding
        end
        
        MigrationScript->>Pinecone: Batch upsert vectors with metadata
    end
    
    MigrationScript->>MigrationScript: Generate migration report
    MigrationScript-->>User: Migration complete
    
    User->>TestScript: Run A/B testing
    
    loop For each test query
        TestScript->>ChromaDB: Query existing system
        ChromaDB-->>TestScript: Return results
        TestScript->>Pinecone: Query new system
        Pinecone-->>TestScript: Return results
        TestScript->>TestScript: Compare quality metrics
    end
    
    TestScript->>TestScript: Generate comparison report
    TestScript-->>User: Quality analysis complete


## requirments.txt

Add new dependencies for Pinecone migration:

- `pinecone-client` for Pinecone vector database integration
    
- `PyPDF2` for enhanced PDF parsing as alternative to pdfplumber
    
- `unstructured` for advanced document parsing capabilities
    
- `tiktoken` for accurate token counting for OpenAI embeddings
    

These additions support the enhanced PDF parsing and Pinecone integration requirements while maintaining existing functionality.

## .env.exampleMODIFY


Add Pinecone configuration variables to the environment template:

```
# =============================================================================
# PINECONE CONFIGURATION (for vector database migration)
# =============================================================================

# Pinecone API Key (REQUIRED for Pinecone migration)
# Get your key from: https://app.pinecone.io/
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Environment (REQUIRED)
# Available environments: us-east-1-aws, us-west-2-aws, etc.
PINECONE_ENVIRONMENT=us-east-1-aws

# Pinecone Index Configuration
PINECONE_INDEX_NAME=michigan-guardianship-v2
PINECONE_NAMESPACE=genesee-county

# Embedding Configuration for Pinecone
# Supported: openai, google, sentence-transformers
PINECONE_EMBEDDING_PROVIDER=google
PINECONE_EMBEDDING_MODEL=text-embedding-004
PINECONE_EMBEDDING_DIMENSION=768
```

These variables configure Pinecone connection, index settings, and embedding provider for the migration script.



## config/pinecone.yaml (NEW)

References
- config/embedding.yaml
- config/chunking.yaml



Create Pinecone-specific configuration file with index settings, metadata schema, and migration parameters:

```yaml
# Pinecone Configuration for Michigan Guardianship Knowledge Base
index_config:
  name: michigan-guardianship-v2
  dimension: 768  # For Google text-embedding-004
  metric: cosine
  pod_type: p1.x1  # Starter tier alternative: starter
  replicas: 1
  shards: 1
  namespace: genesee-county

embedding_config:
  provider: google  # Options: google, openai, sentence-transformers
  model: text-embedding-004
  batch_size: 100
  normalize: true
  task_type: retrieval_document  # For Google embeddings

metadata_schema:
  # Core document metadata
  source: string  # Document filename
  doc_type: string  # procedure, guidance, form, statute, system_instruction
  jurisdiction: string  # Always "Genesee County"
  last_updated: string  # ISO date format
  
  # Legal-specific metadata for filtering
  topic: string  # filing_requirements, icwa, emergency, etc.
  form_numbers: string  # Comma-separated PC/MC form numbers
  statutes: string  # Comma-separated MCL/MCR references
  deadlines: string  # Comma-separated time periods
  fees: string  # Comma-separated dollar amounts
  
  # Enhanced metadata for improved retrieval
  chunk_summary: string  # LLM-generated chunk summary
  chunk_id: string  # Unique chunk identifier
  chunk_index: integer  # Position within document
  content_length: integer  # Character count
  
  # Retrieval optimization
  complexity_level: string  # simple, standard, complex
  target_audience: string  # petitioner, attorney, court_staff

migration_settings:
  chunk_summarization:
    enabled: true
    max_summary_length: 128
    model_provider: local  # Use local LLM via llm_handler
    fallback_to_truncation: true
  
  duplicate_handling:
    strategy: upsert  # overwrite, skip, error
    id_format: "{source}_{chunk_id}_{hash}"
  
  batch_processing:
    documents_per_batch: 10
    chunks_per_batch: 100
    embedding_batch_size: 50
    retry_attempts: 3
    retry_delay: 2

filter_presets:
  # Common filter combinations for testing
  forms_only:
    doc_type: form
  procedures_only:
    doc_type: procedure
  icwa_related:
    topic: icwa
  emergency_procedures:
    topic: emergency
  filing_requirements:
    topic: filing_requirements
```

This configuration provides comprehensive settings for Pinecone index creation, metadata schema definition, and migration behavior control.


## scripts/utilsNEW

Create utilities directory to house shared document processing functions that will be used by both ChromaDB and Pinecone pipelines.

## scripts/utils/document_utils.pyNEW
References
- scripts/embed_kb.py
- scripts/embed_kb_cloud.py

Extract and enhance document processing utilities from existing `embed_kb.py` to create reusable functions:

**Core Functions to Implement:**

1. **Enhanced PDF Parsing**:
    
    - `extract_pdf_text_pypdf2()` - Alternative PDF extraction using PyPDF2
        
    - `extract_pdf_text_unstructured()` - Advanced parsing using Unstructured library
        
    - `extract_pdf_text_hybrid()` - Fallback chain: Unstructured → PyPDF2 → pdfplumber
        
2. **Improved Metadata Extraction**:
    
    - `extract_rich_metadata()` - Enhanced version of existing metadata extraction
        
    - `classify_document_topic()` - Categorize documents by topic (filing, icwa, emergency, etc.)
        
    - `extract_legal_references()` - Find MCL/MCR citations, form numbers, deadlines
        
    - `determine_target_audience()` - Classify content for petitioners vs attorneys vs court staff
        
3. **Chunking Utilities**:
    
    - `chunk_text_semantic()` - Refactored from `embed_kb.py` with legal pattern preservation
        
    - `merge_short_chunks()` - Combine chunks below minimum size threshold
        
    - `validate_chunk_quality()` - Ensure chunks contain meaningful content
        
4. **Content Enhancement**:
    
    - `generate_chunk_summary()` - Interface to LLM handler for summarization
        
    - `enrich_chunk_metadata()` - Add computed metadata fields
        
    - `calculate_content_metrics()` - Length, complexity, readability scores
        
5. **Utility Functions**:
    
    - `generate_stable_chunk_id()` - Create deterministic chunk identifiers
        
    - `normalize_form_numbers()` - Standardize PC/MC form number formats
        
    - `extract_jurisdiction_info()` - Parse location-specific details
        

Reuse existing logic from `embed_kb.py` while adding enhanced PDF parsing options and richer metadata extraction. Ensure all functions accept configuration parameters for flexibility.

## scripts/utils/embedding_utils.pyNEW

Refeferences
- scripts/embed_kb_cloud.py

Create embedding utilities to support multiple providers (Google, OpenAI, local models):

**Core Functions:**

1. **Embedding Generation**:
    
    - `EmbeddingProvider` class with unified interface
        
    - `GoogleEmbeddingProvider` - Uses Google text-embedding-004 API
        
    - `OpenAIEmbeddingProvider` - Uses OpenAI text-embedding-3-large/small
        
    - `LocalEmbeddingProvider` - Uses sentence-transformers models
        
2. **Batch Processing**:
    
    - `batch_embed_texts()` - Process texts in configurable batch sizes
        
    - `embed_with_retry()` - Retry logic for API failures
        
    - `normalize_embeddings()` - Ensure unit vectors for cosine similarity
        
3. **Provider Management**:
    
    - `get_embedding_provider()` - Factory function based on config
        
    - `validate_embedding_dimensions()` - Check dimension compatibility
        
    - `estimate_embedding_cost()` - Calculate API costs for batch operations
        
4. **Quality Assurance**:
    
    - `validate_embeddings()` - Check for NaN/infinite values
        
    - `compare_embedding_quality()` - A/B test different providers
        
    - `embedding_similarity_test()` - Verify semantic similarity preservation
        

This abstraction allows the migration script to easily switch between embedding providers while maintaining consistent interfaces for both ChromaDB and Pinecone pipelines.


## scripts/migrate_to_pinecone.py
References
- scripts/embed_kb.py
- scripts/embed_kb_cloud.py
- scripts/llm_handler.py
- scripts/log_step.py

Create the main Pinecone migration script that loads documents from `kb_files/`, generates embeddings, and uploads to Pinecone with enhanced metadata:

**Script Structure:**

1. **Configuration Loading**:
    
    - Load settings from `config/pinecone.yaml`, `config/chunking.yaml`, and environment variables
        
    - Validate required API keys (Pinecone, embedding provider)
        
    - Initialize logging using existing `log_step()` pattern from `scripts/log_step.py`
        
2. **Pinecone Setup**:
    
    - Initialize Pinecone client with API key and environment
        
    - Create or connect to index with specified dimensions and metric
        
    - Set up namespace for Genesee County documents
        
    - Validate index configuration matches embedding dimensions
        
3. **Document Loading and Processing**:
    
    - Load documents from same sources as `embed_kb.py`: `kb_files/KB (Numbered)/`, `kb_files/Instructive/`, `kb_files/Court Forms/`
        
    - Use enhanced PDF parsing from `document_utils.py` with fallback chain
        
    - Extract rich metadata including topics, form numbers, statutes, deadlines, fees
        
    - Apply semantic chunking with legal pattern preservation
        
4. **Content Enhancement**:
    
    - Generate chunk summaries using local LLM via `llm_handler.py`
        
    - Create summary prompt: "Summarize this legal document chunk in 1-2 sentences, focusing on key procedures, requirements, or deadlines."
        
    - Handle LLM failures gracefully by truncating original text
        
    - Enrich metadata with computed fields (complexity, target audience)
        
5. **Embedding Generation**:
    
    - Use `embedding_utils.py` to generate embeddings with configured provider
        
    - Process in batches with progress tracking using `tqdm`
        
    - Implement retry logic for API failures
        
    - Validate embedding quality and dimensions
        
6. **Pinecone Upload**:
    
    - Format vectors as `(id, embedding, metadata)` tuples
        
    - Use upsert operations to handle re-runs without duplicates
        
    - Process in batches of 100 vectors per Pinecone limits
        
    - Track upload progress and handle errors
        
7. **Quality Validation**:
    
    - Test basic retrieval with sample queries
        
    - Verify metadata filtering works correctly
        
    - Check index statistics (vector count, dimension)
        
    - Generate migration report with statistics
        

**Key Features:**

- Reuse existing document loading logic from `embed_kb.py`
    
- Enhanced PDF parsing using PyPDF2/Unstructured libraries
    
- Local LLM summarization via `llm_handler.py` integration
    
- Rich metadata tagging for filtered searches
    
- Comprehensive error handling and progress tracking
    
- Deterministic chunk IDs for reproducible migrations
    
- Support for incremental updates and re-runs



## Scripts/test_pinecone_retrieval.py

References
- integration_tests/comprehensive_retrieval_test.py
- scripts/adaptive_retrieval.py

Create comprehensive A/B testing script to compare ChromaDB vs Pinecone retrieval quality:

**Testing Framework:**

1. **Test Query Sets**:
    
    - Reuse test queries from `integration_tests/comprehensive_retrieval_test.py`
        
    - Add domain-specific legal queries focusing on forms, procedures, ICWA, emergency situations
        
    - Include edge cases: typos, long queries, ambiguous terms
        
    - Create golden dataset with expected relevant documents
        
2. **Dual Retrieval System**:
    
    - `ChromaDBRetriever` class wrapping existing ChromaDB collection
        
    - `PineconeRetriever` class for new Pinecone index
        
    - Unified interface for fair comparison
        
    - Same embedding model for both systems
        
3. **Retrieval Quality Metrics**:
    
    - **Precision@K**: Relevant documents in top-K results
        
    - **Recall@K**: Coverage of relevant documents
        
    - **MRR (Mean Reciprocal Rank)**: Position of first relevant result
        
    - **NDCG**: Normalized discounted cumulative gain
        
    - **Semantic Similarity**: Average cosine similarity scores
        
4. **Metadata Filtering Tests**:
    
    - Test Pinecone's filtered search capabilities
        
    - Query with jurisdiction, doc_type, topic filters
        
    - Compare filtered vs unfiltered result quality
        
    - Validate filter accuracy and performance
        
5. **Performance Comparison**:
    
    - Measure query latency for both systems
        
    - Track embedding generation time
        
    - Monitor memory usage during retrieval
        
    - Test concurrent query handling
        
6. **Domain-Specific Evaluation**:
    
    - **Form Retrieval**: Test finding specific PC/MC forms
        
    - **Procedure Accuracy**: Verify step-by-step guidance retrieval
        
    - **ICWA Compliance**: Test tribal law document retrieval
        
    - **Emergency Situations**: Test urgent procedure guidance
        
    - **Fee Information**: Test cost-related query accuracy
        
7. **Report Generation**:
    
    - Detailed comparison report with metrics breakdown
        
    - Query-by-query analysis showing improvements/regressions
        
    - Statistical significance testing
        
    - Recommendations for system optimization
        
    - Visual charts showing performance differences
        

**Expected Improvements to Validate:**

- 15-25% improvement in retrieval recall due to chunk summarization
    
- Better precision through rich metadata filtering
    
- Improved handling of domain-specific legal terminology
    
- Enhanced performance for complex multi-part queries
    

The script will provide concrete evidence of retrieval quality improvements and identify areas where Pinecone outperforms the current ChromaDB system


## scripts/utils/__init__.py

Create package initialization file for the utils module to make it importable:

```python
"""Utility modules for document processing and embedding operations"""

from .document_utils import (
    extract_pdf_text_hybrid,
    extract_rich_metadata,
    chunk_text_semantic,
    generate_chunk_summary
)

from .embedding_utils import (
    get_embedding_provider,
    batch_embed_texts,
    validate_embeddings
)

__all__ = [
    'extract_pdf_text_hybrid',
    'extract_rich_metadata', 
    'chunk_text_semantic',
    'generate_chunk_summary',
    'get_embedding_provider',
    'batch_embed_texts',
    'validate_embeddings'
]
```

This enables clean imports in the migration script and other modules.

## docs/pinecone_migration_guide.md

References
- scripts/adaptive_retrieval.py

Create comprehensive documentation for the Pinecone migration process:

### Pinecone Migration Guide

## Overview

This guide covers migrating the Michigan Guardianship knowledge base from ChromaDB to Pinecone with enhanced PDF parsing, local LLM summarization, and rich metadata tagging.

## Prerequisites

1. **Pinecone Account Setup**:
    
    - Create account at https://app.pinecone.io/
        
    - Generate API key
        
    - Note your environment (e.g., us-east-1-aws)
        
2. **Environment Configuration**:
    
    - Copy `.env.example` to `.env`
        
    - Set `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`
        
    - Configure embedding provider (Google AI recommended)
        
3. **Dependencies**:
    
    - Run `pip install -r requirements.txt` to install new dependencies
        
    - Ensure local LLM is configured in `llm_handler.py`
        

## Migration Process

### Step 1: Configuration Review

- Review `config/pinecone.yaml` settings
    
- Adjust embedding dimensions based on chosen provider
    
- Configure metadata schema for your use case
    

### Step 2: Run Migration

```bash
python scripts/migrate_to_pinecone.py
```

### Step 3: Quality Testing

```bash
python scripts/test_pinecone_retrieval.py
```

## Enhanced Features

### PDF Parsing Improvements

- **Unstructured Library**: Advanced document structure detection
    
- **PyPDF2 Fallback**: Reliable text extraction for complex PDFs
    
- **Hybrid Approach**: Automatic fallback chain for maximum coverage
    

### Local LLM Summarization

- **Per-Chunk Summaries**: 1-2 sentence summaries for each chunk
    
- **Legal Focus**: Emphasizes procedures, requirements, deadlines
    
- **Fallback Handling**: Graceful degradation when LLM unavailable
    

### Rich Metadata Tagging

- **Topic Classification**: filing_requirements, icwa, emergency, etc.
    
- **Legal References**: Automatic extraction of MCL/MCR citations
    
- **Form Numbers**: PC/MC form identification and linking
    
- **Temporal Information**: Deadlines and time periods
    
- **Financial Data**: Fees and cost information
    

### Filtered Search Capabilities

```python
# Example filtered searches
results = index.query(
    vector=query_embedding,
    filter={
        "doc_type": "form",
        "topic": "filing_requirements"
    },
    top_k=10
)
```

## Performance Expectations

- **15-25% Improvement** in retrieval recall
    
- **Better Precision** through metadata filtering
    
- **Reduced Context Rot** via chunk summarization
    
- **Enhanced Domain Specificity** for legal queries
    

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure Pinecone index dimensions match embedding model
    
2. **API Rate Limits**: Adjust batch sizes in configuration
    
3. **Memory Issues**: Use smaller batch sizes for large document sets
    
4. **LLM Timeouts**: Configure appropriate timeout values
    

### Validation Steps

1. Check index statistics in Pinecone console
    
2. Test sample queries with expected results
    
3. Verify metadata filtering functionality
    
4. Compare retrieval quality metrics
    

## Integration with Existing System

This migration prepares for future phases where `adaptive_retrieval.py` will be updated to use Pinecone instead of ChromaDB. The current system remains functional during migration.

## Rollback Plan

The original ChromaDB collection remains untouched. To rollback:

1. Stop using Pinecone endpoints
    
2. Continue using existing ChromaDB retrieval
    
3. Delete Pinecone index if needed
    

## Next Steps

After successful migration:

1. Monitor retrieval quality improvements
    
2. Gather user feedback on search relevance
    
3. Prepare for local query embedding pipeline integration
    
4. Plan production deployment with hybrid architecture