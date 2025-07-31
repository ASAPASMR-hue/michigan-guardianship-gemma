# Michigan Guardianship AI

A production-ready RAG (Retrieval-Augmented Generation) system for Genesee County's minor guardianship procedures.

## Overview

This system provides accurate, accessible information about minor guardianship procedures, requirements, and forms specific to Genesee County, Michigan. It features:

- Zero hallucination policy with citation verification
- Adaptive retrieval with keyword boosting
- Semantic similarity validation
- Support for ICWA (Indian Child Welfare Act) requirements
- Integration with multiple LLMs for testing
- User-friendly web interface

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: 3.8 or higher (tested with 3.8 through 3.12)
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: ~2GB free space for models and dependencies
- **API Keys Required**:
  - **Google AI API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
  - **HuggingFace Token**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ASAPASMR-hue/michigan-guardianship-ai.git
   cd michigan-guardianship-ai
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```bash
   # Google AI API Key (required for LLM and cloud embeddings)
   GOOGLE_AI_API_KEY=your_google_api_key_here
   GEMINI_API_KEY=your_google_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   
   # HuggingFace Token (required for model downloads)
   HUGGING_FACE_HUB_TOKEN=your_hf_token_here
   HUGGINGFACE_TOKEN=your_hf_token_here
   
   # Port configuration (default: 5000, use 5001 on macOS to avoid conflicts)
   PORT=5001
   
   # Use small models for testing/development (recommended for initial setup)
   USE_SMALL_MODEL=true
   ```

5. **Create the vector database**
   
   Choose one of the following options:
   
   **Option A: Cloud-based embeddings (Recommended for beginners)**
   ```bash
   python scripts/embed_kb_cloud.py
   ```
   This uses Google AI's embedding API - faster and avoids memory issues.
   
   **Option B: Local embeddings**
   ```bash
   python scripts/embed_kb.py
   ```
   This downloads and runs models locally - fully offline but requires more resources.

6. **Start the application**
   ```bash
   # Set environment variables and start
   export PORT=5001  # On Windows: set PORT=5001
   export USE_SMALL_MODEL=true  # On Windows: set USE_SMALL_MODEL=true
   python app.py
   ```

7. **Open your browser**
   Navigate to [http://127.0.0.1:5001](http://127.0.0.1:5001)

### First-Time Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed via pip
- [ ] `.env` file created with API keys
- [ ] Vector database created (using either cloud or local embeddings)
- [ ] Server started successfully
- [ ] Web interface accessible in browser

## Common Issues and Troubleshooting

### 1. Port Conflicts (Common on macOS)
**Error**: "Address already in use" or connection refused
```bash
# Solution: Use port 5001 instead of 5000
export PORT=5001
python app.py

# Or add to .env file:
echo "PORT=5001" >> .env
```

### 2. Memory Issues (Metal Performance Shaders on macOS)
**Error**: "MPSNDArray initWithDevice:descriptor: Error: total bytes of NDArray > 2**32"
```bash
# Solution 1: Use cloud embeddings instead
python scripts/embed_kb_cloud.py

# Solution 2: Use smaller models
echo "USE_SMALL_MODEL=true" >> .env
```

### 3. Embedding Dimension Mismatch
**Error**: "Collection expecting embedding with dimension of X, got Y"
```bash
# Solution: Clear database and re-embed with consistent model
rm -rf chroma_db/
export USE_SMALL_MODEL=true
python scripts/embed_kb.py  # or embed_kb_cloud.py
```

### 4. API Key Issues
- **Google API Key**: Must start with "AI" - get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **HuggingFace Token**: Must start with "hf_" - get from [HuggingFace Settings](https://huggingface.co/settings/tokens)
- **Multiple key variables**: The system looks for GOOGLE_AI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY

### 5. Model Download Failures
```bash
# For slow connections, increase timeout:
export HF_HUB_DOWNLOAD_TIMEOUT=3600

# For corporate proxies:
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 6. Installation Issues
**Windows**: Install Microsoft C++ Build Tools
**macOS**: `xcode-select --install`
**Linux**: `sudo apt-get install python3-dev`

### 7. Content KeyError
**Error**: KeyError: 'content'
- This happens when retrieval returns 'document' key instead of 'content'
- Solution: Update to latest version which handles both keys

For additional help, please check the [issues page](https://github.com/ASAPASMR-hue/michigan-guardianship-ai/issues) or create a new issue with your error details.

## Embedding Options

### Cloud-Based Embeddings (Recommended)
Use Google AI's text-embedding API for faster, more reliable embeddings:
```bash
python scripts/embed_kb_cloud.py
```
**Pros**: Fast, no memory issues, works on any hardware
**Cons**: Requires internet for initial setup, uses API quota

### Local Embeddings
Download and run embedding models locally:
```bash
# For production (requires 8GB+ RAM):
python scripts/embed_kb.py

# For development/testing (requires 4GB RAM):
export USE_SMALL_MODEL=true
python scripts/embed_kb.py
```
**Pros**: Fully offline after model download, no API limits
**Cons**: Slower, requires more memory, may crash on macOS with large models

## What You Can Ask

Once running, you can ask questions like:
- "How much does it cost to file for guardianship in Genesee County?"
- "What forms do I need to file for guardianship?"
- "What are the requirements to become a guardian?"
- "How do I file for emergency guardianship?"
- "What is the ICWA and when does it apply?"

The system provides:
- **Accurate legal information** with citations to Michigan statutes
- **Genesee County specifics** like filing fees ($175) and court location
- **Step-by-step guidance** for guardianship procedures
- **Form recommendations** with links to official documents

## Key Features

### 1. Document Processing
- Semantic chunking with legal pattern recognition
- Metadata extraction for forms and deadlines
- Choice of cloud or local embeddings

### 2. Adaptive Retrieval
- Query complexity classification (simple/standard/complex)
- Hybrid search combining vector and keyword matching
- Dynamic performance optimization

### 3. Response Validation
- Zero hallucination policy with citation verification
- Procedural accuracy checks
- Legal disclaimer insertion

### 4. Genesee County Specifics
- Filing fee: $175 (fee waiver via Form MC 20)
- Court location: 900 S. Saginaw Street, Flint, MI 48502
- Hearings: Thursdays at 9:00 AM

## Project Structure

```
michigan-guardianship-ai/
├── scripts/              # Core pipeline scripts
│   ├── embed_kb.py      # Local document embedding
│   ├── embed_kb_cloud.py # Cloud-based embedding
│   ├── production_pipeline.py # Main RAG pipeline
│   └── app.py           # Flask web server
├── kb_files/            # Knowledge base documents
│   ├── KB (Numbered)/   # Main knowledge documents
│   ├── Court Forms/     # Official forms
│   └── Instructive/     # System instructions
├── config/              # Configuration files
└── chroma_db/           # Vector database storage
```

## License

[License information here]

## Contact

[Contact information here]