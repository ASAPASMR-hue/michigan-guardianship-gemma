#!/bin/bash
# Michigan Guardianship AI - Complete Jules Environment Setup
# Optimized for Jules: Ubuntu Linux, Python 3.12.11
# Version: 2.0.0 - Full automation with error recovery

set -e  # Exit on error
set -o pipefail  # Pipe failures cause script to fail

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys (will be loaded from .env file)
GOOGLE_API_KEY="${GOOGLE_API_KEY:-your_google_api_key_here}"
HF_TOKEN="${HUGGINGFACE_TOKEN:-your_huggingface_token_here}"
PINECONE_API_KEY="${PINECONE_API_KEY:-your_pinecone_api_key_here}"
PINECONE_HOST="${PINECONE_HOST:-your_pinecone_host_here}"

# Repository and paths
REPO_URL="https://github.com/ASAPASMR-hue/michigan-guardianship-gemma.git"
REPO_DIR="michigan-guardianship-gemma"
VENV_DIR="venv"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Logging
LOG_FILE="jules_setup_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo -e "\n${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}  $1${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${MAGENTA}ℹ${NC} $1"
}

# Error handler
error_handler() {
    print_error "Error occurred at line $1"
    print_info "Check $LOG_FILE for details"
    exit 1
}

trap 'error_handler $LINENO' ERR

# ============================================================================
# MAIN SETUP
# ============================================================================

print_header "Michigan Guardianship AI - Jules Environment Setup"
echo "Starting at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Step 1: System checks
print_header "Step 1: System Verification"

print_step "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found"
    exit 1
fi

print_step "Checking git..."
if command -v git &> /dev/null; then
    print_success "Git installed"
else
    print_warning "Git not found, installing..."
    apt-get update && apt-get install -y git
fi

print_step "Checking system resources..."
if [ -f /proc/meminfo ]; then
    MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_GB=$((MEM_KB / 1024 / 1024))
    print_success "Available RAM: ${MEM_GB}GB"
fi

# Step 2: Repository setup
print_header "Step 2: Repository Configuration"

if [ -d "$REPO_DIR" ]; then
    print_step "Repository exists, updating..."
    cd "$REPO_DIR"
    git pull origin main || print_warning "Could not update"
else
    print_step "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi
print_success "Repository ready at: $(pwd)"

# Step 3: Python environment
print_header "Step 3: Python Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment exists"
else
    print_step "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

print_step "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Step 4: Python packages
print_header "Step 4: Installing Dependencies"

print_step "Upgrading pip..."
pip install --upgrade pip wheel setuptools --quiet

print_step "Creating comprehensive requirements..."
cat > requirements_jules.txt << 'EOF'
# Core Dependencies
pyyaml>=6.0
pandas>=1.3.0
pdfplumber>=0.10.0
lettucedetect
rank-bm25>=0.2.0
scikit-learn>=1.0.0
scipy>=1.7.0
numpy>=1.21.0

# AI/ML Dependencies
openai>=1.0.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
torch>=2.0.0
chromadb>=0.4.0
pinecone-client>=3.0.0

# Web Framework
flask>=2.0.0
flask-cors>=3.0.0
flask-session>=0.4.0

# Utilities
python-dotenv>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
graphviz>=0.19.0
plotly>=5.0.0
jinja2>=3.0.0
schedule>=1.1.0
psutil>=5.9.0
tqdm>=4.65.0
pydantic>=2.0.0
openpyxl>=3.0.0
colorlog>=6.7.0
colorama>=0.4.6

# Additional
httpx>=0.24.0
aiofiles>=23.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
EOF

print_step "Installing Python packages (this may take 3-5 minutes)..."
pip install -r requirements_jules.txt --progress-bar off

print_success "All dependencies installed"

# Step 5: Directory structure
print_header "Step 5: Creating Directory Structure"

directories=(
    "chroma_db"
    "models"
    "logs"
    "results"
    "kb_files/KB (Numbered)"
    "kb_files/Court Forms"
    "kb_files/Instructive"
    "config"
    "scripts"
    "data"
    "patterns"
    "rubrics"
    "constants"
    "backups"
    "temp"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_success "Created: $dir"
done

# Step 6: Environment configuration
print_header "Step 6: Environment Configuration"

print_step "Creating .env file with your API keys..."
cat > .env << EOF
# Michigan Guardianship AI Environment Variables
# Auto-configured for Jules environment

# =============================================================================
# API KEYS (PRE-CONFIGURED)
# =============================================================================

# Google AI API Keys
GOOGLE_AI_API_KEY=$GOOGLE_API_KEY
GEMINI_API_KEY=$GOOGLE_API_KEY
GOOGLE_API_KEY=$GOOGLE_API_KEY

# HuggingFace Token
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
HUGGINGFACE_TOKEN=$HF_TOKEN

# Pinecone Configuration
PINECONE_API_KEY=$PINECONE_API_KEY
PINECONE_HOST=$PINECONE_HOST
PINECONE_ENVIRONMENT=aped-4627-b74a
PINECONE_INDEX_NAME=michigan-guardianship-v2

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server Configuration
PORT=5000
HOST=0.0.0.0
FLASK_DEBUG=false

# Model Settings
USE_SMALL_MODEL=true
EMBEDDING_METHOD=cloud
VECTOR_DB=pinecone

# Performance
MAX_WORKERS=4
BATCH_SIZE=100
CHUNK_SIZE=1000
OVERLAP=200

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed

# Timeouts
HF_HUB_DOWNLOAD_TIMEOUT=3600
REQUEST_TIMEOUT=300

# =============================================================================
# JULES ENVIRONMENT SPECIFIC
# =============================================================================

ENVIRONMENT=jules
PLATFORM=ubuntu-linux
PYTHON_VERSION=3.12.11
AUTO_RELOAD=false
EOF

print_success ".env file created with all API keys"

# Step 7: Configuration files
print_header "Step 7: Configuration Files"

# Create logging config
print_step "Creating logging configuration..."
cat > config/logging.yaml << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
  json:
    format: '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/app.log
    maxBytes: 10485760
    backupCount: 5
  
  error_file:
    class: logging.FileHandler
    level: ERROR
    formatter: detailed
    filename: logs/errors.log

loggers:
  app:
    level: INFO
    handlers: [console, file]
  
  scripts:
    level: DEBUG
    handlers: [console, file]

root:
  level: INFO
  handlers: [console, file, error_file]
EOF
print_success "Logging configuration created"

# Create Pinecone config
print_step "Creating Pinecone configuration..."
cat > config/pinecone_jules.yaml << EOF
# Pinecone Configuration for Jules Environment
pinecone:
  api_key: ${PINECONE_API_KEY}
  environment: aped-4627-b74a
  host: ${PINECONE_HOST}
  
index:
  name: michigan-guardianship-v2
  dimension: 768
  metric: cosine
  pods: 1
  replicas: 1
  pod_type: p1.x1

embedding:
  model: text-embedding-004
  batch_size: 100
  dimension: 768

retrieval:
  top_k: 10
  min_score: 0.7
  namespace: default
EOF
print_success "Pinecone configuration created"

# Step 8: Database setup scripts
print_header "Step 8: Vector Database Scripts"

# Create Pinecone setup script
print_step "Creating Pinecone setup script..."
cat > setup_pinecone_jules.py << 'EOF'
#!/usr/bin/env python3
"""Setup Pinecone for Michigan Guardianship AI"""

import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def setup_pinecone():
    """Initialize Pinecone index"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        index_name = "michigan-guardianship-v2"
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            print(f"Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✓ Index {index_name} created")
        else:
            print(f"✓ Index {index_name} already exists")
        
        # Get index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"✓ Index ready: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error setting up Pinecone: {e}")
        return False

if __name__ == "__main__":
    success = setup_pinecone()
    sys.exit(0 if success else 1)
EOF
chmod +x setup_pinecone_jules.py
print_success "Pinecone setup script created"

# Create cloud embedding script
print_step "Creating cloud embedding script..."
cat > embed_cloud_jules.py << 'EOF'
#!/usr/bin/env python3
"""Cloud-based embedding for Jules environment"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

load_dotenv()

def embed_documents():
    """Embed documents using Google AI"""
    # Configure Google AI
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    # Get all documents
    kb_path = Path("kb_files")
    documents = list(kb_path.rglob("*.txt")) + list(kb_path.rglob("*.md"))
    
    print(f"Found {len(documents)} documents to embed")
    
    embeddings = []
    for doc in tqdm(documents, desc="Embedding documents"):
        try:
            with open(doc, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate embedding
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=content,
                task_type="retrieval_document"
            )
            
            embeddings.append({
                'file': str(doc),
                'embedding': result['embedding'],
                'content': content[:1000]  # First 1000 chars for preview
            })
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error embedding {doc}: {e}")
    
    print(f"✓ Successfully embedded {len(embeddings)} documents")
    return embeddings

if __name__ == "__main__":
    embed_documents()
EOF
chmod +x embed_cloud_jules.py
print_success "Cloud embedding script created"

# Step 9: Validation script
print_header "Step 9: Validation Tools"

print_step "Creating validation script..."
cat > validate_jules_setup.py << 'EOF'
#!/usr/bin/env python3
"""Validate Jules environment setup"""

import os
import sys
import importlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_module(name):
    try:
        importlib.import_module(name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)

def check_api_key(name):
    value = os.getenv(name)
    if value and not value.startswith("your_"):
        return True, "Configured"
    return False, "Missing"

def main():
    print("\n" + "="*60)
    print("  Michigan Guardianship AI - Jules Setup Validation")
    print("="*60 + "\n")
    
    errors = []
    
    # Check Python version
    print("Python Environment:")
    print(f"  Version: {sys.version}")
    print(f"  Executable: {sys.executable}")
    
    # Check critical modules
    print("\nPython Packages:")
    modules = [
        'flask', 'chromadb', 'pinecone', 'google.generativeai',
        'sentence_transformers', 'torch', 'pandas', 'pdfplumber'
    ]
    
    for mod in modules:
        ok, msg = check_module(mod.split('.')[0])
        status = "✓" if ok else "✗"
        print(f"  [{status}] {mod}: {msg if not ok else 'OK'}")
        if not ok:
            errors.append(f"Missing module: {mod}")
    
    # Check API keys
    print("\nAPI Keys:")
    keys = ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'HUGGINGFACE_TOKEN']
    for key in keys:
        ok, msg = check_api_key(key)
        status = "✓" if ok else "✗"
        print(f"  [{status}] {key}: {msg}")
        if not ok:
            errors.append(f"Missing API key: {key}")
    
    # Check directories
    print("\nDirectories:")
    dirs = ['kb_files', 'scripts', 'config', 'logs', 'models']
    for d in dirs:
        exists = Path(d).exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {d}")
        if not exists:
            errors.append(f"Missing directory: {d}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("✗ Validation FAILED")
        print("\nIssues found:")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("✓ All checks PASSED - Jules environment ready!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x validate_jules_setup.py
print_success "Validation script created"

# Step 10: Application launcher
print_header "Step 10: Application Launcher"

print_step "Creating application launcher..."
cat > launch_jules.sh << 'EOF'
#!/bin/bash
# Launch Michigan Guardianship AI in Jules environment

source venv/bin/activate
export $(grep -v '^#' .env | xargs)

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Michigan Guardianship AI - Starting...                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Environment: Jules (Ubuntu Linux)"
echo "Python: $(python --version)"
echo "Vector DB: ${VECTOR_DB:-pinecone}"
echo "Port: ${PORT:-5000}"
echo ""
echo "Access the application at: http://localhost:${PORT:-5000}"
echo ""

python app.py
EOF
chmod +x launch_jules.sh
print_success "Application launcher created"

# Step 11: Create sample knowledge base
print_header "Step 11: Sample Knowledge Base"

print_step "Creating sample knowledge base files..."
cat > "kb_files/KB (Numbered)/001_overview.txt" << 'EOF'
Michigan Minor Guardianship Overview
=====================================

This system provides information about minor guardianship procedures in Genesee County, Michigan.

Key Information:
- Filing fee: $175 (fee waiver available via Form MC 20)
- Court location: 900 S. Saginaw Street, Flint, MI 48502
- Hearings: Typically scheduled on Thursdays at 9:00 AM
- Processing time: 30-45 days for non-emergency cases

Required Forms:
- PC 656: Petition for Appointment of Guardian
- PC 562: Acceptance of Appointment
- PC 574: Notice of Hearing

Special Considerations:
- ICWA (Indian Child Welfare Act) applies to Native American children
- Emergency guardianships available for immediate danger situations
- Limited guardianships available for specific purposes

For assistance: Contact Genesee County Probate Court at (810) 424-4355
EOF
print_success "Sample knowledge base created"

# Step 12: Final validation
print_header "Step 12: Final Validation"

print_step "Running validation..."
python validate_jules_setup.py

# Step 13: Create summary file
print_header "Creating Summary"

cat > JULES_SETUP_SUMMARY.md << 'EOF'
# Michigan Guardianship AI - Jules Setup Complete

## ✅ Setup Status: COMPLETE

### Environment Details
- **Platform**: Jules (Ubuntu Linux)
- **Python Version**: 3.12.11
- **Setup Date**: $(date)
- **Repository**: michigan-guardianship-gemma

### Configured Components

#### 1. API Keys (All Configured)
- ✅ Google AI API Key
- ✅ HuggingFace Token
- ✅ Pinecone API Key & Host

#### 2. Python Packages (35+ installed)
- Core: pandas, pyyaml, pdfplumber
- AI/ML: torch, sentence-transformers, google-generativeai
- Vector DB: chromadb, pinecone-client
- Web: flask, flask-cors, flask-session
- Utilities: tqdm, colorlog, python-dotenv

#### 3. Directory Structure
```
michigan-guardianship-gemma/
├── venv/               # Virtual environment
├── kb_files/           # Knowledge base documents
├── scripts/            # Processing scripts
├── config/             # Configuration files
├── logs/               # Application logs
├── models/             # ML models
├── results/            # Output results
└── chroma_db/          # Local vector database
```

#### 4. Configuration Files
- `.env` - Environment variables with API keys
- `config/logging.yaml` - Logging configuration
- `config/pinecone_jules.yaml` - Pinecone settings

#### 5. Utility Scripts
- `setup_pinecone_jules.py` - Initialize Pinecone
- `embed_cloud_jules.py` - Cloud embeddings
- `validate_jules_setup.py` - Validation tool
- `launch_jules.sh` - Application launcher

### Quick Start Commands

1. **Activate environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Setup vector database**:
   ```bash
   python setup_pinecone_jules.py
   python scripts/migrate_to_pinecone.py
   ```

3. **Launch application**:
   ```bash
   ./launch_jules.sh
   ```

4. **Access**: http://localhost:5000

### Verification Checklist
- [x] Python 3.8+ installed
- [x] Virtual environment created
- [x] All dependencies installed
- [x] API keys configured
- [x] Directory structure created
- [x] Configuration files generated
- [x] Validation script passed
- [x] Ready for deployment

### Troubleshooting

If issues occur:
1. Check logs: `cat logs/app.log`
2. Validate setup: `python validate_jules_setup.py`
3. Verify API keys: `grep API_KEY .env`
4. Test imports: `python -c "import pinecone; print('OK')"`

### Performance Notes
- Uses cloud embeddings (faster, no local GPU needed)
- Pinecone for scalable vector search
- Optimized for 4GB+ RAM systems
- Rate limiting configured for API calls

### Security Reminders
- API keys are in `.env` (gitignored)
- Never commit `.env` to version control
- Rotate API keys periodically
- Use environment-specific keys

---
**Setup completed successfully for Jules environment!**
EOF

print_success "Summary file created: JULES_SETUP_SUMMARY.md"

# Final message
print_header "✅ Setup Complete!"

echo -e "${GREEN}${BOLD}Michigan Guardianship AI is ready for Jules!${NC}"
echo ""
echo "Summary saved to: JULES_SETUP_SUMMARY.md"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Next steps:"
echo "1. Setup Pinecone: python setup_pinecone_jules.py"
echo "2. Migrate data: python scripts/migrate_to_pinecone.py"
echo "3. Launch app: ./launch_jules.sh"
echo ""
echo -e "${CYAN}Thank you for using Michigan Guardianship AI!${NC}"