#!/bin/bash
# Michigan Guardianship AI - Complete Jules Environment Setup Script
# Compatible with Jules (Ubuntu Linux, Python 3.12.11)
# Version: 1.0.0

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.12.11"
MIN_PYTHON_VERSION="3.8"
MIN_MEMORY_MB=4096
REPO_URL="https://github.com/ASAPASMR-hue/michigan-guardianship-gemma.git"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_section() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

# Function to check Python version
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION_CHECK=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
            print_status "Python $PYTHON_VERSION_CHECK detected (minimum 3.8 required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION_CHECK is too old. Minimum 3.8 required"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to check available memory
check_memory() {
    if [ -f /proc/meminfo ]; then
        TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        TOTAL_MEM_MB=$((TOTAL_MEM_KB / 1024))
        
        if [ $TOTAL_MEM_MB -lt $MIN_MEMORY_MB ]; then
            print_warning "System has ${TOTAL_MEM_MB}MB RAM. Minimum recommended: ${MIN_MEMORY_MB}MB"
            print_info "Consider using USE_SMALL_MODEL=true in .env file"
        else
            print_status "System memory: ${TOTAL_MEM_MB}MB (sufficient)"
        fi
    else
        print_warning "Cannot determine system memory"
    fi
}

# Function to check network connectivity
check_network() {
    print_info "Checking network connectivity..."
    
    if ping -c 1 google.com &> /dev/null; then
        print_status "Network connectivity OK"
    else
        print_error "No network connectivity detected"
        return 1
    fi
    
    # Check access to required services
    if curl -s -o /dev/null -w "%{http_code}" https://api.github.com | grep -q "200"; then
        print_status "GitHub API accessible"
    else
        print_warning "GitHub API may be restricted"
    fi
    
    if curl -s -o /dev/null -w "%{http_code}" https://generativelanguage.googleapis.com | grep -q "200\|404"; then
        print_status "Google AI API endpoint accessible"
    else
        print_warning "Google AI API may be restricted"
    fi
}

# Main setup begins
print_section "Michigan Guardianship AI - Jules Environment Setup"
echo "Starting comprehensive setup for Jules environment..."
echo "Repository: $REPO_URL"
echo "Date: $(date)"
echo ""

# Step 1: System checks
print_section "Step 1: System Requirements Check"

print_info "Checking system requirements..."
check_python_version || exit 1
check_memory
check_network || print_warning "Network issues detected - some features may not work"

# Check for git
if ! command -v git &> /dev/null; then
    print_error "Git not found. Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

# Step 2: Clone or update repository
print_section "Step 2: Repository Setup"

if [ -d "michigan-guardianship-gemma" ]; then
    print_info "Repository already exists. Updating..."
    cd michigan-guardianship-gemma
    git pull origin main || print_warning "Could not update repository"
else
    print_info "Cloning repository..."
    git clone $REPO_URL
    cd michigan-guardianship-gemma
fi

# Step 3: Create virtual environment
print_section "Step 3: Python Virtual Environment"

if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
else
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Step 4: Upgrade pip and install wheel
print_section "Step 4: Python Package Manager Setup"

print_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Step 5: Install Python dependencies
print_section "Step 5: Installing Python Dependencies"

# Create comprehensive requirements file
cat > requirements_complete.txt << 'EOF'
# Core dependencies
pyyaml>=6.0
pandas>=1.3.0
pdfplumber>=0.10.0
lettucedetect
rank-bm25>=0.2.0
scikit-learn>=1.0.0
scipy>=1.7.0

# AI/ML dependencies
openai>=1.0.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
torch>=2.0.0
chromadb>=0.4.0

# Web framework
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

# Migration dependencies
pinecone-client>=3.0.0
colorlog>=6.7.0
colorama>=0.4.6

# Additional dependencies for production
numpy>=1.21.0
httpx>=0.24.0
aiofiles>=23.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
EOF

print_info "Installing all Python dependencies..."
pip install -r requirements_complete.txt 2>&1 | tee install_log.txt

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_status "All Python dependencies installed successfully"
else
    print_error "Some dependencies failed to install. Check install_log.txt"
    print_info "Attempting to install core dependencies only..."
    pip install -r requirements.txt
fi

# Step 6: Create directory structure
print_section "Step 6: Creating Directory Structure"

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
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    else
        print_info "Directory exists: $dir"
    fi
done

# Step 7: Environment configuration
print_section "Step 7: Environment Configuration"

if [ ! -f .env ]; then
    print_info "Creating .env file from template..."
    
    cat > .env << 'EOF'
# Michigan Guardianship AI Environment Variables
# Auto-generated by jules_setup.sh

# =============================================================================
# REQUIRED API KEYS - PLEASE CONFIGURE THESE
# =============================================================================

# Google AI API Key (REQUIRED)
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_AI_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_google_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# HuggingFace Token (REQUIRED for local embeddings)
# Get from: https://huggingface.co/settings/tokens
HUGGING_FACE_HUB_TOKEN=your_hf_token_here
HUGGINGFACE_TOKEN=your_hf_token_here

# =============================================================================
# OPTIONAL CONFIGURATION
# =============================================================================

# Port Configuration (5000 or 5001)
PORT=5000

# Model Configuration
# Set to 'true' for smaller models (4GB RAM) or 'false' for full models (8GB RAM)
USE_SMALL_MODEL=true

# OpenRouter API Key (OPTIONAL - for Phase 3 testing)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Pinecone API Key (OPTIONAL - for vector database migration)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

# Logging level
LOG_LEVEL=INFO

# Flask debug mode
FLASK_DEBUG=false

# Host configuration
HOST=127.0.0.1

# Model download timeout (seconds)
HF_HUB_DOWNLOAD_TIMEOUT=3600

# Embedding method (cloud or local)
EMBEDDING_METHOD=cloud
EOF
    
    print_status ".env file created"
    print_warning "IMPORTANT: Edit .env file and add your API keys!"
else
    print_info ".env file already exists"
fi

# Step 8: Download sample data if not present
print_section "Step 8: Knowledge Base Setup"

if [ ! -f "kb_files/KB (Numbered)/001_overview.txt" ]; then
    print_info "Creating sample knowledge base files..."
    
    cat > "kb_files/KB (Numbered)/001_overview.txt" << 'EOF'
Michigan Guardianship Overview
==============================

This document provides information about minor guardianship procedures
in Genesee County, Michigan.

Key Information:
- Filing fee: $175
- Court location: 900 S. Saginaw Street, Flint, MI 48502
- Hearings: Thursdays at 9:00 AM

For more information, contact the Genesee County Probate Court.
EOF
    
    print_status "Sample knowledge base files created"
else
    print_info "Knowledge base files already present"
fi

# Step 9: Configuration files
print_section "Step 9: Configuration Files"

# Create logging configuration
if [ ! -f "config/logging.yaml" ]; then
    cat > "config/logging.yaml" << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/app.log

root:
  level: INFO
  handlers: [console, file]
EOF
    print_status "Logging configuration created"
fi

# Step 10: Database setup script
print_section "Step 10: Vector Database Setup"

cat > setup_vector_db.py << 'EOF'
#!/usr/bin/env python3
"""Setup vector database for Michigan Guardianship AI"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_cloud_embeddings():
    """Setup using Google AI cloud embeddings"""
    print("Setting up cloud-based embeddings...")
    os.system("python scripts/embed_kb_cloud.py")

def setup_local_embeddings():
    """Setup using local embedding models"""
    print("Setting up local embeddings...")
    print("Downloading embedding model (this may take 5-10 minutes)...")
    os.system("python scripts/embed_kb.py")

def main():
    embedding_method = os.getenv('EMBEDDING_METHOD', 'cloud').lower()
    
    if embedding_method == 'cloud':
        setup_cloud_embeddings()
    else:
        setup_local_embeddings()
    
    print("\nVector database setup complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x setup_vector_db.py
print_status "Vector database setup script created"

# Step 11: Create validation script
print_section "Step 11: Validation Script"

cat > validate_setup.py << 'EOF'
#!/usr/bin/env python3
"""Validate Michigan Guardianship AI setup"""

import os
import sys
import importlib
from pathlib import Path

def check_import(module_name):
    """Check if a Python module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)

def check_env_var(var_name):
    """Check if an environment variable is set"""
    value = os.getenv(var_name)
    if value and value != f"your_{var_name.lower()}_here":
        return True, "Set"
    return False, "Not configured"

def main():
    print("\nMichigan Guardianship AI - Setup Validation")
    print("=" * 50)
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    
    # Check critical imports
    print("\nPython Packages:")
    packages = [
        'flask', 'chromadb', 'sentence_transformers',
        'google.generativeai', 'pandas', 'pdfplumber',
        'torch', 'sklearn', 'tqdm', 'yaml'
    ]
    
    all_ok = True
    for pkg in packages:
        ok, msg = check_import(pkg.split('.')[0])
        status = "✓" if ok else "✗"
        print(f"  [{status}] {pkg}: {msg if not ok else 'OK'}")
        if not ok:
            all_ok = False
    
    # Check environment variables
    print("\nEnvironment Variables:")
    env_vars = [
        'GOOGLE_API_KEY',
        'GEMINI_API_KEY',
        'HUGGINGFACE_TOKEN'
    ]
    
    for var in env_vars:
        ok, msg = check_env_var(var)
        status = "✓" if ok else "✗"
        print(f"  [{status}] {var}: {msg}")
    
    # Check directories
    print("\nDirectory Structure:")
    dirs = ['kb_files', 'scripts', 'config', 'chroma_db', 'models', 'logs']
    for dir_name in dirs:
        exists = Path(dir_name).exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {dir_name}")
    
    # Check files
    print("\nConfiguration Files:")
    files = ['.env', 'requirements.txt']
    for file_name in files:
        exists = Path(file_name).exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {file_name}")
    
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ Setup validation PASSED")
        return 0
    else:
        print("✗ Setup validation FAILED - please fix issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x validate_setup.py
print_status "Validation script created"

# Step 12: Create startup script
print_section "Step 12: Creating Startup Script"

cat > start_app.sh << 'EOF'
#!/bin/bash
# Start Michigan Guardianship AI

# Activate virtual environment
source venv/bin/activate

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Start the application
echo "Starting Michigan Guardianship AI..."
echo "Access the application at: http://localhost:${PORT:-5000}"
python app.py
EOF

chmod +x start_app.sh
print_status "Startup script created"

# Step 13: Run validation
print_section "Step 13: Setup Validation"

print_info "Running setup validation..."
python validate_setup.py

# Step 14: Final instructions
print_section "Setup Complete!"

echo -e "${GREEN}✓ Michigan Guardianship AI setup is complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit the .env file and add your API keys:"
echo "   - Google AI API Key (required)"
echo "   - HuggingFace Token (for local embeddings)"
echo ""
echo "2. Set up the vector database:"
echo "   python setup_vector_db.py"
echo ""
echo "3. Start the application:"
echo "   ./start_app.sh"
echo "   OR"
echo "   python app.py"
echo ""
echo "4. Access the application at:"
echo "   http://localhost:5000"
echo ""
echo "For Jules environment:"
echo "- This script has been optimized for Ubuntu Linux"
echo "- Python 3.12.11 compatible"
echo "- All dependencies are installed in the virtual environment"
echo ""
echo "Documentation:"
echo "- README.md for general information"
echo "- MIGRATION_ACTION_PLAN.md for migration details"
echo "- Run 'python validate_setup.py' to verify setup"
echo ""
print_info "Setup log saved to: install_log.txt"
print_info "Virtual environment: venv/"

# Create a Jules-specific configuration file
cat > jules_config.json << 'EOF'
{
  "environment": "jules",
  "platform": "ubuntu-linux",
  "python_version": "3.12.11",
  "setup_complete": true,
  "timestamp": "$(date -Iseconds)",
  "features": {
    "vector_db": "chromadb",
    "embedding_method": "cloud",
    "llm": "gemini",
    "web_framework": "flask"
  }
}
EOF

print_status "Jules configuration saved to jules_config.json"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup completed successfully at $(date)"
echo "═══════════════════════════════════════════════════════════"