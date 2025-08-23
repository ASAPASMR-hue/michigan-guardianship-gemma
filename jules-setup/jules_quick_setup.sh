#!/bin/bash
# Michigan Guardianship AI - Quick Setup with Pre-configured API Keys
# This script uses your provided API keys automatically

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Michigan Guardianship AI - Quick Jules Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Clone repository if not exists
if [ ! -d "michigan-guardianship-gemma" ]; then
    echo -e "${YELLOW}[1/5]${NC} Cloning repository..."
    git clone https://github.com/ASAPASMR-hue/michigan-guardianship-gemma.git
fi

cd michigan-guardianship-gemma

# Copy pre-configured .env file
echo -e "${YELLOW}[2/5]${NC} Setting up environment with your API keys..."
cp ../.env.configured .env

# Create virtual environment
echo -e "${YELLOW}[3/5]${NC} Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}[4/5]${NC} Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
if [ -f requirements-migration.txt ]; then
    pip install -r requirements-migration.txt
fi

# Create necessary directories
echo -e "${YELLOW}[5/5]${NC} Creating directory structure..."
mkdir -p chroma_db models logs results

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Setup Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Your API keys have been configured:"
echo "  ✓ Google AI API Key"
echo "  ✓ HuggingFace Token"
echo "  ✓ Pinecone API Key & Host"
echo ""
echo "Next steps:"
echo "1. Set up vector database (choose one):"
echo "   a) For Pinecone: python scripts/migrate_to_pinecone.py"
echo "   b) For ChromaDB with cloud embeddings: python scripts/embed_kb_cloud.py"
echo "   c) For ChromaDB with local embeddings: python scripts/embed_kb.py"
echo ""
echo "2. Start the application:"
echo "   python app.py"
echo ""
echo "3. Access at: http://localhost:5000"
echo ""
echo -e "${BLUE}For Jules: This setup is optimized for Ubuntu Linux with Python 3.12.11${NC}"