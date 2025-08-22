#!/bin/bash

# Phase 1 Pinecone Migration Test Script
# This script verifies if everything is ready for migration

echo "============================================"
echo "🔍 Phase 1 Setup Verification"
echo "============================================"

# Check Python
echo -n "✓ Python: "
python3 --version || echo "❌ Not found"

# Check current directory
echo -n "✓ Current Directory: "
pwd

# Check .env file
echo -n "✓ Environment File: "
if [ -f .env ]; then
    echo "Found"
    echo -n "  - Pinecone API Key: "
    grep -q "PINECONE_API_KEY=" .env && echo "✅ Configured" || echo "❌ Missing"
    echo -n "  - Google API Key: "
    grep -q "GOOGLE_API_KEY=" .env && echo "✅ Configured" || echo "❌ Missing"
else
    echo "❌ Missing"
fi

# Check required directories
echo ""
echo "📁 Directory Structure:"
echo -n "  - kb_files: "
[ -d kb_files ] && echo "✅ Found $(find kb_files -type f | wc -l) files" || echo "❌ Missing"
echo -n "  - scripts: "
[ -d scripts ] && echo "✅ Found" || echo "❌ Missing"
echo -n "  - config: "
[ -d config ] && echo "✅ Found" || echo "❌ Missing"

# Check key files
echo ""
echo "📄 Required Files:"
echo -n "  - scripts/setup_pinecone.py: "
[ -f scripts/setup_pinecone.py ] && echo "✅ Found" || echo "❌ Missing"
echo -n "  - scripts/migrate_to_pinecone.py: "
[ -f scripts/migrate_to_pinecone.py ] && echo "✅ Found" || echo "❌ Missing"
echo -n "  - scripts/embed_kb_cloud.py: "
[ -f scripts/embed_kb_cloud.py ] && echo "✅ Found" || echo "❌ Missing"
echo -n "  - config/pinecone.yaml: "
[ -f config/pinecone.yaml ] && echo "✅ Found" || echo "❌ Missing"

# Check Python dependencies
echo ""
echo "📦 Python Dependencies:"
python3 -c "import pinecone" 2>/dev/null && echo "  - pinecone: ✅ Installed" || echo "  - pinecone: ❌ Not installed"
python3 -c "import google.generativeai" 2>/dev/null && echo "  - google.generativeai: ✅ Installed" || echo "  - google.generativeai: ❌ Not installed"
python3 -c "import pdfplumber" 2>/dev/null && echo "  - pdfplumber: ✅ Installed" || echo "  - pdfplumber: ❌ Not installed"
python3 -c "import chromadb" 2>/dev/null && echo "  - chromadb: ✅ Installed" || echo "  - chromadb: ❌ Not installed"
python3 -c "import colorlog" 2>/dev/null && echo "  - colorlog: ✅ Installed" || echo "  - colorlog: ❌ Not installed"
python3 -c "import tqdm" 2>/dev/null && echo "  - tqdm: ✅ Installed" || echo "  - tqdm: ❌ Not installed"

echo ""
echo "============================================"
echo "📋 Test Results Summary"
echo "============================================"

# Quick test of the migration script syntax
echo -n "Testing migration script syntax: "
python3 -m py_compile scripts/migrate_to_pinecone.py 2>/dev/null && echo "✅ Valid Python" || echo "❌ Syntax errors"

echo ""
echo "🎯 Next Steps:"
echo "1. Install missing dependencies (if any):"
echo "   pip install pinecone-client google-generativeai pdfplumber chromadb colorlog tqdm"
echo ""
echo "2. Run Pinecone setup:"
echo "   python3 scripts/setup_pinecone.py"
echo ""
echo "3. Test migration with dry run:"
echo "   python3 scripts/migrate_to_pinecone.py --dry-run --verbose"
echo ""
echo "4. Run full migration:"
echo "   python3 scripts/migrate_to_pinecone.py --test --verbose"
