#!/bin/bash
# Setup script for Phase 3 testing

echo "Michigan Guardianship AI - Phase 3 Setup"
echo "========================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - OPENROUTER_API_KEY"
    echo "   - GOOGLE_AI_API_KEY" 
    echo "   - HUGGINGFACE_TOKEN"
else
    echo "✅ .env file already exists"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Test setup
echo ""
echo "Testing setup..."
python scripts/test_phase3_setup.py

echo ""
echo "Setup complete! To run the full evaluation:"
echo "  python scripts/run_full_evaluation.py"
echo ""
echo "To analyze results after completion:"
echo "  python scripts/analyze_results.py --run_id run_YYYYMMDD_HHMM"