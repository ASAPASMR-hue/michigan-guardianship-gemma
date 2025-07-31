#!/bin/bash

# Michigan Guardianship AI - Phase 3 Testing Runner

echo "===================="
echo "Phase 3 Testing Setup"
echo "===================="

# Set environment variables
export OPENROUTER_API_KEY="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"
export GOOGLE_AI_API_KEY="AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI"
export GEMINI_API_KEY="AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI"

# Also set HUGGINGFACE_TOKEN if available
# export HUGGINGFACE_TOKEN=""

echo "✅ Environment variables set"
echo ""

# Create necessary directories
mkdir -p results
mkdir -p logs

echo "📊 Running Phase 3 Test Setup..."
cd scripts

# First test connectivity
echo ""
echo "Testing API connectivity..."
python -c "
import os
from llm_handler import LLMHandler

handler = LLMHandler()

# Test OpenRouter
print('Testing OpenRouter...')
try:
    result = handler.call_llm(
        model_id='mistralai/mistral-nemo:free',
        messages=[{'role': 'user', 'content': 'Hello'}],
        model_api='openrouter',
        max_tokens=10
    )
    if result['response']:
        print('✅ OpenRouter connected successfully')
    else:
        print('❌ OpenRouter connection failed:', result.get('error'))
except Exception as e:
    print('❌ OpenRouter error:', str(e))

# Test Google AI
print('\nTesting Google AI...')
try:
    result = handler.call_llm(
        model_id='google/gemini-2.0-flash-001',
        messages=[{'role': 'user', 'content': 'Hello'}],
        model_api='google',
        max_tokens=10
    )
    if result['response']:
        print('✅ Google AI connected successfully')
    else:
        print('❌ Google AI connection failed:', result.get('error'))
except Exception as e:
    print('❌ Google AI error:', str(e))
"

echo ""
echo "===================="
echo "Starting Phase 3 Evaluation"
echo "===================="

# Run the full evaluation with a subset of models first
echo ""
echo "Running evaluation with test models..."

python run_full_evaluation.py \
  --models "mistralai/mistral-nemo:free" "google/gemini-2.0-flash-001" \
  --questions 5 \
  --output_dir ../results

echo ""
echo "===================="
echo "Phase 3 Testing Complete"
echo "===================="
echo ""
echo "Check results in: results/"
echo "Check logs in: logs/"