#!/bin/bash

# Set environment variables
export GOOGLE_AI_API_KEY="AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI"
export GEMINI_API_KEY="AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI"
export USE_SMALL_MODEL="true"

echo "Running Gemma models retest with rate limiting..."
echo "Models: google/gemma-3n-e4b-it, google/gemma-3-4b-it"
echo "Questions: 200"
echo "Rate limit: 15,000 tokens/minute (handled automatically)"
echo ""

python scripts/run_full_evaluation.py \
  --models "google/gemma-3n-e4b-it" "google/gemma-3-4b-it" \
  --output_dir results/gemma_retest