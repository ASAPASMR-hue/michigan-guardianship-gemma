#!/bin/bash

# Set environment variables
export OPENROUTER_API_KEY="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"
export USE_SMALL_MODEL="true"

echo "Running Microsoft Phi-4 and Meta Llama 3.1 8B evaluation..."
echo "Models: microsoft/phi-4, meta-llama/llama-3.1-8b-instruct"
echo "Questions: 200"
echo ""

python scripts/run_full_evaluation.py \
  --models "microsoft/phi-4" "meta-llama/llama-3.1-8b-instruct" \
  --output_dir results/phi4_llama_test