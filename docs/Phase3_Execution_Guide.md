# Phase 3 Testing Execution Guide

## Overview

Phase 3 testing evaluates 11 different LLMs (6 via Google AI Studio, 5 via OpenRouter) against 200 synthetic Michigan guardianship questions. The testing is split into two phases:

- **Phase 3a**: Data generation (API calls to LLMs)
- **Phase 3b**: Evaluation and analysis

## Prerequisites

1. **API Keys Set Up**:
   - OpenRouter API key
   - Google AI Studio API key
   - (Optional) HuggingFace token

2. **ChromaDB Populated**:
   - Run `make phase1` or `python scripts/embed_kb.py` first
   - Ensure USE_SMALL_MODEL=true if using development setup

3. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Test with 2 Models, 3 Questions (1 minute)
```bash
python scripts/test_phase3_minimal.py
```

### Full Testing (30-60 minutes)
```bash
# Phase 3a: Generate all responses
python scripts/run_phase3_full.py

# Phase 3b: Analyze results
python scripts/run_phase3b_analysis.py
```

## Detailed Instructions

### Phase 3a: Data Generation

1. **Set Environment Variables**:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   export GOOGLE_AI_API_KEY="AIzaSy..."
   export USE_SMALL_MODEL="true"  # If using dev setup
   ```

2. **Run Full Evaluation**:
   ```bash
   python scripts/run_full_evaluation.py
   ```
   
   Or with specific models:
   ```bash
   python scripts/run_full_evaluation.py \
     --models "mistralai/mistral-nemo:free" "google/gemini-2.0-flash-001" \
     --questions 10
   ```

3. **Monitor Progress**:
   ```bash
   # Watch logs
   tail -f logs/phase3_testing_*.log
   
   # Check results
   ls -la results/run_*/
   ```

### Phase 3b: Analysis

1. **Run Basic Analysis**:
   ```bash
   # Analyze latest run
   python scripts/analyze_results.py
   
   # Analyze specific run
   python scripts/analyze_results.py run_20250728_2224
   ```

2. **Advanced Agent Analysis**:
   ```bash
   # Comprehensive analysis
   cd scripts/agents
   make full-analysis RUN_ID=run_20250728_2224
   
   # Generate dashboard
   make dashboard RUN_ID=run_20250728_2224
   ```

## Model List

### Google AI Studio Models (6)
1. `google/gemini-2.5-flash-lite` - Fastest, lightest
2. `google/gemma-3n-e4b-it` - Efficient small model
3. `google/gemma-3-4b-it` - Balanced performance
4. `google/gemini-2.0-flash-lite-001` - Latest lite version
5. `google/gemini-2.0-flash-001` - Latest standard version
6. `google/gemini-flash-1.5-8b` - Larger capacity

### OpenRouter Models (5)
1. `mistralai/mistral-small-24b-instruct-2501` - High quality
2. `mistralai/mistral-nemo:free` - Free tier
3. `meta-llama/llama-3.3-70b-instruct:free` - Large free model
4. `microsoft/phi-4` - Microsoft's efficient model
5. `meta-llama/llama-3.1-8b-instruct` - Balanced Llama model

## Output Structure

```
results/
└── run_20250728_2224/
    ├── metadata.json                    # Run configuration
    ├── run_summary.json                 # Execution summary
    ├── google_gemini-2-0-flash-001_results.json
    ├── mistralai_mistral-nemo_free_results.json
    └── analysis/
        ├── evaluation_summary.json      # 7-dimension scores
        ├── evaluation_report.md         # Human-readable report
        └── model_comparisons.json       # Cross-model analysis
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch Error**:
   ```
   Collection expecting embedding with dimension of 384, got 1024
   ```
   **Solution**: Set `USE_SMALL_MODEL=true`

2. **Google Model Name Error**:
   ```
   400 * GenerateContentRequest.model: unexpected model name format
   ```
   **Solution**: Update to latest code (already fixed)

3. **Timeout Errors**:
   - Models have 180-second timeout
   - Complex questions may timeout on slower models
   - Timeouts are recorded but don't stop execution

4. **API Rate Limits**:
   - OpenRouter: Check your rate limits
   - Google AI: 60 QPM default limit
   - Add delays if hitting limits

### Debug Commands

```bash
# Check environment
python -c "import os; print('APIs configured:', bool(os.getenv('OPENROUTER_API_KEY')))"

# Test single model
python scripts/llm_handler.py

# Verify ChromaDB
python scripts/retrieval_setup.py test

# Check logs
grep ERROR logs/phase3_testing_*.log
```

## Cost Estimates

Based on pricing as of July 2025:

- **Free Models**: $0 (mistral-nemo:free, llama-3.3-70b:free)
- **Google Models**: ~$0.10-0.50 for full 200 question test
- **Premium Models**: ~$1-5 for full test
- **Total Estimated Cost**: <$10 for complete Phase 3 testing

## Next Steps

After Phase 3 testing:

1. **Review Results**:
   ```bash
   cd scripts/agents
   make reports
   ```

2. **Optimize Poorly Performing Models**:
   ```bash
   make optimization RUN_ID=run_xxx
   ```

3. **Deploy Best Model**:
   - Check `results/*/analysis/model_rankings.json`
   - Update `config/model_configs.yaml` with best performer
   - Run production tests

## Advanced Usage

### Custom Test Sets

Create custom question CSV:
```csv
id,question,category,subcategory,complexity_tier
CUSTOM001,"Custom question here",general,info,simple
```

Run with custom questions:
```bash
python scripts/run_full_evaluation.py --questions-file custom_test.csv
```

### Parallel Execution

For faster testing with multiple API keys:
```bash
# Split models into groups
python scripts/run_full_evaluation.py --models "google/*" &
python scripts/run_full_evaluation.py --models "mistralai/*" "meta-llama/*" &
```

### Continuous Monitoring

Set up automated testing:
```bash
cd scripts/agents
make schedule-start

# Add daily test job
python agent_scheduler.py add \
  --name "daily_phase3" \
  --schedule "0 2 * * *" \
  --command "cd /path/to/project && python scripts/test_phase3_minimal.py"
```

## Support

For issues or questions:
1. Check `logs/phase3_testing_*.log`
2. Review error summaries in `results/*/failed_models.json`
3. Use agent diagnostics: `make health-check`