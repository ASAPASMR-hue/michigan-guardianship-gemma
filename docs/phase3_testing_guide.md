# Phase 3 Testing Guide

## Overview

Phase 3 testing evaluates 11 different LLMs (6 via Google AI Studio, 5 via OpenRouter) against 200 synthetic Michigan guardianship questions. The testing is split into two phases:

- **Phase 3a**: Data generation - Runs all models and saves raw responses
- **Phase 3b**: Analysis - Evaluates responses and generates comprehensive reports

## Quick Start

1. **Setup Environment**
   ```bash
   ./setup_phase3.sh
   ```

2. **Configure API Keys**
   Edit `.env` and add your keys:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key
   GOOGLE_AI_API_KEY=AIzaSyYour-actual-key
   HUGGINGFACE_TOKEN=hf_your-actual-token
   ```

3. **Test Setup**
   ```bash
   python scripts/test_phase3_setup.py
   ```

4. **Run Full Evaluation**
   ```bash
   python scripts/run_full_evaluation.py
   ```

5. **Analyze Results**
   ```bash
   python scripts/analyze_results.py --run_id run_20250128_1430
   ```

## Models Tested

### Google AI Studio (6 models)
- google/gemini-2.5-flash-lite
- google/gemma-3n-e4b-it
- google/gemma-3-4b-it
- google/gemini-2.0-flash-lite-001
- google/gemini-2.0-flash-001
- google/gemini-flash-1.5-8b

### OpenRouter (5 models)
- mistralai/mistral-small-24b-instruct-2501
- mistralai/mistral-nemo:free
- meta-llama/llama-3.3-70b-instruct:free
- microsoft/phi-4
- meta-llama/llama-3.1-8b-instruct

## Key Features

### 180-Second Timeout
- Prevents any single model from blocking the entire test run
- Failed requests return blank responses
- Timeouts are tracked in the results

### Cost Tracking
- OpenRouter: Extracted from API headers
- Google AI: Calculated based on token usage
- Free models show $0.00 cost

### Git Tracking
- Each run captures the git commit hash
- Tracks uncommitted changes
- Ensures reproducibility

### Graceful Failure Handling
- Models that fail connectivity tests are skipped
- Partial results are saved even if some questions fail
- Failed models are logged in `failed_models.json`

## Output Structure

```
results/
├── run_20250128_1430/
│   ├── metadata.json                    # Run configuration
│   ├── run_summary.json                 # High-level statistics
│   ├── failed_models.json               # Any models that failed
│   ├── google_gemini-2-5-flash-lite_results.json
│   ├── meta-llama_llama-3-3-70b-instruct_free_results.json
│   ├── ... (other model results)
│   ├── evaluation_summary.md            # Human-readable report
│   └── evaluation_metrics.json          # Detailed scoring data
```

## Evaluation Rubric

Responses are scored on 7 dimensions (10 points total):

1. **Procedural Accuracy** (2.5 pts)
   - Correct forms (PC 651 vs PC 650)
   - Service deadlines (7/14/5 days)
   - Filing fee ($175)
   - Thursday hearings
   - Courthouse address

2. **Substantive Legal Accuracy** (2.0 pts)
   - MCL citations
   - Legal concepts
   - ICWA/MIFPA requirements

3. **Actionability** (2.0 pts)
   - Specific forms to file
   - Where to go
   - What to bring
   - Timeline/next steps

4. **Mode Effectiveness** (1.5 pts)
   - Balance of legal facts and empathy
   - Appropriate tone for complexity

5. **Strategic Caution** (0.5 pts)
   - Warnings about objections
   - Timeline expectations

6. **Citation Quality** (0.5 pts)
   - Inline citations
   - Proper formatting

7. **Harm Prevention** (0.5 pts)
   - No dangerous advice
   - No jurisdiction errors

## Interpreting Results

The evaluation report includes:

- **Model Rankings**: By total score and component scores
- **Success Rates**: Percentage of questions answered without errors
- **Performance by Complexity**: How models handle simple vs complex questions
- **Cost-Benefit Analysis**: Quality per dollar spent
- **Error Analysis**: Timeout and failure rates

### What Makes a Good Model?

- **Score > 7.0**: Excellent performance
- **Score 5.0-7.0**: Good performance with some gaps
- **Score < 5.0**: Significant issues

Consider also:
- **Reliability**: Low error/timeout rates
- **Speed**: Average latency < 5 seconds
- **Cost**: Free models with scores > 6.0 offer great value

## Troubleshooting

### "API key not configured"
- Check your `.env` file has the correct keys
- Ensure no quotes around the keys
- Source: `source .env` or restart terminal

### "Test questions not found"
- Place questions at `data/Synthetic Test Questions.xlsx`
- Or update path in `run_full_evaluation.py`

### "ChromaDB not found"
- Run document embedding first: `python scripts/embed_kb.py`

### Model timeouts
- 180-second timeout is intentional
- Check if model requires special parameters
- Some models may be overloaded

## Advanced Usage

### Test Specific Models
Edit `config/model_configs_phase3.yaml` to comment out models you don't want to test.

### Adjust Timeout
Change timeout in `config/model_configs_phase3.yaml`:
```yaml
testing:
  timeout: 300  # 5 minutes instead of 3
```

### Compare Multiple Runs
```bash
python scripts/agents/test_results_analyzer.py \
  --compare run_20250128_1430 run_20250201_0900
```

## Next Steps

After running Phase 3:

1. Review `evaluation_summary.md` for model rankings
2. Check cost analysis for budget planning
3. Examine complexity breakdown for use case fit
4. Consider implementing the top 2-3 models in production

For production use, consider:
- Using free models for simple questions
- Premium models for complex/crisis situations
- Implementing fallback chains for reliability