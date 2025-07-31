#!/usr/bin/env python3
"""
Re-run Gemma models with proper rate limiting
Gemma models have 15,000 input tokens per minute limit
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def count_tokens_approx(text):
    """Approximate token count (1 token ≈ 4 characters)"""
    return len(text) // 4

def main():
    print("="*60)
    print("GEMMA MODELS RE-RUN WITH RATE LIMITING")
    print("="*60)
    
    # Set environment variables
    # API keys should be set in environment or .env file
    if not os.getenv('GOOGLE_AI_API_KEY'):
        print("ERROR: GOOGLE_AI_API_KEY not set in environment")
        sys.exit(1)
    os.environ['USE_SMALL_MODEL'] = 'true'
    
    # Load existing results to find failures
    run_id = "run_20250728_2255"
    results_dir = Path(f"results/{run_id}")
    
    # Process each Gemma model
    gemma_models = [
        "google/gemma-3n-e4b-it",
        "google/gemma-3-4b-it"
    ]
    
    for model in gemma_models:
        print(f"\n📊 Re-running {model}")
        
        # Load existing results
        result_file = results_dir / f"{model.replace('/', '_')}_results.json"
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Find failed questions
        failed_questions = [r for r in results if r.get('error') and '429' in str(r.get('error', ''))]
        print(f"Found {len(failed_questions)} rate-limited questions to retry")
        
        if not failed_questions:
            continue
            
        # Calculate batch size based on token limit
        # Assume ~1000 tokens per question + context
        # 15,000 tokens/min = 15 questions/min max
        # Add safety margin: 10 questions/min
        batch_size = 10
        delay_between_batches = 60  # seconds
        
        print(f"Processing in batches of {batch_size} with {delay_between_batches}s delay")
        
        # Create temporary question file for batch processing
        temp_questions = []
        for fail in failed_questions:
            temp_questions.append({
                'id': fail['question_id'],
                'question': fail['question_text'],
                'category': fail.get('category', ''),
                'subcategory': fail.get('subcategory', ''),
                'complexity_tier': fail.get('complexity_tier', 'standard')
            })
        
        # Save to temporary CSV
        import pandas as pd
        temp_df = pd.DataFrame(temp_questions)
        temp_file = f"temp_gemma_questions_{model.replace('/', '_')}.csv"
        temp_df.to_csv(temp_file, index=False)
        
        # Process in batches
        total_batches = (len(failed_questions) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(failed_questions))
            
            print(f"\nBatch {batch_num + 1}/{total_batches}: Questions {start_idx + 1}-{end_idx}")
            
            # Run evaluation for this batch
            cmd = [
                sys.executable,
                "scripts/run_full_evaluation.py",
                "--models", model,
                "--questions", str(batch_size),
                "--output_dir", f"results/gemma_retry_{datetime.now().strftime('%Y%m%d_%H%M')}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Batch {batch_num + 1} completed")
            else:
                print(f"❌ Batch {batch_num + 1} failed: {result.stderr}")
            
            # Wait before next batch (except for last batch)
            if batch_num < total_batches - 1:
                print(f"⏳ Waiting {delay_between_batches}s to respect rate limits...")
                time.sleep(delay_between_batches)
        
        # Clean up temp file
        os.remove(temp_file)
    
    print("\n✅ Gemma re-run complete!")
    print("Check results/gemma_retry_* directories for updated results")

if __name__ == "__main__":
    main()