#!/usr/bin/env python3
"""
Full Phase 3 Testing Script
Tests all 11 models against all 200 questions
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("==============================================")
    print("Michigan Guardianship AI - Full Phase 3 Testing")
    print("==============================================")
    
    # Set environment variables
    print("\nSetting up environment variables...")
    os.environ['OPENROUTER_API_KEY'] = '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'
    os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['GEMINI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['USE_SMALL_MODEL'] = 'true'  # Use same model as ChromaDB
    
    print("✅ Environment variables set")
    
    # All 11 models to test
    models = [
        # Google AI Studio models (6)
        "google/gemini-2.5-flash-lite",
        "google/gemma-3n-e4b-it",
        "google/gemma-3-4b-it",
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-2.0-flash-001",
        "google/gemini-flash-1.5-8b",
        
        # OpenRouter models (5)
        "mistralai/mistral-small-24b-instruct-2501",
        "mistralai/mistral-nemo:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "microsoft/phi-4",
        "meta-llama/llama-3.1-8b-instruct"
    ]
    
    print(f"\n📊 Testing {len(models)} models against 200 questions")
    print("This will take approximately 30-60 minutes\n")
    
    # Confirm with user
    response = input("Do you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return 1
    
    # Run full evaluation
    cmd = [
        sys.executable,
        "scripts/run_full_evaluation.py",
        "--output_dir", "results"
    ]
    
    print("\n🚀 Starting full Phase 3 evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Phase 3 full testing completed successfully!")
        print("Check the results directory for detailed outputs")
    else:
        print(f"\n❌ Phase 3 testing failed with return code: {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())