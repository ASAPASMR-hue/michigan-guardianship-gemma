#!/usr/bin/env python3
"""
Extended Phase 3 test to verify multiple models
Tests 5 models with 10 questions each
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("==============================================")
    print("Phase 3 Extended Testing - 5 Models")
    print("==============================================")
    
    # Set environment variables
    print("\nSetting up environment variables...")
    os.environ['OPENROUTER_API_KEY'] = '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'
    os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['GEMINI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['USE_SMALL_MODEL'] = 'true'  # Use same model as ChromaDB
    
    print("✅ Environment variables set")
    
    # Test 5 diverse models
    models = [
        "google/gemini-2.0-flash-001",      # Google's latest
        "google/gemini-2.5-flash-lite",     # Google's fastest
        "mistralai/mistral-nemo:free",      # Free Mistral
        "meta-llama/llama-3.1-8b-instruct", # Llama 3.1
        "microsoft/phi-4"                    # Microsoft's model
    ]
    
    # Run with extended parameters
    cmd = [
        sys.executable,
        "scripts/run_full_evaluation.py",
        "--models"] + models + [
        "--questions", "10",
        "--output_dir", "results"
    ]
    
    print(f"\n📊 Testing {len(models)} models with 10 questions each")
    print("Estimated time: 5-10 minutes")
    print(f"\nCommand: {' '.join(cmd[:4])}...")
    
    # Execute
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Extended Phase 3 test completed successfully!")
        
        # Find the run ID
        results_dir = Path("results")
        run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if run_dirs:
            latest_run = run_dirs[-1].name
            print(f"\nResults saved to: results/{latest_run}/")
            print(f"\nTo analyze results, run:")
            print(f"  python scripts/run_phase3b_analysis.py {latest_run}")
    else:
        print(f"\n❌ Extended test failed with return code: {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())