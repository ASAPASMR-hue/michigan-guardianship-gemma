#!/usr/bin/env python3
"""
Minimal Phase 3 test to verify everything is working
Tests 2 models with 3 questions each
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("Setting up environment variables...")
    
    # Set environment variables
    os.environ['OPENROUTER_API_KEY'] = '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'
    os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['GEMINI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['USE_SMALL_MODEL'] = 'true'  # Use same model as ChromaDB
    
    print("✅ Environment variables set")
    
    # Run with minimal parameters
    cmd = [
        sys.executable,
        "scripts/run_full_evaluation.py",
        "--models", "mistralai/mistral-nemo:free", "google/gemini-2.0-flash-001",
        "--questions", "3",
        "--output_dir", "results"
    ]
    
    print("\nRunning minimal Phase 3 test...")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Phase 3 minimal test completed successfully!")
    else:
        print(f"\n❌ Phase 3 test failed with return code: {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())