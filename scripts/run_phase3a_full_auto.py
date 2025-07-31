#!/usr/bin/env python3
"""
Full Phase 3a Execution Script - Automated
Tests ALL 11 models against ALL 200 questions
No limits, no early stopping, no user prompts
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("="*60)
    print("PHASE 3A FULL EXECUTION - AUTOMATED")
    print("="*60)
    
    # Set environment variables
    print("\nSetting environment variables...")
    os.environ['OPENROUTER_API_KEY'] = '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'
    os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['GEMINI_API_KEY'] = 'AIzaSyATIx4py5dXVcXYke9aGd3Af11zR26hvGI'
    os.environ['USE_SMALL_MODEL'] = 'true'  # Use same model as ChromaDB
    
    print("✅ Environment variables set")
    
    # Confirm configuration
    print("\n📋 EXECUTING WITH CONFIGURATION:")
    print("- Models to test: 11")
    print("  - Google AI Studio: 6 models")
    print("  - OpenRouter: 5 models")
    print("- Questions to test: 200 (full dataset)")
    print("- Question file: data/Synthetic Test Questions (2).xlsx")
    print("- No limits or early stopping will be applied")
    print("- Estimated time: 2-3 hours")
    
    print("\n⚠️  PROCESSING: 2,200 API calls (11 models × 200 questions)")
    
    start_time = datetime.now()
    print(f"\n🚀 Starting FULL Phase 3a execution at {start_time}")
    
    # Run full evaluation without any limits
    cmd = [
        sys.executable,
        "scripts/run_full_evaluation.py",
        "--output_dir", "results"
        # Note: NOT passing --questions or --models to ensure ALL are processed
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("\nProcessing:")
    print("- ALL 11 models from config/model_configs_phase3.yaml")
    print("- ALL 200 questions from data/Synthetic Test Questions (2).xlsx")
    print("\nProgress will be shown every 10 questions per model...")
    print("-"*60)
    
    # Execute with real-time output
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ PHASE 3A FULL EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        print(f"Total duration: {duration}")
        print(f"Average time per model: {duration / 11}")
        print("\nResults saved to: results/")
        print("\nNext steps:")
        print("1. Run Phase 3b analysis: python scripts/run_phase3b_analysis.py")
        print("2. Generate visualizations: python scripts/create_phase3_visualization.py")
        print("3. Review comprehensive report in results/*/evaluation_summary.md")
    else:
        print(f"\n❌ Phase 3a execution failed with return code: {result.returncode}")
        print(f"Duration before failure: {duration}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())