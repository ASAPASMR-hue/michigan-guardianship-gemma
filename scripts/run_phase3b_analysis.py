#!/usr/bin/env python3
"""
Phase 3b: Analyze Phase 3a Results
Evaluates raw test data using the 7-dimension rubric
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def find_latest_run():
    """Find the most recent run directory"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    
    # Sort by modification time
    latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
    return latest.name

def main():
    print("==============================================")
    print("Michigan Guardianship AI - Phase 3b Analysis")
    print("==============================================")
    
    # Find run to analyze
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = find_latest_run()
        if not run_id:
            print("❌ No runs found. Please run Phase 3a first.")
            return 1
    
    print(f"\n📊 Analyzing run: {run_id}")
    
    # Check if results exist
    results_path = Path(f"results/{run_id}")
    if not results_path.exists():
        print(f"❌ Run directory not found: {results_path}")
        return 1
    
    # Run analysis
    cmd = [
        sys.executable,
        "scripts/analyze_results.py",
        "--run_id", run_id
    ]
    
    print("\n🔍 Running 7-dimension evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Analysis completed successfully!")
        print(f"Check results/{run_id}/analysis/ for detailed reports")
        
        # Also run advanced agents
        print("\n🤖 Running intelligent agent analysis...")
        
        # Test results analyzer
        agent_cmd = [
            sys.executable,
            "scripts/agents/test_results_analyzer.py",
            "--run_ids", run_id,
            "--output", f"results/{run_id}/analysis/agent_analysis.md"
        ]
        subprocess.run(agent_cmd, capture_output=False, text=True)
        
        # Dashboard generator
        dashboard_cmd = [
            sys.executable,
            "scripts/agents/dashboard_generator.py",
            "--run_ids", run_id
        ]
        subprocess.run(dashboard_cmd, capture_output=False, text=True)
        
        print("\n✅ All analyses complete!")
        print(f"\nResults available at:")
        print(f"  - Summary: results/{run_id}/analysis/evaluation_summary.json")
        print(f"  - Report: results/{run_id}/analysis/evaluation_report.md")
        print(f"  - Dashboard: results/dashboards/{run_id}_dashboard.html")
        
    else:
        print(f"\n❌ Analysis failed with return code: {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())