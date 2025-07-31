#!/usr/bin/env python3
"""
Workflow Optimizer Agent
Streamlines testing workflows and provides faster feedback loops
"""

import os
import sys
import json
import yaml
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class WorkflowOptimizer:
    """Optimizes development and testing workflows"""
    
    def __init__(self):
        """Initialize workflow optimizer"""
        self.project_root = Path(__file__).parent.parent.parent
        self.golden_questions = self._load_golden_questions()
        self.file_checksums = {}
        self._calculate_checksums()
        
    def _load_golden_questions(self) -> List[Dict[str, Any]]:
        """Load a curated set of golden test questions"""
        # These are the most important questions that cover key functionality
        golden_ids = [
            "SYNTH001",  # ICWA notice address (procedural accuracy)
            "SYNTH028",  # Filing fee (critical procedural)
            "SYNTH105",  # Standing to petition (complex legal)
            "SYNTH107",  # ICWA private agreement (crisis/harm prevention)
            "SYNTH137",  # Guardian denying parenting time (strategic caution)
        ]
        
        # Try to load from CSV
        csv_paths = [
            self.project_root / "data/Synthetic Test Questions.xlsx",
            Path("/Users/claytoncanady/Library/michigan-guardianship-ai/Synthetic Test Questions - Sheet1.csv")
        ]
        
        golden_questions = []
        for path in csv_paths:
            if path.exists():
                try:
                    if path.suffix == '.xlsx':
                        import pandas as pd
                        df = pd.read_excel(path)
                    else:
                        import pandas as pd
                        df = pd.read_csv(path)
                    
                    # Filter to golden questions
                    golden_df = df[df['id'].isin(golden_ids)]
                    golden_questions = golden_df.to_dict('records')
                    break
                except Exception as e:
                    log_step(f"Error loading golden questions: {e}", level="error")
        
        if not golden_questions:
            # Fallback: create minimal test set
            golden_questions = [
                {
                    "id": "GOLDEN_1",
                    "question": "What is the filing fee for guardianship in Genesee County?",
                    "complexity_tier": "simple"
                },
                {
                    "id": "GOLDEN_2", 
                    "question": "My child is Native American. What additional steps are required?",
                    "complexity_tier": "complex"
                }
            ]
        
        log_step(f"Loaded {len(golden_questions)} golden test questions")
        return golden_questions
    
    def _calculate_checksums(self):
        """Calculate checksums for key files"""
        key_files = [
            "scripts/llm_handler.py",
            "scripts/adaptive_retrieval.py",
            "scripts/retrieval_setup.py",
            "config/model_configs_phase3.yaml",
            "config/retrieval_pipeline.yaml",
            "kb_files/Instructive/Master Prompt Template.txt"
        ]
        
        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                with open(full_path, 'rb') as f:
                    self.file_checksums[file_path] = hashlib.md5(f.read()).hexdigest()
    
    def detect_changed_files(self) -> Set[str]:
        """Detect which key files have changed since last check"""
        changed_files = set()
        
        # Load previous checksums if available
        checksum_file = self.project_root / ".workflow_checksums.json"
        previous_checksums = {}
        
        if checksum_file.exists():
            with open(checksum_file, 'r') as f:
                previous_checksums = json.load(f)
        
        # Compare checksums
        for file_path, current_hash in self.file_checksums.items():
            if file_path not in previous_checksums or previous_checksums[file_path] != current_hash:
                changed_files.add(file_path)
        
        # Save current checksums
        with open(checksum_file, 'w') as f:
            json.dump(self.file_checksums, f, indent=2)
        
        return changed_files
    
    def suggest_minimal_tests(self, changed_files: Set[str]) -> Dict[str, List[str]]:
        """Suggest minimal tests based on changed files"""
        suggestions = {
            "required_tests": [],
            "recommended_tests": [],
            "commands": []
        }
        
        # Map files to test suggestions
        if any("llm_handler" in f for f in changed_files):
            suggestions["required_tests"].append("LLM connectivity test")
            suggestions["commands"].append("python scripts/test_phase3_setup.py")
        
        if any("retrieval" in f for f in changed_files):
            suggestions["required_tests"].append("Retrieval accuracy test")
            suggestions["commands"].append("python integration_tests/quick_test.py")
        
        if any("Master Prompt" in f for f in changed_files):
            suggestions["required_tests"].append("Golden question set")
            suggestions["recommended_tests"].append("Mode effectiveness check")
        
        if any("model_configs" in f for f in changed_files):
            suggestions["recommended_tests"].append("Single model quick test")
        
        return suggestions
    
    def run_preflight_check(self) -> Dict[str, Any]:
        """Run quick preflight checks before full testing"""
        log_step("Running preflight checks...")
        results = {
            "status": "ready",
            "checks": {},
            "warnings": [],
            "estimated_time": 0
        }
        
        # Check 1: Environment variables
        log_step("Checking environment variables...")
        env_vars = ["OPENROUTER_API_KEY", "GOOGLE_AI_API_KEY", "HUGGINGFACE_TOKEN"]
        for var in env_vars:
            if os.environ.get(var):
                results["checks"][var] = "✅"
            else:
                results["checks"][var] = "❌"
                results["warnings"].append(f"{var} not set")
                results["status"] = "warning"
        
        # Check 2: ChromaDB exists
        log_step("Checking ChromaDB...")
        chroma_path = self.project_root / "chroma_db"
        if chroma_path.exists() and any(chroma_path.iterdir()):
            results["checks"]["ChromaDB"] = "✅"
        else:
            results["checks"]["ChromaDB"] = "❌"
            results["warnings"].append("ChromaDB not initialized")
            results["status"] = "error"
        
        # Check 3: Test questions available
        log_step("Checking test questions...")
        if self.golden_questions:
            results["checks"]["Test Questions"] = "✅"
        else:
            results["checks"]["Test Questions"] = "❌"
            results["warnings"].append("Test questions not found")
            results["status"] = "error"
        
        # Check 4: Estimate time
        num_models = 11
        num_questions = 200
        avg_time_per_question = 3  # seconds
        results["estimated_time"] = (num_models * num_questions * avg_time_per_question) / 3600
        
        return results
    
    def run_golden_test(self, model_id: str = None) -> Dict[str, Any]:
        """Run quick test with golden questions"""
        log_step("Running golden question test...")
        
        # Import necessary components
        from scripts.llm_handler import LLMHandler
        from scripts.adaptive_retrieval import AdaptiveRetrieval
        
        # Initialize components
        llm_handler = LLMHandler(timeout=30)  # Shorter timeout for quick tests
        retrieval = AdaptiveRetrieval()
        
        # Load system prompt
        prompt_path = self.project_root / "kb_files/Instructive/Master Prompt Template.txt"
        system_prompt = prompt_path.read_text() if prompt_path.exists() else ""
        
        # Select model
        if not model_id:
            # Use a fast, free model for quick tests
            model = {
                "id": "mistralai/mistral-nemo:free",
                "api": "openrouter"
            }
        else:
            # Load from config
            with open(self.project_root / "config/model_configs_phase3.yaml", 'r') as f:
                config = yaml.safe_load(f)
            model = next((m for m in config['models'] if m['id'] == model_id), None)
            if not model:
                return {"error": f"Model {model_id} not found in config"}
        
        # Test golden questions
        results = []
        start_time = time.time()
        
        for question in self.golden_questions[:3]:  # Just test first 3 for speed
            log_step(f"Testing question: {question['id']}")
            
            # Retrieve context
            retrieval_result = retrieval.retrieve(
                query=question['question'],
                complexity=question.get('complexity_tier', 'standard'),
                top_k=3
            )
            
            # Build prompt
            context_text = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in retrieval_result['documents']
            ])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question['question']}"}
            ]
            
            # Call LLM
            llm_result = llm_handler.call_llm(
                model_id=model['id'],
                messages=messages,
                model_api=model['api'],
                max_tokens=500
            )
            
            results.append({
                "question_id": question['id'],
                "success": not llm_result['error'],
                "latency": llm_result['latency'],
                "response_preview": llm_result['response'][:200] + "..." if llm_result['response'] else ""
            })
        
        total_time = time.time() - start_time
        
        return {
            "model": model['id'],
            "total_questions": len(results),
            "successful": sum(1 for r in results if r['success']),
            "total_time": total_time,
            "avg_latency": sum(r['latency'] for r in results) / len(results) if results else 0,
            "results": results
        }
    
    def generate_optimization_report(self) -> str:
        """Generate workflow optimization report"""
        # Detect changes
        changed_files = self.detect_changed_files()
        
        # Get suggestions
        suggestions = self.suggest_minimal_tests(changed_files)
        
        # Run preflight
        preflight = self.run_preflight_check()
        
        report = f"""# Workflow Optimization Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Changed Files
"""
        
        if changed_files:
            for file in changed_files:
                report += f"- `{file}`\n"
        else:
            report += "No changes detected in key files.\n"
        
        report += f"""
## Preflight Check

| Check | Status |
|-------|--------|
"""
        
        for check, status in preflight['checks'].items():
            report += f"| {check} | {status} |\n"
        
        if preflight['warnings']:
            report += "\n### ⚠️ Warnings:\n"
            for warning in preflight['warnings']:
                report += f"- {warning}\n"
        
        report += f"""
## Test Recommendations

### Required Tests:
"""
        
        if suggestions['required_tests']:
            for test in suggestions['required_tests']:
                report += f"- {test}\n"
        else:
            report += "- None (no critical changes)\n"
        
        report += "\n### Suggested Commands:\n```bash\n"
        if suggestions['commands']:
            for cmd in suggestions['commands']:
                report += f"{cmd}\n"
        else:
            report += "# No specific tests needed\n"
        
        report += "```\n"
        
        report += f"""
## Time Estimates

- **Golden Test Set**: ~30 seconds
- **Single Model Test**: ~10 minutes
- **Full Evaluation**: ~{preflight['estimated_time']:.1f} hours

## Quick Test Command

For fastest feedback:
```bash
python scripts/agents/workflow_optimizer.py --golden-test
```
"""
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize testing workflows"
    )
    parser.add_argument(
        "--check",
        action='store_true',
        help="Run preflight checks only"
    )
    parser.add_argument(
        "--golden-test",
        action='store_true',
        help="Run golden question test"
    )
    parser.add_argument(
        "--model",
        help="Model ID for golden test"
    )
    parser.add_argument(
        "--report",
        action='store_true',
        help="Generate optimization report"
    )
    
    args = parser.parse_args()
    
    optimizer = WorkflowOptimizer()
    
    if args.check:
        results = optimizer.run_preflight_check()
        print(json.dumps(results, indent=2))
    
    elif args.golden_test:
        results = optimizer.run_golden_test(args.model)
        print(f"\nGolden Test Results:")
        print(f"Model: {results['model']}")
        print(f"Success Rate: {results['successful']}/{results['total_questions']}")
        print(f"Avg Latency: {results['avg_latency']:.2f}s")
        print(f"Total Time: {results['total_time']:.1f}s")
    
    elif args.report:
        report = optimizer.generate_optimization_report()
        print(report)
        
        # Save report
        report_path = Path("results/workflow_optimization_report.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    else:
        # Default: show current status
        changed = optimizer.detect_changed_files()
        if changed:
            print("Changed files detected:")
            for f in changed:
                print(f"  - {f}")
            print("\nRun with --report for recommendations")
        else:
            print("No changes detected. System ready for testing.")


if __name__ == "__main__":
    main()