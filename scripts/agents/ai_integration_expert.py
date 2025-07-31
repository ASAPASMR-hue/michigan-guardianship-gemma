#!/usr/bin/env python3
"""
AI Integration Expert Agent
Specializes in prompt engineering and AI optimization
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step
from scripts.llm_handler import LLMHandler
from scripts.adaptive_retrieval import AdaptiveRetrieval


class AIIntegrationExpert:
    """Expert in prompt optimization and AI techniques"""
    
    def __init__(self):
        """Initialize AI expert"""
        self.project_root = Path(__file__).parent.parent.parent
        self.llm_handler = LLMHandler(timeout=60)
        self.retrieval = AdaptiveRetrieval()
        
        # Load current prompts
        self.current_prompts = self._load_current_prompts()
        
        # Failed test cases for analysis
        self.failed_cases = []
        
    def _load_current_prompts(self) -> Dict[str, str]:
        """Load current system prompts"""
        prompts = {}
        
        prompt_files = {
            "master": "kb_files/Instructive/Master Prompt Template.txt",
            "dynamic_mode": "kb_files/Instructive/Dynamic Mode Examples V2.txt",
            "genesee": "kb_files/Instructive/Genesee County Specifics.txt",
            "out_of_scope": "kb_files/Instructive/Out-of-Scope Guidelines.txt"
        }
        
        for name, path in prompt_files.items():
            full_path = self.project_root / path
            if full_path.exists():
                prompts[name] = full_path.read_text()
            else:
                log_step(f"Warning: Prompt file not found: {path}", level="warning")
        
        return prompts
    
    def analyze_failed_cases(self, run_id: str) -> List[Dict[str, Any]]:
        """Analyze failed test cases from a run"""
        log_step(f"Analyzing failed cases from run: {run_id}")
        
        results_dir = Path(f"results/{run_id}")
        if not results_dir.exists():
            log_step(f"Results directory not found: {results_dir}", level="error")
            return []
        
        failed_cases = []
        
        # Load evaluation metrics
        metrics_path = results_dir / "evaluation_metrics.json"
        if not metrics_path.exists():
            log_step("Evaluation metrics not found. Run analyze_results.py first.", level="warning")
            return []
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Find models with low scores
        for model_data in metrics:
            if model_data['average_scores']['total_score'] < 6.0:
                # Load raw results for this model
                model_name = model_data['model'].replace("/", "_").replace(":", "_")
                results_file = results_dir / f"{model_name}_results.json"
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Find specific failures
                    for result in results:
                        if result.get('error') or not result.get('response'):
                            continue
                        
                        # Analyze why it might have failed
                        # (In a real implementation, we'd compare against expected answers)
                        if len(result['response']) < 100:
                            failed_cases.append({
                                'question_id': result['question_id'],
                                'question': result['question_text'],
                                'model': model_data['model'],
                                'response': result['response'],
                                'issue': 'Response too short',
                                 'complexity': result.get('complexity_tier', 'unknown')
                            })
        
        self.failed_cases = failed_cases
        log_step(f"Found {len(failed_cases)} problematic cases")
        return failed_cases
    
    def generate_prompt_variations(self, target_issue: str) -> List[Dict[str, str]]:
        """Generate prompt variations to address specific issues"""
        log_step(f"Generating prompt variations for: {target_issue}")
        
        variations = []
        base_prompt = self.current_prompts.get('master', '')
        
        if target_issue == "tribal_notification":
            # Variations for ICWA/tribal notification issues
            variations.extend([
                {
                    "name": "explicit_tribal_emphasis",
                    "description": "Add explicit emphasis on tribal notification requirements",
                    "modification": """
### CRITICAL ICWA/MIFPA REQUIREMENTS

When ANY question involves Native American/Indian children, you MUST:
1. State that tribal notification is MANDATORY (Form PC 678)
2. Provide the complete BIA address: Midwest Regional Director, Bureau of Indian Affairs, Norman Pointe II Building, 5600 W. American Blvd., Suite 500, Bloomington, MN 55437
3. Explain the tribe can request up to 20 extra days
4. Emphasize that private agreements without court oversight are ILLEGAL

"""
                },
                {
                    "name": "few_shot_tribal",
                    "description": "Add few-shot examples for tribal questions",
                    "modification": """
### Example: ICWA Question
Q: "Can I handle guardianship privately for my Indian grandchild?"
A: "No. ICWA and MIFPA are federal and state laws that mandate court oversight for any guardianship placement of an Indian child. A private agreement would be legally invalid and violate the law. You must file with the court and notify the tribe using Form PC 678."

"""
                }
            ])
        
        elif target_issue == "procedural_accuracy":
            variations.extend([
                {
                    "name": "procedural_checklist",
                    "description": "Add explicit procedural checklist",
                    "modification": """
### PROCEDURAL ACCURACY CHECKLIST
For EVERY response, verify you have included:
☐ Correct form numbers (PC 651 for full, PC 650 for limited)
☐ Service deadlines (7 days personal, 14 days mail, 5 days proof filing)
☐ Filing fee: EXACTLY $175 (waiver available with Form MC 20)
☐ Hearing day: THURSDAYS ONLY in Genesee County
☐ Courthouse: 900 S. Saginaw St., Room 502, Flint, MI 48502

"""
                }
            ])
        
        return variations
    
    def test_prompt_variation(
        self,
        variation: Dict[str, str],
        test_questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test a prompt variation on specific questions"""
        log_step(f"Testing prompt variation: {variation['name']}")
        
        # Modify the base prompt
        modified_prompt = self.current_prompts.get('master', '') + "\n\n" + variation['modification']
        
        results = []
        
        for question in test_questions[:5]:  # Test on first 5 questions
            # Retrieve context
            retrieval_result = self.retrieval.retrieve(
                query=question['question'],
                complexity=question.get('complexity', 'standard'),
                top_k=3
            )
            
            # Build prompt
            context_text = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in retrieval_result['documents']
            ])
            
            messages = [
                {"role": "system", "content": modified_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question['question']}"}
            ]
            
            # Test with a fast model
            llm_result = self.llm_handler.call_llm(
                model_id="mistralai/mistral-nemo:free",
                messages=messages,
                model_api="openrouter",
                max_tokens=500
            )
            
            # Simple evaluation
            response = llm_result['response']
            score = 0
            
            # Check for key elements based on issue
            if "tribal" in variation['name']:
                if "PC 678" in response:
                    score += 1
                if "Bloomington, MN" in response:
                    score += 1
                if "20 extra days" in response or "20 days" in response:
                    score += 1
            elif "procedural" in variation['name']:
                if "$175" in response:
                    score += 1
                if "Thursday" in response:
                    score += 1
                if "PC 651" in response or "PC 650" in response:
                    score += 1
            
            results.append({
                'question_id': question.get('id', 'unknown'),
                'score': score,
                'response_length': len(response),
                'has_citations': bool(re.search(r'\(MCL \d+\.\d+\)', response))
            })
        
        # Calculate metrics
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0
        avg_length = sum(r['response_length'] for r in results) / len(results) if results else 0
        citation_rate = sum(1 for r in results if r['has_citations']) / len(results) if results else 0
        
        return {
            'variation_name': variation['name'],
            'description': variation['description'],
            'avg_score': avg_score,
            'avg_response_length': avg_length,
            'citation_rate': citation_rate,
            'num_tested': len(results)
        }
    
    def optimize_for_failed_cases(self) -> Dict[str, Any]:
        """Optimize prompts based on failed cases"""
        if not self.failed_cases:
            return {"error": "No failed cases loaded. Run analyze_failed_cases first."}
        
        # Group failures by issue type
        issues = {}
        for case in self.failed_cases:
            issue = case['issue']
            if issue not in issues:
                issues[issue] = []
            issues[issue].append(case)
        
        optimization_results = {}
        
        # Generate and test variations for each issue
        for issue, cases in issues.items():
            log_step(f"Optimizing for issue: {issue}")
            
            # Map issues to variation types
            if "tribal" in str(cases).lower() or "icwa" in str(cases).lower():
                target_issue = "tribal_notification"
            elif "procedural" in str(cases).lower() or "$175" in str(cases).lower():
                target_issue = "procedural_accuracy"
            else:
                target_issue = "general"
            
            variations = self.generate_prompt_variations(target_issue)
            
            # Test each variation
            test_results = []
            for variation in variations:
                result = self.test_prompt_variation(variation, cases[:5])
                test_results.append(result)
            
            # Find best variation
            best_variation = max(test_results, key=lambda x: x['avg_score'])
            optimization_results[issue] = {
                'tested_variations': len(variations),
                'best_variation': best_variation,
                'improvement': best_variation['avg_score']
            }
        
        return optimization_results
    
    def generate_optimization_report(self, run_id: str) -> str:
        """Generate comprehensive prompt optimization report"""
        # Analyze failed cases
        failed_cases = self.analyze_failed_cases(run_id)
        
        # Run optimization
        optimization_results = self.optimize_for_failed_cases()
        
        report = f"""# AI Integration Expert Report

**Run ID**: {run_id}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Failed Case Analysis

Total problematic cases identified: {len(failed_cases)}

### Issues Breakdown:
"""
        
        # Group by issue type
        issue_counts = {}
        for case in failed_cases:
            issue = case['issue']
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        for issue, count in issue_counts.items():
            report += f"- {issue}: {count} cases\n"
        
        report += """
## Prompt Optimization Results

### Tested Variations:
"""
        
        for issue, results in optimization_results.items():
            if 'error' not in results:
                report += f"\n**Issue: {issue}**\n"
                report += f"- Variations tested: {results['tested_variations']}\n"
                report += f"- Best variation: {results['best_variation']['variation_name']}\n"
                report += f"- Score improvement: {results['improvement']:.2f}\n"
        
        report += """
## Recommended Prompt Modifications

Based on the analysis, here are the recommended modifications:

### 1. Enhanced ICWA/Tribal Handling
```
When ANY question involves Native American/Indian children, you MUST:
1. State that tribal notification is MANDATORY (Form PC 678)
2. Provide the complete BIA address
3. Explain the tribe can request up to 20 extra days
4. Emphasize that private agreements without court oversight are ILLEGAL
```

### 2. Procedural Accuracy Improvements
```
For EVERY response, verify inclusion of:
- Correct form numbers (PC 651 vs PC 650)
- Service deadlines (7/14/5 days)
- Filing fee: EXACTLY $175
- Hearing day: THURSDAYS ONLY
- Complete courthouse address
```

### 3. Dynamic Few-Shot Examples
Consider adding successful response examples for common failure patterns directly into the prompt.

## Implementation Strategy

1. **Test modifications** on golden question set first
2. **A/B test** original vs modified prompts
3. **Monitor** for any regression in other areas
4. **Iterate** based on results

## Next Steps

1. Apply recommended modifications to `Master Prompt Template.txt`
2. Run golden test with modified prompt
3. If successful, run full evaluation on subset of models
4. Deploy best performing variation
"""
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Integration Expert - Prompt optimization"
    )
    parser.add_argument(
        "--analyze",
        required=True,
        help="Run ID to analyze"
    )
    parser.add_argument(
        "--optimize",
        action='store_true',
        help="Run prompt optimization"
    )
    parser.add_argument(
        "--report",
        action='store_true',
        help="Generate optimization report"
    )
    
    args = parser.parse_args()
    
    expert = AIIntegrationExpert()
    
    if args.optimize:
        failed_cases = expert.analyze_failed_cases(args.analyze)
        if failed_cases:
            results = expert.optimize_for_failed_cases()
            print(json.dumps(results, indent=2))
        else:
            print("No failed cases found to optimize")
    
    elif args.report:
        report = expert.generate_optimization_report(args.analyze)
        print(report)
        
        # Save report
        report_path = Path(f"results/ai_optimization_report_{args.analyze}.md")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    else:
        # Just analyze
        failed_cases = expert.analyze_failed_cases(args.analyze)
        print(f"Found {len(failed_cases)} problematic cases")
        if failed_cases:
            print("\nSample failures:")
            for case in failed_cases[:3]:
                print(f"- {case['question_id']}: {case['issue']}")


if __name__ == "__main__":
    main()