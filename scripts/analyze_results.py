#!/usr/bin/env python3
"""
Phase 3b: Results Analysis and Evaluation
Analyzes raw test data and generates comprehensive reports
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.logger import log_step
from integration_tests.full_pipeline_test import IntegrationTester


class ResultsAnalyzer:
    """Analyzes test results and generates evaluation reports"""
    
    def __init__(self, run_id: str):
        """Initialize analyzer with specific run"""
        self.run_id = run_id
        self.results_dir = Path(f"results/{run_id}")
        
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {self.results_dir}")
        
        # Load evaluation rubric
        with open("rubrics/eval_rubric.yaml", 'r') as f:
            self.rubric = yaml.safe_load(f)['evaluation_rubric']
        
        # Initialize validator for semantic similarity
        self.validator = IntegrationTester()
        
        # Load metadata
        with open(self.results_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        log_step(f"Initialized analyzer for run: {run_id}")
    
    def load_all_results(self) -> Dict[str, List[Dict]]:
        """Load all model results"""
        results = {}
        
        for result_file in self.results_dir.glob("*_results.json"):
            if result_file.name == "failed_models.json":
                continue
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            model_name = result_file.stem.replace("_results", "")
            results[model_name] = data
            
            log_step(f"Loaded {len(data)} results for {model_name}")
        
        return results
    
    def evaluate_response(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single response against the rubric"""
        scores = {}
        response = result.get('response', '')
        question = result.get('question_text', '')
        complexity = result.get('complexity_tier', 'standard')
        
        # Handle empty responses
        if not response or result.get('error'):
            return {
                'procedural_accuracy': 0,
                'substantive_legal_accuracy': 0,
                'actionability': 0,
                'mode_effectiveness': 0,
                'strategic_caution': 0,
                'citation_quality': 0,
                'harm_prevention': 0,
                'total_score': 0
            }
        
        # 1. Procedural Accuracy (2.5 pts)
        proc_score = self._evaluate_procedural_accuracy(response, complexity)
        scores['procedural_accuracy'] = proc_score
        
        # 2. Substantive Legal Accuracy (2.0 pts)
        legal_score = self._evaluate_legal_accuracy(response, complexity)
        scores['substantive_legal_accuracy'] = legal_score
        
        # 3. Actionability (2.0 pts)
        action_score = self._evaluate_actionability(response)
        scores['actionability'] = action_score
        
        # 4. Mode Effectiveness (1.5 pts)
        mode_score = self._evaluate_mode_effectiveness(response, complexity)
        scores['mode_effectiveness'] = mode_score
        
        # 5. Strategic Caution (0.5 pts)
        caution_score = self._evaluate_strategic_caution(response)
        scores['strategic_caution'] = caution_score
        
        # 6. Citation Quality (0.5 pts)
        citation_score = self._evaluate_citation_quality(response)
        scores['citation_quality'] = citation_score
        
        # 7. Harm Prevention (0.5 pts)
        harm_score = self._evaluate_harm_prevention(response)
        scores['harm_prevention'] = harm_score
        
        # Calculate total
        scores['total_score'] = sum(scores.values())
        
        return scores
    
    def _evaluate_procedural_accuracy(self, response: str, complexity: str) -> float:
        """Evaluate procedural accuracy (forms, deadlines, fees)"""
        weight = self.rubric['procedural_accuracy']['adaptive_weight_by_complexity'].get(
            complexity, 2.5
        )
        
        critical_items = self.rubric['procedural_accuracy']['critical_items']
        score = 0
        max_score = len(critical_items)
        
        # Check for critical procedural elements
        if "PC 651" in response or "PC 650" in response:
            score += 1
        if any(deadline in response for deadline in ["7 day", "14 day", "5 day"]):
            score += 1
        if "$175" in response:
            score += 1
        if "Thursday" in response:
            score += 1
        if "900 S. Saginaw" in response or "Room 502" in response:
            score += 1
        
        # Apply fail override if any critical error
        if self.rubric['procedural_accuracy'].get('fail_override'):
            if "$175" in response.replace("$175", ""):  # Wrong fee amount
                return 0
        
        return (score / max_score) * weight
    
    def _evaluate_legal_accuracy(self, response: str, complexity: str) -> float:
        """Evaluate substantive legal accuracy"""
        weight = self.rubric['substantive_legal_accuracy']['adaptive_weight_by_complexity'].get(
            complexity, 2.0
        )
        
        score = 0
        max_score = 4
        
        # Check for key legal concepts
        if "MCL 700" in response:
            score += 1
        if any(term in response for term in ["consent", "grounds", "best interests"]):
            score += 1
        if "ICWA" in response or "MIFPA" in response:
            score += 0.5
        if any(term in response for term in ["guardian", "conservator", "duties"]):
            score += 1
        if any(term in response for term in ["termination", "modification", "review"]):
            score += 0.5
        
        return min((score / max_score) * weight, weight)
    
    def _evaluate_actionability(self, response: str) -> float:
        """Evaluate actionability of response"""
        weight = 2.0
        requirements = self.rubric['actionability']['requirements']
        
        score = 0
        max_score = len(requirements)
        
        # Check for actionable elements
        if any(form in response for form in ["Form PC", "PC "]):
            score += 1
        if "900 S. Saginaw" in response or "courthouse" in response:
            score += 1
        if any(term in response for term in ["bring", "need", "required", "must"]):
            score += 1
        if any(term in response for term in ["next", "then", "after", "timeline"]):
            score += 1
        
        return (score / max_score) * weight
    
    def _evaluate_mode_effectiveness(self, response: str, complexity: str) -> float:
        """Evaluate dynamic mode switching effectiveness"""
        weight = self.rubric['mode_effectiveness']['adaptive_weight_by_complexity'].get(
            complexity, 1.5
        )
        
        # Simple heuristic: check for balance of legal and empathetic language
        legal_terms = sum(1 for term in ["MCL", "PC", "statute", "requirement"] if term in response)
        empathy_terms = sum(1 for term in ["understand", "help", "support", "difficult"] if term in response)
        
        if legal_terms > 0 and empathy_terms > 0:
            balance_score = 1.0
        elif legal_terms > 0 or empathy_terms > 0:
            balance_score = 0.5
        else:
            balance_score = 0
        
        return balance_score * weight
    
    def _evaluate_strategic_caution(self, response: str) -> float:
        """Evaluate strategic caution"""
        weight = 0.5
        
        good_cautions = [
            "court needs evidence",
            "ICWA cases take longer",
            "if.*object",
            "may require"
        ]
        
        score = sum(0.25 for caution in good_cautions 
                   if any(term in response.lower() for term in caution.split()))
        
        return min(score * weight, weight)
    
    def _evaluate_citation_quality(self, response: str) -> float:
        """Evaluate citation quality"""
        weight = 0.5
        
        # Count citations
        citations = response.count("(MCL") + response.count("(Form") + response.count("(PC")
        
        if citations >= 3:
            return weight
        elif citations >= 1:
            return weight * 0.5
        else:
            return 0
    
    def _evaluate_harm_prevention(self, response: str) -> float:
        """Evaluate harm prevention"""
        weight = 0.5
        
        red_flags = self.rubric['harm_prevention']['red_flags']
        
        # Check for harmful advice
        for flag in red_flags:
            if flag.lower() in response.lower():
                return 0
        
        return weight
    
    def analyze_model_performance(self, model_name: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance for a single model"""
        log_step(f"Analyzing {model_name}...")
        
        # Evaluate each response
        evaluations = []
        for result in results:
            scores = self.evaluate_response(result)
            evaluations.append({
                'question_id': result['question_id'],
                'complexity': result.get('complexity_tier', 'standard'),
                'scores': scores,
                'error': result.get('error'),
                'latency': result.get('latency', 0),
                'cost_usd': result.get('cost_usd', 0)
            })
        
        # Calculate aggregate metrics
        valid_evals = [e for e in evaluations if not e['error']]
        
        metrics = {
            'model': model_name,
            'total_questions': len(results),
            'successful_responses': len(valid_evals),
            'error_rate': (len(results) - len(valid_evals)) / len(results) if results else 0,
            'timeout_rate': sum(1 for r in results if r.get('error') == 'timeout') / len(results) if results else 0,
            'average_scores': {},
            'complexity_breakdown': {},
            'total_cost': sum(r.get('cost_usd', 0) for r in results),
            'average_latency': np.mean([r.get('latency', 0) for r in results if r.get('latency')])
        }
        
        # Calculate average scores
        if valid_evals:
            score_keys = valid_evals[0]['scores'].keys()
            for key in score_keys:
                scores = [e['scores'][key] for e in valid_evals]
                metrics['average_scores'][key] = np.mean(scores)
        
        # Breakdown by complexity
        for complexity in ['simple', 'standard', 'complex', 'crisis']:
            complexity_evals = [e for e in valid_evals if e['complexity'] == complexity]
            if complexity_evals:
                avg_total = np.mean([e['scores']['total_score'] for e in complexity_evals])
                metrics['complexity_breakdown'][complexity] = {
                    'count': len(complexity_evals),
                    'average_score': avg_total
                }
        
        return metrics
    
    def generate_report(self, output_path: str = None):
        """Generate comprehensive evaluation report"""
        log_step("Generating evaluation report...")
        
        # Load all results
        all_results = self.load_all_results()
        
        # Analyze each model
        model_metrics = []
        for model_name, results in all_results.items():
            metrics = self.analyze_model_performance(model_name, results)
            model_metrics.append(metrics)
        
        # Sort by overall score
        model_metrics.sort(
            key=lambda x: x['average_scores'].get('total_score', 0),
            reverse=True
        )
        
        # Generate report
        report = self._format_report(model_metrics)
        
        # Save report
        if not output_path:
            output_path = self.results_dir / "evaluation_summary.md"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        log_step(f"Report saved to: {output_path}")
        
        # Also save metrics as JSON
        json_path = self.results_dir / "evaluation_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        return model_metrics
    
    def _format_report(self, metrics: List[Dict]) -> str:
        """Format evaluation report as markdown"""
        report = f"""# Michigan Guardianship AI - Phase 3 Evaluation Report

**Run ID**: {self.run_id}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Git Commit**: {self.metadata.get('git_commit_hash', 'unknown')}  
**Total Questions**: {self.metadata.get('total_questions', 0)}  
**Models Tested**: {len(metrics)}

## Executive Summary

### Top Performing Models (by Total Score)

| Rank | Model | Total Score | Success Rate | Avg Latency | Total Cost |
|------|-------|-------------|--------------|-------------|------------|
"""
        
        for i, m in enumerate(metrics[:5], 1):
            total_score = m['average_scores'].get('total_score', 0)
            success_rate = (1 - m['error_rate']) * 100
            report += f"| {i} | {m['model']} | {total_score:.2f}/10 | {success_rate:.1f}% | {m['average_latency']:.1f}s | ${m['total_cost']:.4f} |\n"
        
        report += """
## Detailed Scoring Breakdown

### Score Components (Maximum Points)

- **Procedural Accuracy** (2.5): Forms, deadlines, fees, courthouse details
- **Substantive Legal** (2.0): Legal concepts, statutes, requirements
- **Actionability** (2.0): Concrete steps and guidance
- **Mode Effectiveness** (1.5): Balance of legal facts and empathy
- **Strategic Caution** (0.5): Warnings and caveats
- **Citation Quality** (0.5): Proper inline citations
- **Harm Prevention** (0.5): Avoiding dangerous advice

### Full Results

| Model | Procedural | Legal | Action | Mode | Caution | Citation | Harm | **Total** |
|-------|------------|-------|---------|------|---------|----------|------|-----------|
"""
        
        for m in metrics:
            scores = m['average_scores']
            if scores:
                report += f"| {m['model']} "
                report += f"| {scores.get('procedural_accuracy', 0):.2f} "
                report += f"| {scores.get('substantive_legal_accuracy', 0):.2f} "
                report += f"| {scores.get('actionability', 0):.2f} "
                report += f"| {scores.get('mode_effectiveness', 0):.2f} "
                report += f"| {scores.get('strategic_caution', 0):.2f} "
                report += f"| {scores.get('citation_quality', 0):.2f} "
                report += f"| {scores.get('harm_prevention', 0):.2f} "
                report += f"| **{scores.get('total_score', 0):.2f}** |\n"
        
        report += """
## Performance by Complexity Tier

| Model | Simple | Standard | Complex | Crisis |
|-------|--------|----------|---------|---------|
"""
        
        for m in metrics[:10]:  # Top 10 models
            report += f"| {m['model']} "
            for tier in ['simple', 'standard', 'complex', 'crisis']:
                if tier in m['complexity_breakdown']:
                    score = m['complexity_breakdown'][tier]['average_score']
                    count = m['complexity_breakdown'][tier]['count']
                    report += f"| {score:.2f} (n={count}) "
                else:
                    report += "| N/A "
            report += "|\n"
        
        report += """
## Cost-Benefit Analysis

| Model | Cost per Question | Quality Score | Value Ratio |
|-------|-------------------|---------------|-------------|
"""
        
        for m in metrics:
            if m['total_questions'] > 0:
                cost_per_q = m['total_cost'] / m['total_questions']
                quality = m['average_scores'].get('total_score', 0)
                if cost_per_q > 0:
                    value_ratio = quality / cost_per_q
                else:
                    value_ratio = float('inf') if quality > 0 else 0
                
                report += f"| {m['model']} | ${cost_per_q:.5f} | {quality:.2f} | "
                if value_ratio == float('inf'):
                    report += "∞ (free) |\n"
                else:
                    report += f"{value_ratio:.1f} |\n"
        
        report += """
## Error Analysis

| Model | Error Rate | Timeout Rate | Common Issues |
|-------|------------|--------------|---------------|
"""
        
        for m in metrics[:10]:
            error_rate = m['error_rate'] * 100
            timeout_rate = m['timeout_rate'] * 100
            issues = []
            if timeout_rate > 10:
                issues.append("High timeout rate")
            if error_rate > 20:
                issues.append("High error rate")
            
            report += f"| {m['model']} | {error_rate:.1f}% | {timeout_rate:.1f}% | {', '.join(issues) or 'None'} |\n"
        
        report += """
## Recommendations

Based on this evaluation:

1. **Best Overall**: Models with highest total scores and good reliability
2. **Best Value**: Free models with scores > 6.0 provide excellent cost-effectiveness
3. **Best for Complex**: Check complex tier performance for nuanced questions
4. **Speed Critical**: Models with < 2s latency suitable for real-time use

## Notes

- Scores are out of 10 points total
- Timeout was set at 180 seconds
- Cost calculations include both input and output tokens
- Value Ratio = Quality Score / Cost per Question
"""
        
        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze Phase 3 test results"
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="Run ID to analyze (e.g., run_20250128_1430)"
    )
    parser.add_argument(
        "--output",
        help="Output path for report (default: results/{run_id}/evaluation_summary.md)"
    )
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.run_id)
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()