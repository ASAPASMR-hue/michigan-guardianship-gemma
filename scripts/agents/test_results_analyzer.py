#!/usr/bin/env python3
"""
Test Results Analyzer Agent
Advanced analysis of Phase 3 test results with qualitative insights
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class TestResultsAnalyzer:
    """Advanced analyzer for comprehensive test insights"""
    
    def __init__(self, run_ids: List[str]):
        """
        Initialize analyzer with one or more run IDs
        
        Args:
            run_ids: List of run IDs to analyze
        """
        self.run_ids = run_ids
        self.all_results = {}
        self.all_metadata = {}
        
        # Load rubric for reference
        with open("rubrics/eval_rubric.yaml", 'r') as f:
            self.rubric = yaml.safe_load(f)['evaluation_rubric']
        
        # Load all runs
        for run_id in run_ids:
            self._load_run(run_id)
    
    def _load_run(self, run_id: str):
        """Load results and metadata for a single run"""
        results_dir = Path(f"results/{run_id}")
        
        if not results_dir.exists():
            log_step(f"Warning: Results directory not found for {run_id}", level="warning")
            return
        
        # Load metadata
        with open(results_dir / "metadata.json", 'r') as f:
            self.all_metadata[run_id] = json.load(f)
        
        # Load evaluation metrics if available
        metrics_path = results_dir / "evaluation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.all_results[run_id] = {m['model']: m for m in metrics}
        else:
            log_step(f"No evaluation metrics found for {run_id}. Run analyze_results.py first.", level="warning")
    
    def analyze_qualitative_trends(self, run_id: str) -> Dict[str, Any]:
        """Identify qualitative trends in model performance"""
        log_step(f"Analyzing qualitative trends for {run_id}...")
        
        trends = {
            "model_strengths": {},
            "model_weaknesses": {},
            "category_insights": {},
            "common_failures": []
        }
        
        # Analyze each model's performance patterns
        for model_name, metrics in self.all_results.get(run_id, {}).items():
            strengths = []
            weaknesses = []
            
            # Check component scores
            scores = metrics.get('average_scores', {})
            
            # Identify strengths (above 80% of max)
            if scores.get('procedural_accuracy', 0) > 2.0:
                strengths.append("Excellent procedural accuracy (forms, fees, deadlines)")
            if scores.get('actionability', 0) > 1.6:
                strengths.append("Highly actionable responses with clear next steps")
            if scores.get('citation_quality', 0) > 0.4:
                strengths.append("Strong citation practices")
            
            # Identify weaknesses (below 50% of max)
            if scores.get('procedural_accuracy', 0) < 1.25:
                weaknesses.append("Weak on procedural details")
            if scores.get('mode_effectiveness', 0) < 0.75:
                weaknesses.append("Poor balance of legal facts and empathy")
            if scores.get('strategic_caution', 0) < 0.25:
                weaknesses.append("Lacks strategic warnings and caveats")
            
            # Check complexity performance
            complexity_breakdown = metrics.get('complexity_breakdown', {})
            if 'complex' in complexity_breakdown and 'simple' in complexity_breakdown:
                complex_score = complexity_breakdown['complex']['average_score']
                simple_score = complexity_breakdown['simple']['average_score']
                
                if complex_score > simple_score * 0.9:
                    strengths.append("Maintains quality on complex questions")
                elif complex_score < simple_score * 0.7:
                    weaknesses.append("Significant degradation on complex questions")
            
            # ICWA/Crisis performance (would need to analyze raw results for this)
            # Placeholder for now
            
            trends["model_strengths"][model_name] = strengths
            trends["model_weaknesses"][model_name] = weaknesses
        
        return trends
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary across all runs"""
        summary = f"""# Michigan Guardianship AI - Executive Summary

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Runs Analyzed**: {', '.join(self.run_ids)}

## Key Findings

### Top Performers
"""
        
        # Find best models across all runs
        all_scores = []
        for run_id in self.run_ids:
            for model_name, metrics in self.all_results.get(run_id, {}).items():
                total_score = metrics.get('average_scores', {}).get('total_score', 0)
                all_scores.append({
                    'model': model_name,
                    'run_id': run_id,
                    'score': total_score,
                    'cost': metrics.get('total_cost', 0),
                    'success_rate': 1 - metrics.get('error_rate', 0)
                })
        
        # Sort by score
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Top 3 overall
        summary += "\n**Overall Best Models:**\n"
        for i, item in enumerate(all_scores[:3], 1):
            summary += f"{i}. **{item['model']}** - Score: {item['score']:.2f}/10 "
            summary += f"(Success Rate: {item['success_rate']*100:.1f}%)\n"
        
        # Best free models
        free_models = [s for s in all_scores if s['cost'] == 0]
        if free_models:
            summary += "\n**Best Free Models:**\n"
            for i, item in enumerate(free_models[:3], 1):
                summary += f"{i}. **{item['model']}** - Score: {item['score']:.2f}/10\n"
        
        # Qualitative insights
        summary += "\n### Qualitative Insights\n\n"
        
        for run_id in self.run_ids[:1]:  # Just analyze the first run for brevity
            trends = self.analyze_qualitative_trends(run_id)
            
            # Find models with specific strengths
            icwa_strong = []
            procedural_strong = []
            
            for model, strengths in trends['model_strengths'].items():
                if any('procedural' in s.lower() for s in strengths):
                    procedural_strong.append(model)
            
            if procedural_strong:
                summary += f"**Strong on Procedural Details**: {', '.join(procedural_strong[:3])}\n\n"
        
        summary += self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> str:
        """Generate strategic recommendations"""
        recommendations = """
## Strategic Recommendations

### For Production Deployment

1. **Tiered Approach**:
   - Use free models (Llama 3.3 70B, Mistral Nemo) for simple questions
   - Reserve premium models for complex/crisis situations
   - Implement fallback chains for reliability

2. **Model Selection by Use Case**:
   - **High Volume, Simple Queries**: Free models with score > 6.0
   - **Complex Legal Questions**: Models with strong complexity scores
   - **Crisis Situations**: Models with high mode effectiveness scores

3. **Cost Optimization**:
   - Estimated monthly cost for 1000 queries/day:
     - All premium: $X.XX
     - Tiered approach: $Y.YY (ZZ% savings)

### Next Steps

1. **Pilot Testing**: Deploy top 3 models in staging environment
2. **A/B Testing**: Compare user satisfaction between models
3. **Fine-tuning**: Consider fine-tuning best open models on your specific data
4. **Monitoring**: Track real-world performance metrics

### Risk Mitigation

- Implement response validation for all models
- Set up fallback to human expert for low-confidence responses
- Regular retraining as laws and procedures change
"""
        return recommendations
    
    def compare_runs(self) -> pd.DataFrame:
        """Compare performance across multiple runs"""
        if len(self.run_ids) < 2:
            log_step("Need at least 2 runs to compare", level="warning")
            return pd.DataFrame()
        
        comparison_data = []
        
        # Get all unique models
        all_models = set()
        for run_id in self.run_ids:
            all_models.update(self.all_results.get(run_id, {}).keys())
        
        # Compare each model across runs
        for model in all_models:
            row = {'model': model}
            
            for run_id in self.run_ids:
                if model in self.all_results.get(run_id, {}):
                    metrics = self.all_results[run_id][model]
                    score = metrics.get('average_scores', {}).get('total_score', 0)
                    row[f'{run_id}_score'] = score
                else:
                    row[f'{run_id}_score'] = None
            
            # Calculate improvement if applicable
            if len(self.run_ids) == 2:
                score1 = row.get(f'{self.run_ids[0]}_score')
                score2 = row.get(f'{self.run_ids[1]}_score')
                if score1 is not None and score2 is not None and score1 > 0:
                    row['improvement_%'] = ((score2 - score1) / score1) * 100
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('model')
    
    def identify_failure_patterns(self, run_id: str) -> Dict[str, List[str]]:
        """Identify common failure patterns across models"""
        log_step(f"Identifying failure patterns for {run_id}...")
        
        patterns = defaultdict(list)
        
        # Load raw results to analyze failures
        results_dir = Path(f"results/{run_id}")
        
        for result_file in results_dir.glob("*_results.json"):
            if result_file.name == "failed_models.json":
                continue
            
            model_name = result_file.stem.replace("_results", "")
            
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Analyze errors
            for result in results:
                if result.get('error'):
                    error_type = result['error']
                    question_id = result.get('question_id', 'unknown')
                    
                    if error_type == 'timeout':
                        patterns['timeout_questions'].append(question_id)
                    elif 'rate limit' in error_type.lower():
                        patterns['rate_limited_models'].append(model_name)
                    else:
                        patterns['other_errors'].append(f"{model_name}: {error_type}")
        
        # Find questions that consistently timeout
        timeout_counter = Counter(patterns['timeout_questions'])
        patterns['problematic_questions'] = [
            q for q, count in timeout_counter.items() if count > 2
        ]
        
        return dict(patterns)
    
    def generate_detailed_report(self, output_path: str = None):
        """Generate comprehensive analysis report"""
        log_step("Generating detailed analysis report...")
        
        # Generate executive summary
        report = self.generate_executive_summary()
        
        # Add detailed analysis for each run
        report += "\n\n## Detailed Run Analysis\n"
        
        for run_id in self.run_ids:
            report += f"\n### Run: {run_id}\n"
            
            metadata = self.all_metadata.get(run_id, {})
            report += f"- **Date**: {metadata.get('start_time', 'Unknown')}\n"
            report += f"- **Git Commit**: `{metadata.get('git_commit_hash', 'Unknown')[:8]}`\n"
            report += f"- **Models Tested**: {len(self.all_results.get(run_id, {}))}\n"
            
            # Qualitative trends
            trends = self.analyze_qualitative_trends(run_id)
            
            report += "\n**Model-Specific Insights:**\n"
            for model, strengths in trends['model_strengths'].items():
                if strengths:
                    report += f"\n*{model}*:\n"
                    report += f"- Strengths: {', '.join(strengths)}\n"
                    
                    weaknesses = trends['model_weaknesses'].get(model, [])
                    if weaknesses:
                        report += f"- Weaknesses: {', '.join(weaknesses)}\n"
            
            # Failure patterns
            failures = self.identify_failure_patterns(run_id)
            if failures.get('problematic_questions'):
                report += f"\n**Problematic Questions**: {', '.join(failures['problematic_questions'])}\n"
        
        # Add comparison if multiple runs
        if len(self.run_ids) > 1:
            report += "\n\n## Cross-Run Comparison\n"
            comparison_df = self.compare_runs()
            if not comparison_df.empty:
                report += "\n```\n"
                report += comparison_df.to_string()
                report += "\n```\n"
        
        # Save report
        if not output_path:
            output_path = f"results/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        log_step(f"Detailed report saved to: {output_path}")
        
        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced analysis of Phase 3 test results"
    )
    parser.add_argument(
        "--run_ids",
        nargs='+',
        required=True,
        help="One or more run IDs to analyze"
    )
    parser.add_argument(
        "--compare",
        action='store_true',
        help="Compare multiple runs"
    )
    parser.add_argument(
        "--output",
        help="Output path for report"
    )
    
    args = parser.parse_args()
    
    analyzer = TestResultsAnalyzer(args.run_ids)
    
    if args.compare and len(args.run_ids) > 1:
        log_step("Comparing runs...")
        comparison = analyzer.compare_runs()
        print("\nRun Comparison:")
        print(comparison)
    
    analyzer.generate_detailed_report(args.output)


if __name__ == "__main__":
    main()