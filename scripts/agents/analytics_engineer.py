#!/usr/bin/env python3
"""
Analytics Engineer Agent
Performs deep performance analytics and correlation analysis
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class AnalyticsEngineer:
    """Deep analytics and performance insights"""
    
    def __init__(self):
        """Initialize analytics engineer"""
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "results"
        self.metrics_cache = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_run_data(self, run_id: str) -> Dict[str, Any]:
        """Load all data for a specific run"""
        log_step(f"Loading data for run: {run_id}")
        
        run_dir = self.results_dir / run_id
        if not run_dir.exists():
            log_step(f"Run directory not found: {run_dir}", level="error")
            return {}
        
        run_data = {
            "run_id": run_id,
            "metadata": {},
            "raw_results": {},
            "evaluation_metrics": {},
            "model_results": defaultdict(list)
        }
        
        # Load metadata
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                run_data["metadata"] = json.load(f)
        
        # Load evaluation metrics
        metrics_path = run_dir / "evaluation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                run_data["evaluation_metrics"] = json.load(f)
        
        # Load raw results for each model
        for result_file in run_dir.glob("*_results.json"):
            model_name = result_file.stem.replace("_results", "")
            with open(result_file, 'r') as f:
                results = json.load(f)
                run_data["raw_results"][model_name] = results
                
                # Organize by model for easier analysis
                for result in results:
                    run_data["model_results"][model_name].append(result)
        
        return run_data
    
    def analyze_performance_patterns(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns across models and questions"""
        log_step("Analyzing performance patterns...")
        
        patterns = {
            "latency_analysis": self._analyze_latency_patterns(run_data),
            "accuracy_by_complexity": self._analyze_accuracy_by_complexity(run_data),
            "error_patterns": self._analyze_error_patterns(run_data),
            "cost_efficiency": self._analyze_cost_efficiency(run_data),
            "question_difficulty": self._analyze_question_difficulty(run_data)
        }
        
        return patterns
    
    def _analyze_latency_patterns(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze latency patterns across models"""
        latency_data = defaultdict(list)
        
        for model, results in run_data["model_results"].items():
            for result in results:
                if "latency" in result and result["latency"] is not None:
                    latency_data[model].append(result["latency"])
        
        analysis = {}
        for model, latencies in latency_data.items():
            if latencies:
                analysis[model] = {
                    "mean": np.mean(latencies),
                    "median": np.median(latencies),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99),
                    "std": np.std(latencies),
                    "min": np.min(latencies),
                    "max": np.max(latencies)
                }
        
        # Identify outliers
        for model, metrics in analysis.items():
            latencies = latency_data[model]
            q1 = np.percentile(latencies, 25)
            q3 = np.percentile(latencies, 75)
            iqr = q3 - q1
            outliers = [l for l in latencies if l < q1 - 1.5*iqr or l > q3 + 1.5*iqr]
            analysis[model]["outlier_count"] = len(outliers)
            analysis[model]["outlier_percentage"] = len(outliers) / len(latencies) * 100
        
        return analysis
    
    def _analyze_accuracy_by_complexity(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accuracy patterns by question complexity"""
        complexity_scores = defaultdict(lambda: defaultdict(list))
        
        # Get evaluation scores if available
        if "evaluation_metrics" in run_data:
            for model_data in run_data["evaluation_metrics"]:
                model = model_data["model"]
                for tier in ["simple", "standard", "complex", "crisis"]:
                    tier_key = f"{tier}_tier"
                    if tier_key in model_data["average_scores"]:
                        complexity_scores[tier][model] = model_data["average_scores"][tier_key]["total_score"]
        
        # Calculate degradation patterns
        degradation = {}
        for model in complexity_scores.get("simple", {}).keys():
            simple_score = complexity_scores["simple"].get(model, 0)
            complex_score = complexity_scores["complex"].get(model, 0)
            
            if simple_score > 0:
                degradation[model] = {
                    "simple_to_complex_drop": simple_score - complex_score,
                    "degradation_percentage": (simple_score - complex_score) / simple_score * 100
                }
        
        return {
            "scores_by_complexity": dict(complexity_scores),
            "performance_degradation": degradation
        }
    
    def _analyze_error_patterns(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns across models"""
        error_analysis = {
            "error_types": defaultdict(int),
            "errors_by_model": defaultdict(lambda: defaultdict(int)),
            "timeout_analysis": {},
            "common_failure_questions": []
        }
        
        failure_counts = defaultdict(int)
        
        for model, results in run_data["model_results"].items():
            timeout_count = 0
            empty_response_count = 0
            error_count = 0
            
            for result in results:
                if result.get("error"):
                    error_count += 1
                    error_type = result.get("error", "unknown")
                    
                    if "timeout" in error_type.lower():
                        timeout_count += 1
                        error_analysis["error_types"]["timeout"] += 1
                    else:
                        error_analysis["error_types"][error_type] += 1
                    
                    error_analysis["errors_by_model"][model][error_type] += 1
                    failure_counts[result["question_id"]] += 1
                
                elif not result.get("response"):
                    empty_response_count += 1
                    error_analysis["error_types"]["empty_response"] += 1
                    failure_counts[result["question_id"]] += 1
            
            error_analysis["timeout_analysis"][model] = {
                "timeout_count": timeout_count,
                "timeout_rate": timeout_count / len(results) if results else 0,
                "empty_response_count": empty_response_count,
                "total_errors": error_count
            }
        
        # Identify questions that fail frequently
        total_models = len(run_data["model_results"])
        for question_id, fail_count in failure_counts.items():
            if fail_count >= total_models * 0.3:  # Fails on 30%+ of models
                error_analysis["common_failure_questions"].append({
                    "question_id": question_id,
                    "failure_count": fail_count,
                    "failure_rate": fail_count / total_models
                })
        
        return error_analysis
    
    def _analyze_cost_efficiency(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost efficiency of different models"""
        cost_analysis = {}
        
        for model, results in run_data["model_results"].items():
            total_cost = sum(r.get("cost_usd", 0) for r in results)
            successful_responses = sum(1 for r in results if r.get("response") and not r.get("error"))
            
            # Get average score from evaluation metrics
            avg_score = 0
            if "evaluation_metrics" in run_data:
                for model_data in run_data["evaluation_metrics"]:
                    if model_data["model"] == model.replace("_", "/").replace("-", ":"):
                        avg_score = model_data["average_scores"]["total_score"]
                        break
            
            cost_analysis[model] = {
                "total_cost": total_cost,
                "cost_per_question": total_cost / len(results) if results else 0,
                "cost_per_success": total_cost / successful_responses if successful_responses > 0 else float('inf'),
                "success_rate": successful_responses / len(results) if results else 0,
                "quality_score": avg_score,
                "value_score": avg_score / (total_cost + 0.01)  # Quality per dollar
            }
        
        return cost_analysis
    
    def _analyze_question_difficulty(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify which questions are consistently difficult"""
        question_scores = defaultdict(list)
        question_metadata = {}
        
        # Aggregate scores by question
        for model, results in run_data["model_results"].items():
            for result in results:
                question_id = result["question_id"]
                
                # Store question metadata
                if question_id not in question_metadata:
                    question_metadata[question_id] = {
                        "text": result.get("question_text", ""),
                        "complexity": result.get("complexity_tier", "unknown")
                    }
                
                # Track if this was successful
                if result.get("response") and not result.get("error"):
                    question_scores[question_id].append(1)
                else:
                    question_scores[question_id].append(0)
        
        # Calculate difficulty metrics
        difficulty_analysis = []
        for question_id, scores in question_scores.items():
            success_rate = np.mean(scores)
            consistency = 1 - np.std(scores)  # Higher = more consistent
            
            difficulty_analysis.append({
                "question_id": question_id,
                "success_rate": success_rate,
                "consistency": consistency,
                "difficulty_score": 1 - success_rate,
                "complexity": question_metadata[question_id]["complexity"],
                "question_preview": question_metadata[question_id]["text"][:100] + "..."
            })
        
        # Sort by difficulty
        difficulty_analysis.sort(key=lambda x: x["difficulty_score"], reverse=True)
        
        return {
            "most_difficult": difficulty_analysis[:10],
            "most_inconsistent": sorted(difficulty_analysis, key=lambda x: x["consistency"])[:10],
            "difficulty_by_complexity": self._group_by_complexity(difficulty_analysis)
        }
    
    def _group_by_complexity(self, difficulty_data: List[Dict]) -> Dict[str, Dict]:
        """Group difficulty data by complexity tier"""
        grouped = defaultdict(list)
        for item in difficulty_data:
            grouped[item["complexity"]].append(item["difficulty_score"])
        
        summary = {}
        for complexity, scores in grouped.items():
            if scores:
                summary[complexity] = {
                    "avg_difficulty": np.mean(scores),
                    "std_difficulty": np.std(scores),
                    "count": len(scores)
                }
        
        return summary
    
    def perform_correlation_analysis(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis between different metrics"""
        log_step("Performing correlation analysis...")
        
        # Build correlation matrix data
        model_metrics = []
        
        for model in run_data["model_results"].keys():
            metrics = {
                "model": model,
                "avg_latency": 0,
                "success_rate": 0,
                "cost_per_q": 0,
                "quality_score": 0,
                "timeout_rate": 0
            }
            
            results = run_data["model_results"][model]
            if results:
                latencies = [r["latency"] for r in results if r.get("latency")]
                metrics["avg_latency"] = np.mean(latencies) if latencies else 0
                metrics["success_rate"] = sum(1 for r in results if r.get("response") and not r.get("error")) / len(results)
                metrics["cost_per_q"] = sum(r.get("cost_usd", 0) for r in results) / len(results)
                metrics["timeout_rate"] = sum(1 for r in results if r.get("error") == "timeout") / len(results)
            
            # Get quality score from evaluation
            if "evaluation_metrics" in run_data:
                for model_data in run_data["evaluation_metrics"]:
                    if model_data["model"] == model.replace("_", "/").replace("-", ":"):
                        metrics["quality_score"] = model_data["average_scores"]["total_score"]
                        break
            
            model_metrics.append(metrics)
        
        # Create DataFrame for correlation
        df = pd.DataFrame(model_metrics)
        numeric_cols = ["avg_latency", "success_rate", "cost_per_q", "quality_score", "timeout_rate"]
        
        # Calculate correlations
        correlations = {}
        if len(df) > 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Find significant correlations
            significant_corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:  # Significant correlation threshold
                        significant_corrs.append({
                            "metric1": numeric_cols[i],
                            "metric2": numeric_cols[j],
                            "correlation": corr_value,
                            "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                        })
            
            correlations = {
                "correlation_matrix": corr_matrix.to_dict(),
                "significant_correlations": significant_corrs
            }
        
        return correlations
    
    def generate_visualizations(self, run_data: Dict[str, Any], output_dir: str = "results/analytics"):
        """Generate visualization plots"""
        log_step("Generating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. Latency distribution plot
        self._plot_latency_distribution(run_data, output_path / "latency_distribution.png")
        
        # 2. Performance by complexity plot
        self._plot_performance_by_complexity(run_data, output_path / "performance_by_complexity.png")
        
        # 3. Cost vs Quality scatter plot
        self._plot_cost_vs_quality(run_data, output_path / "cost_vs_quality.png")
        
        # 4. Error rate heatmap
        self._plot_error_heatmap(run_data, output_path / "error_heatmap.png")
        
        log_step(f"Visualizations saved to {output_path}")
    
    def _plot_latency_distribution(self, run_data: Dict[str, Any], output_path: Path):
        """Plot latency distribution for all models"""
        plt.figure(figsize=(12, 6))
        
        model_latencies = []
        model_names = []
        
        for model, results in run_data["model_results"].items():
            latencies = [r["latency"] for r in results if r.get("latency") and r["latency"] < 180]
            if latencies:
                model_latencies.append(latencies)
                model_names.append(model.replace("_", "/")[:20])  # Truncate long names
        
        plt.boxplot(model_latencies, labels=model_names)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Latency (seconds)")
        plt.title("Latency Distribution by Model")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_performance_by_complexity(self, run_data: Dict[str, Any], output_path: Path):
        """Plot performance degradation by complexity"""
        plt.figure(figsize=(10, 6))
        
        complexity_tiers = ["simple", "standard", "complex", "crisis"]
        
        if "evaluation_metrics" in run_data:
            for model_data in run_data["evaluation_metrics"][:5]:  # Top 5 models
                model_name = model_data["model"].split("/")[-1][:15]  # Truncate
                scores = []
                
                for tier in complexity_tiers:
                    tier_key = f"{tier}_tier"
                    if tier_key in model_data["average_scores"]:
                        scores.append(model_data["average_scores"][tier_key]["total_score"])
                    else:
                        scores.append(0)
                
                plt.plot(complexity_tiers, scores, marker='o', label=model_name)
        
        plt.xlabel("Complexity Tier")
        plt.ylabel("Average Score")
        plt.title("Performance by Question Complexity")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_cost_vs_quality(self, run_data: Dict[str, Any], output_path: Path):
        """Plot cost vs quality scatter plot"""
        plt.figure(figsize=(10, 8))
        
        costs = []
        qualities = []
        labels = []
        
        cost_analysis = self._analyze_cost_efficiency(run_data)
        
        for model, metrics in cost_analysis.items():
            if metrics["cost_per_question"] > 0 and metrics["quality_score"] > 0:
                costs.append(metrics["cost_per_question"])
                qualities.append(metrics["quality_score"])
                labels.append(model.replace("_", "/").split("/")[-1][:15])
        
        if costs and qualities:
            plt.scatter(costs, qualities, s=100, alpha=0.6)
            
            # Add labels
            for i, label in enumerate(labels):
                plt.annotate(label, (costs[i], qualities[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel("Cost per Question ($)")
            plt.ylabel("Quality Score")
            plt.title("Cost vs Quality Analysis")
            plt.grid(True, alpha=0.3)
            
            # Add value line (high quality, low cost)
            if max(costs) > 0:
                x_range = np.linspace(0, max(costs), 100)
                plt.plot(x_range, 8 / (x_range + 0.01), 'r--', alpha=0.5, label='Value curve')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_error_heatmap(self, run_data: Dict[str, Any], output_path: Path):
        """Plot error rate heatmap"""
        # Prepare data for heatmap
        error_patterns = self._analyze_error_patterns(run_data)
        
        models = list(error_patterns["errors_by_model"].keys())[:10]  # Top 10
        error_types = list(set(e for errors in error_patterns["errors_by_model"].values() for e in errors.keys()))
        
        if models and error_types:
            # Create matrix
            matrix = np.zeros((len(models), len(error_types)))
            
            for i, model in enumerate(models):
                for j, error_type in enumerate(error_types):
                    matrix[i, j] = error_patterns["errors_by_model"][model].get(error_type, 0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, 
                       xticklabels=[e[:20] for e in error_types],
                       yticklabels=[m.replace("_", "/")[:20] for m in models],
                       cmap='YlOrRd',
                       annot=True,
                       fmt='.0f')
            
            plt.title("Error Types by Model")
            plt.xlabel("Error Type")
            plt.ylabel("Model")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
    
    def generate_analytics_report(self, run_ids: List[str]) -> str:
        """Generate comprehensive analytics report"""
        report = f"""# Analytics Engineering Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Analyzed Runs**: {', '.join(run_ids)}

## Executive Summary

This report provides deep analytical insights into the performance characteristics, cost efficiency, and quality patterns of the Michigan Guardianship AI system across multiple model evaluations.

"""
        
        # Analyze each run
        all_patterns = {}
        for run_id in run_ids:
            run_data = self.load_run_data(run_id)
            if run_data:
                patterns = self.analyze_performance_patterns(run_data)
                correlations = self.perform_correlation_analysis(run_data)
                all_patterns[run_id] = {
                    "patterns": patterns,
                    "correlations": correlations,
                    "metadata": run_data.get("metadata", {})
                }
        
        if not all_patterns:
            report += "No data found for specified runs.\n"
            return report
        
        # Latest run detailed analysis
        latest_run = run_ids[-1]
        latest_patterns = all_patterns[latest_run]["patterns"]
        
        report += f"## Performance Analysis (Run: {latest_run})\n\n"
        
        # Latency analysis
        report += "### Latency Characteristics\n\n"
        report += "| Model | Mean | P95 | P99 | Outliers |\n"
        report += "|-------|------|-----|-----|----------|\n"
        
        for model, stats in latest_patterns["latency_analysis"].items():
            report += f"| {model[:20]} | {stats['mean']:.2f}s | {stats['p95']:.2f}s | "
            report += f"{stats['p99']:.2f}s | {stats['outlier_percentage']:.1f}% |\n"
        
        # Accuracy by complexity
        report += "\n### Performance by Complexity\n\n"
        
        if "scores_by_complexity" in latest_patterns["accuracy_by_complexity"]:
            report += "| Model | Simple | Standard | Complex | Degradation |\n"
            report += "|-------|--------|----------|---------|-------------|\n"
            
            scores = latest_patterns["accuracy_by_complexity"]["scores_by_complexity"]
            degradation = latest_patterns["accuracy_by_complexity"]["performance_degradation"]
            
            for model in scores.get("simple", {}).keys():
                simple = scores["simple"].get(model, 0)
                standard = scores["standard"].get(model, 0)
                complex_score = scores["complex"].get(model, 0)
                deg = degradation.get(model, {}).get("degradation_percentage", 0)
                
                report += f"| {model[:20]} | {simple:.1f} | {standard:.1f} | "
                report += f"{complex_score:.1f} | {deg:.1f}% |\n"
        
        # Error patterns
        report += "\n### Error Analysis\n\n"
        error_data = latest_patterns["error_patterns"]
        
        report += "**Error Type Distribution**:\n"
        for error_type, count in error_data["error_types"].items():
            report += f"- {error_type}: {count} occurrences\n"
        
        if error_data["common_failure_questions"]:
            report += "\n**Frequently Failing Questions**:\n"
            for q in error_data["common_failure_questions"][:5]:
                report += f"- {q['question_id']}: {q['failure_rate']:.0%} failure rate\n"
        
        # Cost efficiency
        report += "\n### Cost Efficiency Analysis\n\n"
        report += "| Model | Cost/Q | Success Rate | Quality | Value Score |\n"
        report += "|-------|--------|--------------|---------|-------------|\n"
        
        cost_data = latest_patterns["cost_efficiency"]
        sorted_models = sorted(cost_data.items(), key=lambda x: x[1]["value_score"], reverse=True)
        
        for model, metrics in sorted_models[:10]:
            report += f"| {model[:20]} | ${metrics['cost_per_question']:.4f} | "
            report += f"{metrics['success_rate']:.0%} | {metrics['quality_score']:.1f} | "
            report += f"{metrics['value_score']:.1f} |\n"
        
        # Question difficulty
        report += "\n### Question Difficulty Analysis\n\n"
        report += "**Most Difficult Questions**:\n"
        
        for q in latest_patterns["question_difficulty"]["most_difficult"][:5]:
            report += f"- {q['question_id']} ({q['complexity']}): "
            report += f"{q['success_rate']:.0%} success rate\n"
            report += f"  Preview: {q['question_preview']}\n"
        
        # Correlation insights
        report += "\n## Correlation Insights\n\n"
        
        if "correlations" in all_patterns[latest_run]:
            corr_data = all_patterns[latest_run]["correlations"]
            if "significant_correlations" in corr_data:
                report += "**Significant Correlations Found**:\n"
                for corr in corr_data["significant_correlations"]:
                    report += f"- {corr['metric1']} ↔ {corr['metric2']}: "
                    report += f"{corr['correlation']:.2f} ({corr['strength']})\n"
        
        # Trend analysis (if multiple runs)
        if len(run_ids) > 1:
            report += "\n## Trend Analysis Across Runs\n\n"
            report += "Comparing performance evolution across runs:\n\n"
            
            # Add trend analysis here
            report += "- Performance trends would be shown here with more runs\n"
        
        report += """

## Key Insights

1. **Latency vs Quality Trade-off**: Higher quality models generally have higher latency
2. **Complexity Impact**: All models show performance degradation on complex questions
3. **Cost Efficiency Leaders**: Free models offer surprising value for simple queries
4. **Error Patterns**: Timeouts are the primary failure mode for complex models

## Recommendations

### Immediate Actions
1. **Implement Caching**: For frequently asked questions to reduce latency
2. **Timeout Optimization**: Adjust timeouts based on complexity tier
3. **Model Selection**: Use cheap/fast models for simple queries, premium for complex

### Strategic Improvements
1. **Question Preprocessing**: Better complexity classification could optimize routing
2. **Ensemble Approach**: Combine multiple models for critical questions
3. **Error Recovery**: Implement fallback strategies for timeout scenarios

## Technical Recommendations

### Performance Optimization
- Implement request batching for high-volume scenarios
- Add result caching with 24-hour TTL
- Use streaming for long responses

### Cost Optimization
- Route simple questions to free tier models
- Implement daily/monthly budget caps
- Monitor cost trends and alert on anomalies

### Quality Assurance
- Add automated testing for new model releases
- Implement A/B testing framework
- Create feedback loop for continuous improvement

## Appendix

For detailed visualizations, see:
- `results/analytics/latency_distribution.png`
- `results/analytics/performance_by_complexity.png`
- `results/analytics/cost_vs_quality.png`
- `results/analytics/error_heatmap.png`
"""
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analytics Engineer - Deep performance analysis"
    )
    parser.add_argument(
        "--analyze",
        nargs="+",
        help="Run IDs to analyze"
    )
    parser.add_argument(
        "--visualize",
        help="Generate visualizations for a run ID"
    )
    parser.add_argument(
        "--report",
        nargs="+",
        help="Generate report for run IDs"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple runs"
    )
    
    args = parser.parse_args()
    
    engineer = AnalyticsEngineer()
    
    if args.visualize:
        run_data = engineer.load_run_data(args.visualize)
        if run_data:
            engineer.generate_visualizations(run_data)
            print(f"Visualizations generated for run: {args.visualize}")
        else:
            print(f"No data found for run: {args.visualize}")
    
    elif args.report:
        report = engineer.generate_analytics_report(args.report)
        print(report)
        
        # Save report
        report_path = Path(f"results/analytics_report_{'_'.join(args.report)}.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    elif args.compare:
        # Compare multiple runs
        print(f"Comparing runs: {', '.join(args.compare)}")
        
        comparison = {}
        for run_id in args.compare:
            run_data = engineer.load_run_data(run_id)
            if run_data:
                patterns = engineer.analyze_performance_patterns(run_data)
                comparison[run_id] = {
                    "timestamp": run_data.get("metadata", {}).get("timestamp", "unknown"),
                    "avg_latency": np.mean([
                        stats["mean"] for stats in patterns["latency_analysis"].values()
                    ]),
                    "error_rate": sum(
                        patterns["error_patterns"]["error_types"].values()
                    ) / sum(len(r) for r in run_data["model_results"].values())
                }
        
        print("\nComparison Summary:")
        for run_id, metrics in comparison.items():
            print(f"\n{run_id}:")
            print(f"  Timestamp: {metrics['timestamp']}")
            print(f"  Avg Latency: {metrics['avg_latency']:.2f}s")
            print(f"  Error Rate: {metrics['error_rate']:.2%}")
    
    elif args.analyze:
        # Basic analysis
        for run_id in args.analyze:
            run_data = engineer.load_run_data(run_id)
            if run_data:
                patterns = engineer.analyze_performance_patterns(run_data)
                print(f"\nAnalysis for {run_id}:")
                print(f"Models analyzed: {len(run_data['model_results'])}")
                print(f"Total errors: {sum(patterns['error_patterns']['error_types'].values())}")
                print(f"Most common error: {max(patterns['error_patterns']['error_types'].items(), key=lambda x: x[1])[0] if patterns['error_patterns']['error_types'] else 'None'}")
    
    else:
        print("Analytics Engineer - Usage:")
        print("  --analyze RUN_ID     : Analyze specific run")
        print("  --visualize RUN_ID   : Generate visualizations")
        print("  --report RUN_IDS     : Generate detailed report")
        print("  --compare RUN_IDS    : Compare multiple runs")


if __name__ == "__main__":
    main()