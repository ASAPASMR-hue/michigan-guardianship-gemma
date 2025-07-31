#!/usr/bin/env python3
"""
Agent Pipeline Runner
Chains multiple agents together for comprehensive analysis
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step

# Import all agents
from test_results_analyzer import TestResultsAnalyzer
from workflow_optimizer import WorkflowOptimizer
from ai_integration_expert import AIIntegrationExpert
from code_refactorer import CodeRefactorer
from system_architect import SystemArchitect
from product_strategist import ProductStrategist
from analytics_engineer import AnalyticsEngineer


class AgentPipeline:
    """Orchestrates multiple agents for comprehensive analysis"""
    
    def __init__(self):
        """Initialize agent pipeline"""
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "results" / "agent_pipeline"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize agents
        self.agents = {
            "test_analyzer": TestResultsAnalyzer(),
            "workflow_optimizer": WorkflowOptimizer(),
            "ai_expert": AIIntegrationExpert(),
            "code_refactorer": CodeRefactorer(),
            "system_architect": SystemArchitect(),
            "product_strategist": ProductStrategist(),
            "analytics_engineer": AnalyticsEngineer()
        }
        
        # Define standard pipelines
        self.pipelines = {
            "quick_check": ["workflow_optimizer", "test_analyzer"],
            "full_analysis": ["test_analyzer", "analytics_engineer", "ai_expert"],
            "system_review": ["system_architect", "code_refactorer", "product_strategist"],
            "optimization": ["ai_expert", "workflow_optimizer"],
            "comprehensive": list(self.agents.keys())
        }
    
    def run_pipeline(
        self,
        pipeline_name: str,
        run_id: Optional[str] = None,
        custom_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run a predefined or custom agent pipeline"""
        log_step(f"Starting agent pipeline: {pipeline_name}")
        
        # Determine which agents to run
        if custom_agents:
            agent_sequence = custom_agents
        else:
            agent_sequence = self.pipelines.get(pipeline_name, [])
        
        if not agent_sequence:
            log_step(f"Unknown pipeline: {pipeline_name}", level="error")
            return {"error": f"Pipeline '{pipeline_name}' not found"}
        
        # Track results
        pipeline_results = {
            "pipeline": pipeline_name,
            "timestamp": datetime.now().isoformat(),
            "agents_run": agent_sequence,
            "results": {},
            "summary": {},
            "recommendations": []
        }
        
        # Run each agent in sequence
        for agent_name in agent_sequence:
            if agent_name not in self.agents:
                log_step(f"Unknown agent: {agent_name}", level="warning")
                continue
            
            log_step(f"Running {agent_name}...")
            
            try:
                agent_result = self._run_single_agent(agent_name, run_id, pipeline_results)
                pipeline_results["results"][agent_name] = agent_result
                
                # Extract key insights
                if "recommendations" in agent_result:
                    pipeline_results["recommendations"].extend(agent_result["recommendations"])
                
            except Exception as e:
                log_step(f"Error running {agent_name}: {e}", level="error")
                pipeline_results["results"][agent_name] = {"error": str(e)}
        
        # Generate summary
        pipeline_results["summary"] = self._generate_summary(pipeline_results)
        
        # Save results
        self._save_results(pipeline_results)
        
        return pipeline_results
    
    def _run_single_agent(
        self,
        agent_name: str,
        run_id: Optional[str],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single agent and return results"""
        agent = self.agents[agent_name]
        
        if agent_name == "test_analyzer" and run_id:
            # Test Results Analyzer
            return {
                "analysis": agent.analyze_run(run_id),
                "insights": agent.get_qualitative_insights(run_id),
                "recommendations": agent.generate_recommendations(run_id)
            }
        
        elif agent_name == "workflow_optimizer":
            # Workflow Optimizer
            preflight = agent.run_preflight_check()
            changed_files = agent.detect_changed_files()
            suggestions = agent.suggest_minimal_tests(changed_files)
            
            return {
                "preflight": preflight,
                "changed_files": list(changed_files),
                "test_suggestions": suggestions,
                "recommendations": [
                    {"type": "test", "action": cmd} for cmd in suggestions.get("commands", [])
                ]
            }
        
        elif agent_name == "ai_expert" and run_id:
            # AI Integration Expert
            failed_cases = agent.analyze_failed_cases(run_id)
            optimizations = agent.optimize_for_failed_cases() if failed_cases else {}
            
            return {
                "failed_cases": len(failed_cases),
                "optimization_results": optimizations,
                "recommendations": self._extract_ai_recommendations(optimizations)
            }
        
        elif agent_name == "code_refactorer":
            # Code Refactorer
            analysis = agent.analyze_project()
            suggestions = agent.suggest_refactorings(analysis)
            
            return {
                "code_metrics": analysis["summary"],
                "refactoring_suggestions": suggestions,
                "recommendations": [
                    {"type": "refactor", "priority": s["priority"], "action": s["suggestion"]}
                    for s in suggestions[:5]
                ]
            }
        
        elif agent_name == "system_architect":
            # System Architect
            analysis = agent.analyze_architecture()
            suggestions = agent.suggest_improvements(analysis)
            
            return {
                "architecture_metrics": analysis["metrics"],
                "component_count": len(analysis["components"]),
                "improvement_suggestions": suggestions,
                "recommendations": [
                    {"type": "architecture", "priority": s["priority"], "action": s["suggestion"]}
                    for s in suggestions[:5]
                ]
            }
        
        elif agent_name == "product_strategist":
            # Product Strategist
            coverage = agent.analyze_question_coverage()
            opportunities = agent.identify_feature_opportunities()
            
            return {
                "question_coverage": coverage,
                "feature_opportunities": opportunities[:5],
                "recommendations": [
                    {"type": "feature", "priority": "high", "action": f"Implement {f['name']}"}
                    for f in opportunities[:3]
                ]
            }
        
        elif agent_name == "analytics_engineer" and run_id:
            # Analytics Engineer
            run_data = agent.load_run_data(run_id)
            patterns = agent.analyze_performance_patterns(run_data) if run_data else {}
            correlations = agent.perform_correlation_analysis(run_data) if run_data else {}
            
            return {
                "performance_patterns": patterns,
                "correlations": correlations,
                "recommendations": self._extract_analytics_recommendations(patterns)
            }
        
        return {"status": "skipped", "reason": "Missing required parameters"}
    
    def _extract_ai_recommendations(self, optimizations: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract recommendations from AI optimization results"""
        recommendations = []
        
        for issue, results in optimizations.items():
            if "best_variation" in results:
                recommendations.append({
                    "type": "prompt",
                    "priority": "high",
                    "action": f"Apply {results['best_variation']['variation_name']} for {issue}"
                })
        
        return recommendations
    
    def _extract_analytics_recommendations(self, patterns: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract recommendations from analytics patterns"""
        recommendations = []
        
        # Check latency issues
        if "latency_analysis" in patterns:
            for model, stats in patterns["latency_analysis"].items():
                if stats.get("p95", 0) > 10:  # 10 second threshold
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "action": f"Optimize {model} - P95 latency is {stats['p95']:.1f}s"
                    })
        
        # Check cost efficiency
        if "cost_efficiency" in patterns:
            for model, metrics in patterns["cost_efficiency"].items():
                if metrics.get("value_score", 0) < 10:
                    recommendations.append({
                        "type": "cost",
                        "priority": "medium",
                        "action": f"Consider replacing {model} - low value score"
                    })
        
        return recommendations
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from pipeline results"""
        summary = {
            "total_agents_run": len(results["agents_run"]),
            "successful_agents": sum(1 for r in results["results"].values() if "error" not in r),
            "total_recommendations": len(results["recommendations"]),
            "high_priority_count": sum(1 for r in results["recommendations"] if r.get("priority") == "high"),
            "key_findings": []
        }
        
        # Extract key findings
        for agent_name, agent_result in results["results"].items():
            if "error" in agent_result:
                continue
            
            if agent_name == "test_analyzer" and "insights" in agent_result:
                summary["key_findings"].append(
                    f"Test Analysis: {agent_result['insights'].get('summary', 'Analysis complete')}"
                )
            
            elif agent_name == "code_refactorer" and "code_metrics" in agent_result:
                metrics = agent_result["code_metrics"]
                summary["key_findings"].append(
                    f"Code Quality: {metrics.get('total_issues', 0)} issues found across {metrics.get('total_files', 0)} files"
                )
            
            elif agent_name == "system_architect" and "architecture_metrics" in agent_result:
                health = agent_result["architecture_metrics"].get("health_score", {})
                summary["key_findings"].append(
                    f"Architecture Health: {health.get('overall', 0):.2f}/1.0"
                )
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save pipeline results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results['pipeline']}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log_step(f"Results saved to: {filepath}")
    
    def generate_pipeline_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report from pipeline results"""
        report = f"""# Agent Pipeline Report

**Pipeline**: {results['pipeline']}
**Timestamp**: {results['timestamp']}
**Agents Run**: {', '.join(results['agents_run'])}

## Executive Summary

- **Total Agents**: {results['summary']['total_agents_run']}
- **Successful**: {results['summary']['successful_agents']}
- **Total Recommendations**: {results['summary']['total_recommendations']}
- **High Priority Items**: {results['summary']['high_priority_count']}

### Key Findings

"""
        
        for finding in results['summary']['key_findings']:
            report += f"- {finding}\n"
        
        report += "\n## Detailed Results\n\n"
        
        # Add results from each agent
        for agent_name, agent_result in results['results'].items():
            report += f"### {agent_name.replace('_', ' ').title()}\n\n"
            
            if "error" in agent_result:
                report += f"**Error**: {agent_result['error']}\n\n"
                continue
            
            # Add agent-specific results
            if agent_name == "workflow_optimizer":
                report += "**Preflight Status**: "
                report += agent_result.get("preflight", {}).get("status", "unknown") + "\n\n"
                
                if agent_result.get("changed_files"):
                    report += "**Changed Files**:\n"
                    for file in agent_result["changed_files"][:5]:
                        report += f"- {file}\n"
                    report += "\n"
            
            elif agent_name == "code_refactorer" and "code_metrics" in agent_result:
                metrics = agent_result["code_metrics"]
                report += f"**Code Metrics**:\n"
                report += f"- Lines of Code: {metrics.get('total_lines', 0):,}\n"
                report += f"- Functions: {metrics.get('total_functions', 0)}\n"
                report += f"- Avg Complexity: {metrics.get('avg_complexity', 0):.2f}\n\n"
            
            elif agent_name == "system_architect" and "architecture_metrics" in agent_result:
                report += f"**Components**: {agent_result.get('component_count', 0)}\n\n"
            
            elif agent_name == "product_strategist" and "feature_opportunities" in agent_result:
                report += "**Top Feature Opportunities**:\n"
                for feature in agent_result["feature_opportunities"][:3]:
                    report += f"- {feature['name']}: {feature['description']}\n"
                report += "\n"
        
        report += "## Consolidated Recommendations\n\n"
        
        # Group recommendations by type
        by_type = {}
        for rec in results['recommendations']:
            rec_type = rec.get('type', 'other')
            if rec_type not in by_type:
                by_type[rec_type] = []
            by_type[rec_type].append(rec)
        
        for rec_type, recs in by_type.items():
            report += f"### {rec_type.title()}\n\n"
            for i, rec in enumerate(recs[:5], 1):
                priority = rec.get('priority', 'medium')
                report += f"{i}. [{priority.upper()}] {rec['action']}\n"
            report += "\n"
        
        report += """## Next Steps

1. Review high-priority recommendations
2. Create action items for immediate fixes
3. Schedule architectural improvements
4. Plan feature development based on opportunities

## Notes

This report was generated automatically by chaining multiple analysis agents. 
For detailed analysis of any specific area, run individual agents with focused parameters.
"""
        
        return report
    
    def list_pipelines(self):
        """List available pipelines"""
        print("Available Agent Pipelines:\n")
        for name, agents in self.pipelines.items():
            print(f"{name}:")
            print(f"  Agents: {', '.join(agents)}")
            print(f"  Description: {self._get_pipeline_description(name)}\n")
    
    def _get_pipeline_description(self, pipeline_name: str) -> str:
        """Get description for a pipeline"""
        descriptions = {
            "quick_check": "Fast validation of changes and test recommendations",
            "full_analysis": "Comprehensive test results and performance analysis",
            "system_review": "Architecture, code quality, and product strategy review",
            "optimization": "AI prompt and workflow optimization",
            "comprehensive": "Run all available agents for complete analysis"
        }
        return descriptions.get(pipeline_name, "Custom pipeline")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run agent analysis pipelines"
    )
    parser.add_argument(
        "--pipeline",
        choices=["quick_check", "full_analysis", "system_review", "optimization", "comprehensive"],
        help="Predefined pipeline to run"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Custom list of agents to run"
    )
    parser.add_argument(
        "--run-id",
        help="Run ID for test analysis (required for some agents)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pipelines"
    )
    parser.add_argument(
        "--report",
        help="Generate report from saved results file"
    )
    
    args = parser.parse_args()
    
    pipeline_runner = AgentPipeline()
    
    if args.list:
        pipeline_runner.list_pipelines()
    
    elif args.report:
        # Load and report on saved results
        report_path = Path(args.report)
        if report_path.exists():
            with open(report_path, 'r') as f:
                results = json.load(f)
            
            report = pipeline_runner.generate_pipeline_report(results)
            print(report)
            
            # Save markdown report
            md_path = report_path.with_suffix('.md')
            with open(md_path, 'w') as f:
                f.write(report)
            print(f"\nMarkdown report saved to: {md_path}")
        else:
            print(f"Results file not found: {args.report}")
    
    elif args.pipeline or args.agents:
        # Run pipeline
        if args.agents:
            results = pipeline_runner.run_pipeline(
                "custom",
                run_id=args.run_id,
                custom_agents=args.agents
            )
        else:
            results = pipeline_runner.run_pipeline(
                args.pipeline,
                run_id=args.run_id
            )
        
        # Generate and display report
        report = pipeline_runner.generate_pipeline_report(results)
        print(report)
        
        # Save markdown report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = pipeline_runner.results_dir / f"report_{args.pipeline or 'custom'}_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    else:
        print("Agent Pipeline Runner\n")
        print("Usage:")
        print("  --list              : Show available pipelines")
        print("  --pipeline NAME     : Run a predefined pipeline")
        print("  --agents A1 A2 ...  : Run specific agents")
        print("  --run-id ID         : Specify test run ID")
        print("  --report FILE       : Generate report from results")
        print("\nExample:")
        print("  python run_agent_pipeline.py --pipeline quick_check")
        print("  python run_agent_pipeline.py --agents test_analyzer analytics_engineer --run-id run_20250128_1430")


if __name__ == "__main__":
    main()