#!/usr/bin/env python3
"""
Dashboard Generator
Creates interactive HTML dashboards from agent analysis results
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from jinja2 import Template

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class DashboardGenerator:
    """Generates interactive HTML dashboards from agent results"""
    
    def __init__(self):
        """Initialize dashboard generator"""
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "results"
        self.dashboards_dir = self.results_dir / "dashboards"
        self.dashboards_dir.mkdir(exist_ok=True, parents=True)
        
        # Dashboard template
        self.dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Michigan Guardianship AI - {{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #1e3a8a;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .metric-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #1e3a8a;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .insights {
            background: #e0f2fe;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }
        .insights h3 {
            color: #1e3a8a;
            margin-top: 0;
        }
        .recommendations {
            background: #fef3c7;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .status-good { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-error { color: #ef4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <div class="container">
        <!-- Metric Cards -->
        <div class="metric-cards">
            {% for metric in metrics %}
            <div class="metric-card">
                <div class="metric-value {{ metric.status_class }}">{{ metric.value }}</div>
                <div class="metric-label">{{ metric.label }}</div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Charts -->
        {% for chart in charts %}
        <div class="chart-container">
            <h3>{{ chart.title }}</h3>
            <div id="{{ chart.id }}"></div>
        </div>
        {% endfor %}
        
        <!-- Insights -->
        {% if insights %}
        <div class="insights">
            <h3>Key Insights</h3>
            <ul>
                {% for insight in insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <!-- Recommendations -->
        {% if recommendations %}
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ol>
                {% for rec in recommendations %}
                <li><strong>{{ rec.priority|upper }}:</strong> {{ rec.action }}</li>
                {% endfor %}
            </ol>
        </div>
        {% endif %}
    </div>
    
    <div class="footer">
        <p>Michigan Guardianship AI - Agent Analysis Dashboard</p>
    </div>
    
    <script>
        {% for chart in charts %}
        {{ chart.script|safe }}
        {% endfor %}
    </script>
</body>
</html>
"""
    
    def generate_test_results_dashboard(self, run_id: str) -> str:
        """Generate dashboard for test results"""
        log_step(f"Generating test results dashboard for {run_id}")
        
        # Load run data
        run_dir = self.results_dir / run_id
        if not run_dir.exists():
            log_step(f"Run directory not found: {run_dir}", level="error")
            return None
        
        # Load evaluation metrics
        metrics_path = run_dir / "evaluation_metrics.json"
        if not metrics_path.exists():
            log_step("Evaluation metrics not found", level="error")
            return None
        
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
        
        # Prepare dashboard data
        dashboard_data = {
            "title": f"Test Results Analysis - {run_id}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self._extract_key_metrics(eval_metrics),
            "charts": self._create_test_charts(eval_metrics),
            "insights": self._extract_insights(eval_metrics),
            "recommendations": self._extract_recommendations(eval_metrics)
        }
        
        # Generate HTML
        template = Template(self.dashboard_template)
        html_content = template.render(**dashboard_data)
        
        # Save dashboard
        dashboard_path = self.dashboards_dir / f"test_results_{run_id}.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        log_step(f"Dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def _extract_key_metrics(self, eval_metrics: List[Dict]) -> List[Dict]:
        """Extract key metrics for display"""
        if not eval_metrics:
            return []
        
        # Calculate overall metrics
        avg_score = sum(m["average_scores"]["total_score"] for m in eval_metrics) / len(eval_metrics)
        best_model = max(eval_metrics, key=lambda x: x["average_scores"]["total_score"])
        worst_model = min(eval_metrics, key=lambda x: x["average_scores"]["total_score"])
        
        metrics = [
            {
                "value": len(eval_metrics),
                "label": "Models Tested",
                "status_class": "status-good"
            },
            {
                "value": f"{avg_score:.1f}/10",
                "label": "Average Score",
                "status_class": "status-good" if avg_score >= 7 else "status-warning"
            },
            {
                "value": best_model["model"].split("/")[-1][:15],
                "label": "Best Model",
                "status_class": "status-good"
            },
            {
                "value": f"{best_model['average_scores']['total_score']:.1f}",
                "label": "Best Score",
                "status_class": "status-good"
            }
        ]
        
        return metrics
    
    def _create_test_charts(self, eval_metrics: List[Dict]) -> List[Dict]:
        """Create charts for test results"""
        charts = []
        
        # 1. Model Performance Comparison
        models = [m["model"].split("/")[-1][:20] for m in eval_metrics]
        scores = [m["average_scores"]["total_score"] for m in eval_metrics]
        
        fig1 = go.Figure(data=[
            go.Bar(x=models, y=scores, marker_color='#1e3a8a')
        ])
        fig1.update_layout(
            xaxis_title="Model",
            yaxis_title="Total Score",
            showlegend=False,
            height=400
        )
        
        charts.append({
            "id": "model_comparison",
            "title": "Model Performance Comparison",
            "script": f"Plotly.newPlot('model_comparison', {fig1.to_json()});"
        })
        
        # 2. Performance by Complexity
        complexity_data = []
        for tier in ["simple", "standard", "complex", "crisis"]:
            tier_scores = []
            for m in eval_metrics[:5]:  # Top 5 models
                tier_key = f"{tier}_tier"
                if tier_key in m["average_scores"]:
                    tier_scores.append(m["average_scores"][tier_key]["total_score"])
            
            if tier_scores:
                complexity_data.append({
                    "tier": tier.title(),
                    "avg_score": sum(tier_scores) / len(tier_scores)
                })
        
        if complexity_data:
            df_complexity = pd.DataFrame(complexity_data)
            fig2 = px.line(df_complexity, x='tier', y='avg_score', 
                          markers=True, line_shape='linear')
            fig2.update_layout(
                xaxis_title="Complexity Tier",
                yaxis_title="Average Score",
                showlegend=False,
                height=400
            )
            
            charts.append({
                "id": "complexity_performance",
                "title": "Performance by Question Complexity",
                "script": f"Plotly.newPlot('complexity_performance', {fig2.to_json()});"
            })
        
        # 3. Score Distribution Heatmap
        dimensions = ["procedural_accuracy", "legal_accuracy", "actionability", 
                     "mode_effectiveness", "strategic_caution", "citation_quality", 
                     "harm_prevention"]
        
        heatmap_data = []
        for m in eval_metrics[:10]:  # Top 10 models
            row = []
            for dim in dimensions:
                if dim in m["average_scores"]:
                    row.append(m["average_scores"][dim])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        fig3 = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[d.replace("_", " ").title() for d in dimensions],
            y=[m["model"].split("/")[-1][:20] for m in eval_metrics[:10]],
            colorscale='Blues'
        ))
        fig3.update_layout(
            xaxis_title="Scoring Dimension",
            yaxis_title="Model",
            height=500
        )
        
        charts.append({
            "id": "score_heatmap",
            "title": "Detailed Score Breakdown",
            "script": f"Plotly.newPlot('score_heatmap', {fig3.to_json()});"
        })
        
        return charts
    
    def _extract_insights(self, eval_metrics: List[Dict]) -> List[str]:
        """Extract key insights from metrics"""
        insights = []
        
        # Best performing model
        best_model = max(eval_metrics, key=lambda x: x["average_scores"]["total_score"])
        insights.append(f"Best performing model: {best_model['model']} with score {best_model['average_scores']['total_score']:.1f}/10")
        
        # Performance degradation
        degradation_rates = []
        for m in eval_metrics:
            if "simple_tier" in m["average_scores"] and "complex_tier" in m["average_scores"]:
                simple = m["average_scores"]["simple_tier"]["total_score"]
                complex_score = m["average_scores"]["complex_tier"]["total_score"]
                if simple > 0:
                    degradation_rates.append((simple - complex_score) / simple * 100)
        
        if degradation_rates:
            avg_degradation = sum(degradation_rates) / len(degradation_rates)
            insights.append(f"Average performance degradation from simple to complex: {avg_degradation:.1f}%")
        
        # Dimension analysis
        dimension_avgs = {}
        for dim in ["procedural_accuracy", "legal_accuracy", "actionability"]:
            scores = [m["average_scores"].get(dim, 0) for m in eval_metrics]
            dimension_avgs[dim] = sum(scores) / len(scores) if scores else 0
        
        weakest_dim = min(dimension_avgs.items(), key=lambda x: x[1])
        insights.append(f"Weakest dimension across models: {weakest_dim[0].replace('_', ' ').title()} ({weakest_dim[1]:.1f})")
        
        return insights
    
    def _extract_recommendations(self, eval_metrics: List[Dict]) -> List[Dict]:
        """Extract recommendations from metrics"""
        recommendations = []
        
        # Check for low scorers
        low_scorers = [m for m in eval_metrics if m["average_scores"]["total_score"] < 6]
        if low_scorers:
            recommendations.append({
                "priority": "high",
                "action": f"Consider removing {len(low_scorers)} models scoring below 6.0"
            })
        
        # Check procedural accuracy
        proc_scores = [m["average_scores"].get("procedural_accuracy", 0) for m in eval_metrics]
        if proc_scores and sum(proc_scores) / len(proc_scores) < 2.0:
            recommendations.append({
                "priority": "high",
                "action": "Improve procedural accuracy through prompt engineering"
            })
        
        # Cost efficiency
        recommendations.append({
            "priority": "medium",
            "action": "Run cost analysis to identify most cost-effective models"
        })
        
        return recommendations[:5]  # Top 5 recommendations
    
    def generate_architecture_dashboard(self, architecture_data: Dict[str, Any]) -> str:
        """Generate dashboard for architecture analysis"""
        log_step("Generating architecture dashboard")
        
        # Prepare dashboard data
        dashboard_data = {
            "title": "System Architecture Analysis",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self._extract_architecture_metrics(architecture_data),
            "charts": self._create_architecture_charts(architecture_data),
            "insights": self._extract_architecture_insights(architecture_data),
            "recommendations": []
        }
        
        # Generate HTML
        template = Template(self.dashboard_template)
        html_content = template.render(**dashboard_data)
        
        # Save dashboard
        dashboard_path = self.dashboards_dir / f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        log_step(f"Dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def _extract_architecture_metrics(self, data: Dict[str, Any]) -> List[Dict]:
        """Extract architecture metrics"""
        metrics = []
        
        if "components" in data:
            metrics.append({
                "value": len(data["components"]),
                "label": "Total Components",
                "status_class": "status-good"
            })
        
        if "metrics" in data and "health_score" in data["metrics"]:
            health = data["metrics"]["health_score"]["overall"]
            metrics.append({
                "value": f"{health:.2f}",
                "label": "Health Score",
                "status_class": "status-good" if health > 0.7 else "status-warning"
            })
        
        return metrics
    
    def _create_architecture_charts(self, data: Dict[str, Any]) -> List[Dict]:
        """Create architecture charts"""
        charts = []
        
        # Component coupling chart
        if "metrics" in data and "coupling" in data["metrics"]:
            coupling_data = data["metrics"]["coupling"]
            
            components = list(coupling_data.keys())
            internal_deps = [coupling_data[c]["internal_dependencies"] for c in components]
            external_deps = [coupling_data[c]["external_dependencies"] for c in components]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Internal', x=components, y=internal_deps))
            fig.add_trace(go.Bar(name='External', x=components, y=external_deps))
            
            fig.update_layout(
                barmode='stack',
                xaxis_title="Component",
                yaxis_title="Dependencies",
                height=400
            )
            
            charts.append({
                "id": "coupling_chart",
                "title": "Component Dependencies",
                "script": f"Plotly.newPlot('coupling_chart', {fig.to_json()});"
            })
        
        return charts
    
    def _extract_architecture_insights(self, data: Dict[str, Any]) -> List[str]:
        """Extract architecture insights"""
        insights = []
        
        if "metrics" in data and "health_score" in data["metrics"]:
            health = data["metrics"]["health_score"]
            insights.append(f"Overall system health: {health['overall']:.2f}/1.0")
            insights.append(f"Documentation coverage: {health['documentation_health']:.0%}")
        
        return insights
    
    def generate_unified_dashboard(self, pipeline_results: Dict[str, Any]) -> str:
        """Generate unified dashboard from pipeline results"""
        log_step("Generating unified dashboard from pipeline results")
        
        # Extract data from all agents
        all_metrics = []
        all_insights = []
        all_recommendations = []
        
        for agent_name, agent_results in pipeline_results.get("results", {}).items():
            if "error" not in agent_results:
                # Extract agent-specific data
                if agent_name == "test_analyzer" and "insights" in agent_results:
                    all_insights.extend(agent_results["insights"].get("key_findings", []))
                
                if "recommendations" in agent_results:
                    all_recommendations.extend(agent_results["recommendations"])
        
        # Add summary metrics
        summary = pipeline_results.get("summary", {})
        all_metrics.extend([
            {
                "value": summary.get("total_agents_run", 0),
                "label": "Agents Run",
                "status_class": "status-good"
            },
            {
                "value": summary.get("successful_agents", 0),
                "label": "Successful",
                "status_class": "status-good" if summary.get("successful_agents", 0) == summary.get("total_agents_run", 0) else "status-warning"
            },
            {
                "value": summary.get("total_recommendations", 0),
                "label": "Recommendations",
                "status_class": "status-good"
            },
            {
                "value": summary.get("high_priority_count", 0),
                "label": "High Priority",
                "status_class": "status-warning" if summary.get("high_priority_count", 0) > 5 else "status-good"
            }
        ])
        
        # Prepare dashboard data
        dashboard_data = {
            "title": f"Agent Pipeline Analysis - {pipeline_results.get('pipeline', 'Custom')}",
            "timestamp": pipeline_results.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "metrics": all_metrics,
            "charts": [],  # Could add charts based on pipeline data
            "insights": all_insights[:10],  # Top 10 insights
            "recommendations": sorted(all_recommendations, key=lambda x: 0 if x.get('priority') == 'high' else 1)[:10]
        }
        
        # Generate HTML
        template = Template(self.dashboard_template)
        html_content = template.render(**dashboard_data)
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = self.dashboards_dir / f"pipeline_{pipeline_results.get('pipeline', 'custom')}_{timestamp}.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        log_step(f"Dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def create_index_page(self) -> str:
        """Create an index page listing all dashboards"""
        log_step("Creating dashboard index page")
        
        # Find all dashboard files
        dashboards = []
        for dashboard_file in self.dashboards_dir.glob("*.html"):
            if dashboard_file.name != "index.html":
                dashboards.append({
                    "name": dashboard_file.stem.replace("_", " ").title(),
                    "file": dashboard_file.name,
                    "created": datetime.fromtimestamp(dashboard_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
        
        # Sort by creation date
        dashboards.sort(key=lambda x: x["created"], reverse=True)
        
        # Create index HTML
        index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Michigan Guardianship AI - Dashboard Index</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #1e3a8a;
            color: white;
            padding: 30px;
            text-align: center;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 30px;
        }
        .dashboard-list {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .dashboard-item {
            padding: 20px;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }
        .dashboard-item:hover {
            background: #f9fafb;
        }
        .dashboard-item:last-child {
            border-bottom: none;
        }
        .dashboard-name {
            font-size: 1.1em;
            color: #1e3a8a;
            text-decoration: none;
            font-weight: 500;
        }
        .dashboard-date {
            color: #6b7280;
            font-size: 0.9em;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #6b7280;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Michigan Guardianship AI Dashboards</h1>
        <p>Analysis and Visualization Portal</p>
    </div>
    
    <div class="container">
        <div class="dashboard-list">
            {% if dashboards %}
                {% for dashboard in dashboards %}
                <div class="dashboard-item">
                    <a href="{{ dashboard.file }}" class="dashboard-name">{{ dashboard.name }}</a>
                    <span class="dashboard-date">{{ dashboard.created }}</span>
                </div>
                {% endfor %}
            {% else %}
                <div class="empty-state">
                    <p>No dashboards generated yet.</p>
                    <p>Run agent analyses to generate dashboards.</p>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""
        
        template = Template(index_template)
        html_content = template.render(dashboards=dashboards)
        
        index_path = self.dashboards_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        log_step(f"Index page created at: {index_path}")
        return str(index_path)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate interactive dashboards from agent results"
    )
    parser.add_argument(
        "--test-results",
        help="Generate dashboard for test results (provide run ID)"
    )
    parser.add_argument(
        "--pipeline",
        help="Generate dashboard from pipeline results JSON file"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Create index page for all dashboards"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in browser after generation"
    )
    
    args = parser.parse_args()
    
    generator = DashboardGenerator()
    
    dashboard_path = None
    
    if args.test_results:
        dashboard_path = generator.generate_test_results_dashboard(args.test_results)
    
    elif args.pipeline:
        # Load pipeline results
        pipeline_path = Path(args.pipeline)
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                pipeline_results = json.load(f)
            dashboard_path = generator.generate_unified_dashboard(pipeline_results)
        else:
            print(f"Pipeline results not found: {args.pipeline}")
    
    elif args.index:
        dashboard_path = generator.create_index_page()
    
    else:
        print("Dashboard Generator - Usage:")
        print("  --test-results RUN_ID : Generate test results dashboard")
        print("  --pipeline FILE       : Generate dashboard from pipeline results")
        print("  --index              : Create index page")
        print("  --open               : Open in browser")
    
    # Open in browser if requested
    if dashboard_path and args.open:
        import webbrowser
        webbrowser.open(f"file://{dashboard_path}")
        print(f"Opening dashboard in browser...")


if __name__ == "__main__":
    main()