#!/usr/bin/env python3
"""
System Architect Agent
Analyzes system architecture, component coupling, and structural improvements
"""

import os
import sys
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict
import graphviz
import ast

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class SystemArchitect:
    """Analyzes system architecture and component relationships"""
    
    def __init__(self):
        """Initialize system architect"""
        self.project_root = Path(__file__).parent.parent.parent
        self.components = {}
        self.dependencies = defaultdict(set)
        self.interfaces = {}
        self.data_flows = []
        
    def analyze_architecture(self) -> Dict[str, Any]:
        """Analyze the overall system architecture"""
        log_step("Analyzing system architecture...")
        
        # Identify components
        self.components = self._identify_components()
        
        # Analyze dependencies
        self._analyze_dependencies()
        
        # Map data flows
        self.data_flows = self._map_data_flows()
        
        # Analyze interfaces
        self.interfaces = self._analyze_interfaces()
        
        # Calculate metrics
        metrics = self._calculate_architecture_metrics()
        
        return {
            "components": self.components,
            "dependencies": dict(self.dependencies),
            "data_flows": self.data_flows,
            "interfaces": self.interfaces,
            "metrics": metrics
        }
    
    def _identify_components(self) -> Dict[str, Dict[str, Any]]:
        """Identify major system components"""
        components = {}
        
        # Core pipeline components
        pipeline_scripts = [
            ("Document Processor", "scripts/embed_kb.py", "Processes and embeds knowledge base documents"),
            ("Retrieval System", "scripts/adaptive_retrieval.py", "Adaptive document retrieval with complexity handling"),
            ("Response Generator", "scripts/run_phase3_tests.py", "Generates responses using multiple LLMs"),
            ("Validator", "scripts/validator_setup.py", "Validates responses for hallucination and accuracy"),
            ("Evaluator", "scripts/eval_rubric.py", "Evaluates responses using quality rubric")
        ]
        
        for name, path, description in pipeline_scripts:
            full_path = self.project_root / path
            if full_path.exists():
                components[name] = {
                    "path": path,
                    "description": description,
                    "type": "core",
                    "exists": True,
                    "lines_of_code": len(full_path.read_text().splitlines())
                }
        
        # Supporting components
        support_dirs = [
            ("Integration Tests", "integration_tests", "test"),
            ("Configuration", "config", "config"),
            ("Agents", "scripts/agents", "agent"),
            ("Knowledge Base", "kb_files", "data"),
            ("Utilities", "scripts", "utility")
        ]
        
        for name, path, comp_type in support_dirs:
            full_path = self.project_root / path
            if full_path.exists():
                components[name] = {
                    "path": path,
                    "description": f"{name} components",
                    "type": comp_type,
                    "exists": True,
                    "file_count": len(list(full_path.rglob("*.py"))) if full_path.is_dir() else 1
                }
        
        return components
    
    def _analyze_dependencies(self):
        """Analyze dependencies between components"""
        # Map imports for each Python file
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._add_dependency(py_file, alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self._add_dependency(py_file, node.module)
            except:
                continue
    
    def _add_dependency(self, file_path: Path, module_name: str):
        """Add a dependency relationship"""
        # Map file to component
        rel_path = str(file_path.relative_to(self.project_root))
        component = self._file_to_component(rel_path)
        
        if component and not module_name.startswith("__"):
            # Categorize dependency
            if module_name in ["chromadb", "transformers", "torch", "openai"]:
                dep_type = "external_ml"
            elif module_name in ["pandas", "numpy", "yaml", "json"]:
                dep_type = "external_data"
            elif "scripts." in module_name or module_name.startswith("."):
                dep_type = "internal"
            else:
                dep_type = "external_other"
            
            self.dependencies[component].add((module_name, dep_type))
    
    def _file_to_component(self, file_path: str) -> str:
        """Map a file path to its component"""
        if "embed_kb" in file_path:
            return "Document Processor"
        elif "adaptive_retrieval" in file_path:
            return "Retrieval System"
        elif "run_phase3" in file_path:
            return "Response Generator"
        elif "validator" in file_path:
            return "Validator"
        elif "integration_tests" in file_path:
            return "Integration Tests"
        elif "agents" in file_path:
            return "Agents"
        elif "config" in file_path:
            return "Configuration"
        return None
    
    def _map_data_flows(self) -> List[Dict[str, Any]]:
        """Map data flows through the system"""
        flows = [
            {
                "name": "Document Ingestion Flow",
                "steps": [
                    ("KB Files", "Document Processor", "raw documents"),
                    ("Document Processor", "ChromaDB", "embeddings + metadata"),
                    ("ChromaDB", "Retrieval System", "vector search results")
                ]
            },
            {
                "name": "Query Processing Flow",
                "steps": [
                    ("User Query", "Retrieval System", "question"),
                    ("Retrieval System", "ChromaDB", "search query"),
                    ("ChromaDB", "Retrieval System", "relevant documents"),
                    ("Retrieval System", "Response Generator", "context + query"),
                    ("Response Generator", "LLM API", "prompt"),
                    ("LLM API", "Response Generator", "response"),
                    ("Response Generator", "Validator", "response for validation"),
                    ("Validator", "User", "validated response")
                ]
            },
            {
                "name": "Evaluation Flow",
                "steps": [
                    ("Test Questions", "Response Generator", "batch queries"),
                    ("Response Generator", "Evaluator", "responses"),
                    ("Evaluator", "Metrics Store", "scores"),
                    ("Metrics Store", "Analysis Agents", "performance data")
                ]
            }
        ]
        return flows
    
    def _analyze_interfaces(self) -> Dict[str, Dict[str, Any]]:
        """Analyze component interfaces"""
        interfaces = {}
        
        # Analyze key interface files
        interface_files = [
            ("scripts/adaptive_retrieval.py", "AdaptiveRetrieval"),
            ("scripts/llm_handler.py", "LLMHandler"),
            ("scripts/validator_setup.py", "Validator")
        ]
        
        for file_path, class_name in interface_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                interface_info = self._extract_interface_info(full_path, class_name)
                if interface_info:
                    interfaces[class_name] = interface_info
        
        return interfaces
    
    def _extract_interface_info(self, file_path: Path, class_name: str) -> Dict[str, Any]:
        """Extract interface information from a class"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                            # Extract method signature
                            args = []
                            for arg in item.args.args[1:]:  # Skip self
                                arg_info = {"name": arg.arg}
                                if arg.annotation:
                                    arg_info["type"] = ast.unparse(arg.annotation)
                                args.append(arg_info)
                            
                            methods.append({
                                "name": item.name,
                                "args": args,
                                "has_docstring": ast.get_docstring(item) is not None
                            })
                    
                    return {
                        "methods": methods,
                        "method_count": len(methods),
                        "documented": sum(1 for m in methods if m["has_docstring"])
                    }
        except:
            pass
        
        return None
    
    def _calculate_architecture_metrics(self) -> Dict[str, Any]:
        """Calculate architecture quality metrics"""
        metrics = {}
        
        # Component coupling
        coupling_scores = {}
        for component, deps in self.dependencies.items():
            internal_deps = sum(1 for _, dep_type in deps if dep_type == "internal")
            external_deps = sum(1 for _, dep_type in deps if dep_type.startswith("external"))
            
            coupling_scores[component] = {
                "internal_dependencies": internal_deps,
                "external_dependencies": external_deps,
                "coupling_ratio": internal_deps / (internal_deps + external_deps) if (internal_deps + external_deps) > 0 else 0
            }
        
        metrics["coupling"] = coupling_scores
        
        # Interface completeness
        interface_scores = {}
        for name, interface in self.interfaces.items():
            doc_ratio = interface["documented"] / interface["method_count"] if interface["method_count"] > 0 else 0
            interface_scores[name] = {
                "documentation_ratio": doc_ratio,
                "method_count": interface["method_count"]
            }
        
        metrics["interfaces"] = interface_scores
        
        # Data flow complexity
        total_steps = sum(len(flow["steps"]) for flow in self.data_flows)
        metrics["data_flow_complexity"] = {
            "total_flows": len(self.data_flows),
            "total_steps": total_steps,
            "average_steps_per_flow": total_steps / len(self.data_flows) if self.data_flows else 0
        }
        
        # Overall health score
        avg_coupling = sum(s["coupling_ratio"] for s in coupling_scores.values()) / len(coupling_scores) if coupling_scores else 0
        avg_doc = sum(s["documentation_ratio"] for s in interface_scores.values()) / len(interface_scores) if interface_scores else 0
        
        metrics["health_score"] = {
            "coupling_health": 1 - avg_coupling,  # Lower coupling is better
            "documentation_health": avg_doc,
            "overall": (1 - avg_coupling + avg_doc) / 2
        }
        
        return metrics
    
    def generate_architecture_diagram(self, output_path: str = "architecture_diagram"):
        """Generate architecture diagram using Graphviz"""
        dot = graphviz.Digraph(comment='Michigan Guardianship AI Architecture')
        dot.attr(rankdir='TB')
        
        # Add components as nodes
        for name, info in self.components.items():
            if info["type"] == "core":
                dot.node(name, name, shape='box', style='filled', fillcolor='lightblue')
            elif info["type"] == "data":
                dot.node(name, name, shape='cylinder', style='filled', fillcolor='lightgreen')
            else:
                dot.node(name, name, shape='box')
        
        # Add data flow edges
        for flow in self.data_flows:
            for i, (source, target, label) in enumerate(flow["steps"]):
                dot.edge(source, target, label=label, fontsize='10')
        
        # Render diagram
        dot.render(output_path, format='png', cleanup=True)
        log_step(f"Architecture diagram saved to {output_path}.png")
    
    def suggest_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest architectural improvements"""
        suggestions = []
        
        # Check coupling
        for component, scores in analysis["metrics"]["coupling"].items():
            if scores["coupling_ratio"] > 0.5:
                suggestions.append({
                    "type": "decouple_component",
                    "priority": "high",
                    "component": component,
                    "reason": f"High internal coupling ratio ({scores['coupling_ratio']:.2f})",
                    "suggestion": "Consider introducing interfaces or dependency injection to reduce coupling"
                })
        
        # Check interface documentation
        for interface, scores in analysis["metrics"]["interfaces"].items():
            if scores["documentation_ratio"] < 0.8:
                suggestions.append({
                    "type": "improve_documentation",
                    "priority": "moderate",
                    "component": interface,
                    "reason": f"Low documentation coverage ({scores['documentation_ratio']:.0%})",
                    "suggestion": "Add docstrings to all public methods"
                })
        
        # Check data flow complexity
        if analysis["metrics"]["data_flow_complexity"]["average_steps_per_flow"] > 6:
            suggestions.append({
                "type": "simplify_data_flow",
                "priority": "moderate",
                "reason": "Complex data flows with many steps",
                "suggestion": "Consider introducing message queues or event-driven architecture for complex flows"
            })
        
        # Check for missing components
        expected_components = ["Monitoring", "Caching", "API Gateway"]
        existing = set(analysis["components"].keys())
        for expected in expected_components:
            if expected not in existing:
                suggestions.append({
                    "type": "add_component",
                    "priority": "low",
                    "component": expected,
                    "reason": f"{expected} component not found",
                    "suggestion": f"Consider adding {expected} for production readiness"
                })
        
        return suggestions
    
    def generate_architecture_report(self) -> str:
        """Generate comprehensive architecture report"""
        analysis = self.analyze_architecture()
        suggestions = self.suggest_improvements(analysis)
        
        report = f"""# System Architecture Report

**Project**: Michigan Guardianship AI
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Component Overview

### Core Components
"""
        
        for name, info in analysis["components"].items():
            if info.get("type") == "core":
                report += f"\n**{name}**\n"
                report += f"- Path: `{info['path']}`\n"
                report += f"- Description: {info['description']}\n"
                if 'lines_of_code' in info:
                    report += f"- Size: {info['lines_of_code']} lines\n"
        
        report += "\n### Supporting Components\n"
        
        for name, info in analysis["components"].items():
            if info.get("type") != "core":
                report += f"- **{name}**: {info['description']}"
                if 'file_count' in info:
                    report += f" ({info['file_count']} files)"
                report += "\n"
        
        report += "\n## Data Flow Analysis\n\n"
        
        for flow in analysis["data_flows"]:
            report += f"### {flow['name']}\n"
            for i, (source, target, data) in enumerate(flow["steps"], 1):
                report += f"{i}. {source} → {target} ({data})\n"
            report += "\n"
        
        report += "## Architecture Metrics\n\n"
        
        # Coupling analysis
        report += "### Component Coupling\n\n"
        report += "| Component | Internal Deps | External Deps | Coupling Ratio |\n"
        report += "|-----------|---------------|---------------|----------------|\n"
        
        for comp, scores in analysis["metrics"]["coupling"].items():
            report += f"| {comp} | {scores['internal_dependencies']} | "
            report += f"{scores['external_dependencies']} | "
            report += f"{scores['coupling_ratio']:.2f} |\n"
        
        # Interface analysis
        report += "\n### Interface Quality\n\n"
        report += "| Interface | Methods | Documented | Coverage |\n"
        report += "|-----------|---------|------------|----------|\n"
        
        for interface, scores in analysis["metrics"]["interfaces"].items():
            report += f"| {interface} | {scores['method_count']} | "
            report += f"{scores['method_count'] * scores['documentation_ratio']:.0f} | "
            report += f"{scores['documentation_ratio']:.0%} |\n"
        
        # Health scores
        health = analysis["metrics"]["health_score"]
        report += f"\n### Overall Health Scores\n\n"
        report += f"- **Coupling Health**: {health['coupling_health']:.2f}/1.0\n"
        report += f"- **Documentation Health**: {health['documentation_health']:.2f}/1.0\n"
        report += f"- **Overall Score**: {health['overall']:.2f}/1.0\n"
        
        report += "\n## Architectural Improvements\n\n"
        
        # Group suggestions by priority
        priority_groups = defaultdict(list)
        for suggestion in suggestions:
            priority_groups[suggestion["priority"]].append(suggestion)
        
        for priority in ["high", "moderate", "low"]:
            if priority in priority_groups:
                report += f"### {priority.title()} Priority\n\n"
                for i, suggestion in enumerate(priority_groups[priority], 1):
                    report += f"{i}. **{suggestion['type'].replace('_', ' ').title()}**\n"
                    if 'component' in suggestion:
                        report += f"   - Component: {suggestion['component']}\n"
                    report += f"   - Reason: {suggestion['reason']}\n"
                    report += f"   - Action: {suggestion['suggestion']}\n\n"
        
        report += """## Key Architectural Patterns

### Current Strengths
1. **Modular Design**: Clear separation between processing, retrieval, and generation
2. **Configuration-Driven**: Centralized config management in YAML
3. **Extensible Agent System**: Easy to add new analysis capabilities

### Areas for Enhancement
1. **Caching Layer**: Add Redis/Memcached for frequent queries
2. **API Gateway**: Standardize external interfaces
3. **Monitoring**: Add OpenTelemetry for production observability
4. **Event Bus**: Consider event-driven architecture for better decoupling

## Next Steps

1. **Immediate**: Address high-priority coupling issues
2. **Short-term**: Improve interface documentation
3. **Long-term**: Implement suggested architectural components

## Appendix

To generate an architecture diagram:
```bash
python scripts/agents/system_architect.py --diagram
```
"""
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="System Architect - Analyze system design"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform architecture analysis"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate architecture report"
    )
    parser.add_argument(
        "--diagram",
        action="store_true",
        help="Generate architecture diagram"
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Show improvement suggestions"
    )
    
    args = parser.parse_args()
    
    architect = SystemArchitect()
    
    if args.diagram:
        try:
            architect.generate_architecture_diagram()
            print("Architecture diagram generated successfully")
        except Exception as e:
            print(f"Error generating diagram: {e}")
            print("Make sure graphviz is installed: pip install graphviz")
    
    elif args.report:
        report = architect.generate_architecture_report()
        print(report)
        
        # Save report
        report_path = Path("results/architecture_report.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    elif args.suggest:
        analysis = architect.analyze_architecture()
        suggestions = architect.suggest_improvements(analysis)
        
        print("Architectural Improvement Suggestions:\n")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion['type']} ({suggestion['priority']} priority)")
            print(f"   Reason: {suggestion['reason']}")
            print(f"   Action: {suggestion['suggestion']}\n")
    
    else:
        # Default: show analysis summary
        analysis = architect.analyze_architecture()
        
        print("System Architecture Summary:")
        print(f"Components: {len(analysis['components'])}")
        print(f"Data flows: {len(analysis['data_flows'])}")
        print(f"Health score: {analysis['metrics']['health_score']['overall']:.2f}/1.0")
        
        print("\nKey components:")
        for name, info in analysis['components'].items():
            if info.get('type') == 'core':
                print(f"  - {name}")


if __name__ == "__main__":
    main()