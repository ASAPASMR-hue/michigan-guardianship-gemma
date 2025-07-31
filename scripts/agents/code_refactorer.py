#!/usr/bin/env python3
"""
Code Refactorer Agent
Analyzes code quality and suggests refactoring improvements
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class CodeRefactorer:
    """Analyzes code quality and suggests refactoring improvements"""
    
    def __init__(self):
        """Initialize code refactorer"""
        self.project_root = Path(__file__).parent.parent.parent
        self.code_metrics = {}
        self.issues = []
        self.refactoring_suggestions = []
        
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for quality metrics"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error in {file_path}: {e}"}
        
        metrics = {
            "file": str(file_path.relative_to(self.project_root)),
            "lines_of_code": len(content.splitlines()),
            "functions": [],
            "classes": [],
            "complexity": 0,
            "imports": [],
            "docstring_coverage": 0,
            "issues": []
        }
        
        # Analyze AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_metrics = self._analyze_function(node, content)
                metrics["functions"].append(func_metrics)
                metrics["complexity"] += func_metrics["complexity"]
                
            elif isinstance(node, ast.ClassDef):
                class_metrics = self._analyze_class(node, content)
                metrics["classes"].append(class_metrics)
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["imports"].append(self._get_import_name(node))
        
        # Calculate docstring coverage
        total_entities = len(metrics["functions"]) + len(metrics["classes"])
        if total_entities > 0:
            documented = sum(1 for f in metrics["functions"] if f["has_docstring"])
            documented += sum(1 for c in metrics["classes"] if c["has_docstring"])
            metrics["docstring_coverage"] = documented / total_entities
        
        # Check for common issues
        metrics["issues"] = self._check_common_issues(content, tree)
        
        return metrics
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a function for complexity and quality"""
        return {
            "name": node.name,
            "line": node.lineno,
            "complexity": self._calculate_complexity(node),
            "parameters": len(node.args.args),
            "lines": node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0,
            "has_docstring": ast.get_docstring(node) is not None,
            "is_too_long": (node.end_lineno - node.lineno > 50) if hasattr(node, 'end_lineno') else False,
            "has_type_hints": any(arg.annotation for arg in node.args.args)
        }
    
    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze a class for quality metrics"""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        return {
            "name": node.name,
            "line": node.lineno,
            "methods": len(methods),
            "has_docstring": ast.get_docstring(node) is not None,
            "has_init": any(m.name == "__init__" for m in methods),
            "lines": node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_import_name(self, node: ast.AST) -> str:
        """Get the name of an import"""
        if isinstance(node, ast.Import):
            return node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            return f"{node.module}.{node.names[0].name}" if node.module else node.names[0].name
        return ""
    
    def _check_common_issues(self, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
        """Check for common code quality issues"""
        issues = []
        
        # Check for long lines
        for i, line in enumerate(content.splitlines(), 1):
            if len(line) > 100:
                issues.append({
                    "type": "long_line",
                    "line": i,
                    "severity": "minor",
                    "message": f"Line {i} exceeds 100 characters ({len(line)} chars)"
                })
        
        # Check for hardcoded values
        hardcoded_patterns = [
            (r'\d{3,}', "hardcoded_number"),
            (r'[\'"]\/Users\/[^\'"]+[\'"]', "hardcoded_path"),
            (r'[\'"]http[s]?:\/\/[^\'"]+[\'"]', "hardcoded_url")
        ]
        
        for pattern, issue_type in hardcoded_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    "type": issue_type,
                    "line": line_num,
                    "severity": "moderate",
                    "message": f"Hardcoded value found: {match.group()[:50]}..."
                })
        
        # Check for missing error handling
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_try = any(isinstance(child, ast.Try) for child in ast.walk(node))
                if "handler" in node.name.lower() and not has_try:
                    issues.append({
                        "type": "missing_error_handling",
                        "line": node.lineno,
                        "severity": "moderate",
                        "message": f"Function '{node.name}' might need error handling"
                    })
        
        return issues
    
    def analyze_project(self, focus_dirs: List[str] = None) -> Dict[str, Any]:
        """Analyze entire project or specific directories"""
        log_step("Analyzing project code quality...")
        
        if not focus_dirs:
            focus_dirs = ["scripts", "integration_tests"]
        
        all_metrics = []
        
        for dir_name in focus_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                    
                log_step(f"Analyzing {py_file.relative_to(self.project_root)}")
                metrics = self.analyze_python_file(py_file)
                all_metrics.append(metrics)
        
        return self._aggregate_metrics(all_metrics)
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from all analyzed files"""
        total_lines = sum(m.get("lines_of_code", 0) for m in all_metrics)
        total_functions = sum(len(m.get("functions", [])) for m in all_metrics)
        total_classes = sum(len(m.get("classes", [])) for m in all_metrics)
        total_complexity = sum(m.get("complexity", 0) for m in all_metrics)
        
        all_issues = []
        for m in all_metrics:
            for issue in m.get("issues", []):
                issue["file"] = m["file"]
                all_issues.append(issue)
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in all_issues:
            issues_by_type[issue["type"]].append(issue)
        
        # Find most complex functions
        complex_functions = []
        for m in all_metrics:
            for func in m.get("functions", []):
                if func["complexity"] > 10:
                    complex_functions.append({
                        "file": m["file"],
                        "function": func["name"],
                        "complexity": func["complexity"],
                        "line": func["line"]
                    })
        
        complex_functions.sort(key=lambda x: x["complexity"], reverse=True)
        
        return {
            "summary": {
                "total_files": len(all_metrics),
                "total_lines": total_lines,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "avg_complexity": total_complexity / total_functions if total_functions > 0 else 0,
                "total_issues": len(all_issues)
            },
            "issues_by_type": dict(issues_by_type),
            "complex_functions": complex_functions[:10],  # Top 10
            "file_metrics": all_metrics
        }
    
    def suggest_refactorings(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate refactoring suggestions based on analysis"""
        suggestions = []
        
        # Suggest refactoring for complex functions
        for func in analysis.get("complex_functions", []):
            if func["complexity"] > 15:
                suggestions.append({
                    "type": "extract_method",
                    "priority": "high",
                    "file": func["file"],
                    "line": func["line"],
                    "target": func["function"],
                    "reason": f"Complexity of {func['complexity']} is too high",
                    "suggestion": "Consider breaking this function into smaller, focused methods"
                })
        
        # Suggest fixing long lines
        long_line_issues = analysis.get("issues_by_type", {}).get("long_line", [])
        if len(long_line_issues) > 10:
            suggestions.append({
                "type": "code_formatting",
                "priority": "low",
                "files": list(set(issue["file"] for issue in long_line_issues)),
                "reason": f"{len(long_line_issues)} long lines found",
                "suggestion": "Use a code formatter like black to automatically fix line lengths"
            })
        
        # Suggest extracting hardcoded values
        hardcoded_issues = []
        for issue_type in ["hardcoded_number", "hardcoded_path", "hardcoded_url"]:
            hardcoded_issues.extend(analysis.get("issues_by_type", {}).get(issue_type, []))
        
        if hardcoded_issues:
            suggestions.append({
                "type": "extract_constants",
                "priority": "moderate",
                "files": list(set(issue["file"] for issue in hardcoded_issues)),
                "count": len(hardcoded_issues),
                "reason": "Hardcoded values make code less maintainable",
                "suggestion": "Extract hardcoded values to configuration files or constants"
            })
        
        # Suggest adding docstrings
        for file_metrics in analysis.get("file_metrics", []):
            if file_metrics.get("docstring_coverage", 1) < 0.5:
                suggestions.append({
                    "type": "add_documentation",
                    "priority": "moderate",
                    "file": file_metrics["file"],
                    "coverage": file_metrics["docstring_coverage"],
                    "reason": "Low docstring coverage",
                    "suggestion": "Add docstrings to undocumented functions and classes"
                })
        
        # Suggest adding type hints
        files_without_types = []
        for file_metrics in analysis.get("file_metrics", []):
            has_types = any(
                func.get("has_type_hints", False) 
                for func in file_metrics.get("functions", [])
            )
            if not has_types and file_metrics.get("functions"):
                files_without_types.append(file_metrics["file"])
        
        if files_without_types:
            suggestions.append({
                "type": "add_type_hints",
                "priority": "moderate",
                "files": files_without_types,
                "reason": "Type hints improve code clarity and enable better tooling",
                "suggestion": "Add type hints to function signatures"
            })
        
        return suggestions
    
    def generate_refactoring_report(self, focus_dirs: List[str] = None) -> str:
        """Generate comprehensive refactoring report"""
        # Analyze project
        analysis = self.analyze_project(focus_dirs)
        
        # Generate suggestions
        suggestions = self.suggest_refactorings(analysis)
        
        # Sort suggestions by priority
        priority_order = {"high": 0, "moderate": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        report = f"""# Code Refactoring Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Project**: Michigan Guardianship AI

## Summary

- **Total Files Analyzed**: {analysis['summary']['total_files']}
- **Total Lines of Code**: {analysis['summary']['total_lines']:,}
- **Total Functions**: {analysis['summary']['total_functions']}
- **Total Classes**: {analysis['summary']['total_classes']}
- **Average Complexity**: {analysis['summary']['avg_complexity']:.2f}
- **Total Issues Found**: {analysis['summary']['total_issues']}

## Top Complex Functions

These functions have high cyclomatic complexity and should be refactored:

| File | Function | Complexity | Line |
|------|----------|------------|------|
"""
        
        for func in analysis['complex_functions'][:5]:
            report += f"| {func['file']} | {func['function']} | {func['complexity']} | {func['line']} |\n"
        
        report += "\n## Issues by Type\n\n"
        
        for issue_type, issues in analysis['issues_by_type'].items():
            report += f"### {issue_type.replace('_', ' ').title()}\n"
            report += f"Found {len(issues)} occurrences\n\n"
        
        report += "\n## Refactoring Suggestions\n\n"
        
        for i, suggestion in enumerate(suggestions, 1):
            report += f"### {i}. {suggestion['type'].replace('_', ' ').title()}\n"
            report += f"**Priority**: {suggestion['priority'].upper()}\n\n"
            report += f"**Reason**: {suggestion['reason']}\n\n"
            report += f"**Suggestion**: {suggestion['suggestion']}\n\n"
            
            if 'files' in suggestion:
                report += "**Affected Files**:\n"
                for file in suggestion['files'][:5]:
                    report += f"- {file}\n"
                if len(suggestion['files']) > 5:
                    report += f"- ...and {len(suggestion['files']) - 5} more\n"
            elif 'file' in suggestion:
                report += f"**File**: {suggestion['file']}\n"
            
            report += "\n"
        
        report += """## Quick Wins

1. **Run Black formatter**:
   ```bash
   black scripts/ integration_tests/
   ```

2. **Add type hints** to function signatures in core modules

3. **Extract configuration** from hardcoded values to config files

4. **Break down complex functions** into smaller, testable units

## Implementation Priority

1. **High Priority**: Address complex functions that impact maintainability
2. **Moderate Priority**: Add documentation and type hints
3. **Low Priority**: Fix formatting issues

## Next Steps

1. Review complex functions and plan decomposition
2. Create configuration management for hardcoded values
3. Add comprehensive docstrings to public APIs
4. Consider adopting a linter (e.g., pylint, flake8) in CI/CD
"""
        
        return report
    
    def apply_safe_refactorings(self, file_path: Path) -> List[str]:
        """Apply safe, automated refactorings to a file"""
        # This would implement actual refactoring logic
        # For now, return suggestions
        return [
            f"Would format {file_path} with black",
            f"Would add type hints to {file_path}",
            f"Would extract constants from {file_path}"
        ]


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Code Refactorer - Improve code quality"
    )
    parser.add_argument(
        "--analyze",
        nargs="*",
        help="Directories to analyze (default: scripts, integration_tests)"
    )
    parser.add_argument(
        "--file",
        help="Analyze a specific file"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full refactoring report"
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Show refactoring suggestions"
    )
    
    args = parser.parse_args()
    
    refactorer = CodeRefactorer()
    
    if args.file:
        # Analyze single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File {file_path} not found")
            return
        
        metrics = refactorer.analyze_python_file(file_path)
        print(json.dumps(metrics, indent=2))
    
    elif args.report:
        # Generate full report
        report = refactorer.generate_refactoring_report(args.analyze)
        print(report)
        
        # Save report
        report_path = Path("results/code_refactoring_report.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    elif args.suggest:
        # Show suggestions
        analysis = refactorer.analyze_project(args.analyze)
        suggestions = refactorer.suggest_refactorings(analysis)
        
        print("Refactoring Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['type']} ({suggestion['priority']} priority)")
            print(f"   Reason: {suggestion['reason']}")
            print(f"   Action: {suggestion['suggestion']}")
    
    else:
        # Default: analyze project
        analysis = refactorer.analyze_project(args.analyze)
        print(f"\nCode Quality Summary:")
        print(f"Files analyzed: {analysis['summary']['total_files']}")
        print(f"Total lines: {analysis['summary']['total_lines']:,}")
        print(f"Average complexity: {analysis['summary']['avg_complexity']:.2f}")
        print(f"Issues found: {analysis['summary']['total_issues']}")
        
        if analysis['complex_functions']:
            print("\nMost complex functions:")
            for func in analysis['complex_functions'][:3]:
                print(f"  - {func['function']} in {func['file']} (complexity: {func['complexity']})")


if __name__ == "__main__":
    main()