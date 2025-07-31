#!/usr/bin/env python3
"""
generate_test_report.py - Generate comprehensive test report for Michigan Guardianship AI
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def generate_markdown_report(test_results_dir: Path) -> str:
    """Generate a comprehensive markdown report from test results"""
    
    report_lines = []
    report_lines.append("# Michigan Guardianship AI - Integration Test Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n---\n")
    
    # Find all test result files
    result_files = list(test_results_dir.glob("*.json"))
    
    if not result_files:
        report_lines.append("## No test results found\n")
        return "\n".join(report_lines)
    
    # Process each test result file
    all_tests_summary = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "pipeline_components": {},
        "golden_qa_results": [],
        "latency_compliance": {}
    }
    
    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        report_lines.append(f"## Test Run: {result_file.stem}")
        report_lines.append(f"**Timestamp:** {data.get('timestamp', 'Unknown')}\n")
        
        # Pipeline tests
        if "pipeline_tests" in data:
            report_lines.append("### Pipeline Component Tests\n")
            report_lines.append("| Component | Status | Details |")
            report_lines.append("|-----------|--------|---------|")
            
            for component, result in data["pipeline_tests"].items():
                status = result.get("status", "UNKNOWN")
                details = ""
                
                if component == "document_embedding":
                    details = f"{result.get('chunks_created', 0)} chunks created, {result.get('time_seconds', 0):.2f}s"
                elif component == "latency_compliance":
                    if "summary" in result:
                        compliance_rates = []
                        for tier, stats in result["summary"].items():
                            rate = stats.get("compliance_rate", 0)
                            compliance_rates.append(f"{tier}: {rate:.0f}%")
                        details = ", ".join(compliance_rates)
                
                report_lines.append(f"| {component.replace('_', ' ').title()} | {status} | {details} |")
                
                # Update summary
                all_tests_summary["pipeline_components"][component] = status
        
        # Golden Q&A tests
        if "golden_qa_tests" in data:
            report_lines.append("\n### Golden Q&A Test Results\n")
            
            passed = sum(1 for t in data["golden_qa_tests"] if t.get("overall_status") == "PASSED")
            total = len(data["golden_qa_tests"])
            
            report_lines.append(f"**Overall:** {passed}/{total} passed ({passed/total*100:.1f}%)\n")
            
            # Group by status
            failed_tests = [t for t in data["golden_qa_tests"] if t.get("overall_status") != "PASSED"]
            
            if failed_tests:
                report_lines.append("#### Failed Tests:\n")
                for test in failed_tests:
                    report_lines.append(f"- **Question:** {test['question']}")
                    report_lines.append(f"  - **Expected Type:** {test.get('expected_type', 'Unknown')}")
                    
                    if "tests" in test:
                        for test_name, result in test["tests"].items():
                            if result.get("status") != "PASSED":
                                report_lines.append(f"  - **{test_name}:** {result}")
                    report_lines.append("")
            
            # Update summary
            all_tests_summary["total_tests"] += total
            all_tests_summary["passed"] += passed
            all_tests_summary["failed"] += (total - passed)
            all_tests_summary["golden_qa_results"].extend(data["golden_qa_tests"])
        
        # E2E test results
        if "tests" in data:  # End-to-end format
            report_lines.append("\n### End-to-End Test Results\n")
            
            passed = sum(1 for t in data["tests"] if t.get("overall_status") == "PASSED")
            total = len(data["tests"])
            
            report_lines.append(f"**Overall:** {passed}/{total} passed ({passed/total*100:.1f}%)\n")
            
            # Average timings
            if "summary" in data:
                if "avg_retrieval_time" in data["summary"]:
                    report_lines.append(f"**Avg Retrieval Time:** {data['summary']['avg_retrieval_time']:.2f}s")
                if "avg_generation_time" in data["summary"]:
                    report_lines.append(f"**Avg Generation Time:** {data['summary']['avg_generation_time']:.2f}s")
                report_lines.append("")
            
            # Failed tests details
            failed_tests = [t for t in data["tests"] if t.get("overall_status") != "PASSED"]
            if failed_tests:
                report_lines.append("#### Failed Tests:\n")
                for test in failed_tests:
                    report_lines.append(f"- **Query:** {test['query']}")
                    if "steps" in test:
                        for step_name, step in test["steps"].items():
                            if step.get("status") != "success":
                                report_lines.append(f"  - **{step_name}:** {step.get('issues', step.get('failure_reason', 'Failed'))}")
                    report_lines.append("")
        
        report_lines.append("\n---\n")
    
    # Overall Summary
    report_lines.append("## Overall Test Summary\n")
    
    if all_tests_summary["total_tests"] > 0:
        pass_rate = (all_tests_summary["passed"] / all_tests_summary["total_tests"]) * 100
        report_lines.append(f"**Total Tests:** {all_tests_summary['total_tests']}")
        report_lines.append(f"**Passed:** {all_tests_summary['passed']}")
        report_lines.append(f"**Failed:** {all_tests_summary['failed']}")
        report_lines.append(f"**Pass Rate:** {pass_rate:.1f}%\n")
    
    # Component status
    if all_tests_summary["pipeline_components"]:
        report_lines.append("### Pipeline Component Status\n")
        for component, status in all_tests_summary["pipeline_components"].items():
            emoji = "✅" if status == "PASSED" else "❌"
            report_lines.append(f"- {emoji} {component.replace('_', ' ').title()}: {status}")
    
    # Key findings
    report_lines.append("\n## Key Findings\n")
    
    # Analyze common failure patterns
    if all_tests_summary["golden_qa_results"]:
        failure_patterns = {}
        for test in all_tests_summary["golden_qa_results"]:
            if test.get("overall_status") != "PASSED":
                for test_name, result in test.get("tests", {}).items():
                    if result.get("status") != "PASSED":
                        if test_name not in failure_patterns:
                            failure_patterns[test_name] = 0
                        failure_patterns[test_name] += 1
        
        if failure_patterns:
            report_lines.append("### Common Failure Patterns:\n")
            for pattern, count in sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"- **{pattern}:** {count} occurrences")
    
    report_lines.append("\n## Recommendations\n")
    
    # Generate recommendations based on findings
    recommendations = []
    
    if pass_rate < 80:
        recommendations.append("- **Critical:** Overall pass rate is below 80%. Focus on improving retrieval accuracy and response generation.")
    
    if "content_check" in failure_patterns and failure_patterns["content_check"] > 3:
        recommendations.append("- **High Priority:** Many content verification failures. Review document chunking and retrieval strategies.")
    
    if "latency_compliance" in all_tests_summary["pipeline_components"]:
        if all_tests_summary["pipeline_components"]["latency_compliance"] != "PASSED":
            recommendations.append("- **Performance:** Latency targets not met. Consider optimizing retrieval pipeline or adjusting budgets.")
    
    if not recommendations:
        recommendations.append("- System is performing well. Continue monitoring for edge cases.")
    
    for rec in recommendations:
        report_lines.append(rec)
    
    return "\n".join(report_lines)

def main():
    """Generate test report"""
    test_results_dir = Path(__file__).parent / "test_results"
    
    if not test_results_dir.exists():
        print("No test results directory found. Run tests first.")
        return
    
    # Generate markdown report
    report_content = generate_markdown_report(test_results_dir)
    
    # Save report
    report_path = test_results_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Test report generated: {report_path}")
    
    # Also print to console
    print("\n" + "="*60)
    print(report_content)
    print("="*60)

if __name__ == "__main__":
    main()