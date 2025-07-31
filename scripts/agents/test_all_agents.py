#!/usr/bin/env python3
"""
Comprehensive Testing Suite for All Agents
Tests functionality, performance, and integration of all agents
"""

import os
import sys
import json
import yaml
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import time
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all agents
from test_results_analyzer import TestResultsAnalyzer
from workflow_optimizer import WorkflowOptimizer
from ai_integration_expert import AIIntegrationExpert
from code_refactorer import CodeRefactorer
from system_architect import SystemArchitect
from product_strategist import ProductStrategist
from analytics_engineer import AnalyticsEngineer
from dashboard_generator import DashboardGenerator
from agent_scheduler import AgentScheduler, ScheduledJob
from performance_monitor import PerformanceMonitor, monitor_agent
from config_manager import ConfigManager, AgentConfig
from run_agent_pipeline import AgentPipeline


class TestAgentBase(unittest.TestCase):
    """Base class for agent tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.project_root = Path(__file__).parent.parent.parent
        
        # Create test data structure
        (cls.test_dir / "results").mkdir(exist_ok=True)
        (cls.test_dir / "scripts").mkdir(exist_ok=True)
        (cls.test_dir / "kb_files").mkdir(exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def create_test_run_data(self, run_id: str) -> Path:
        """Create test run data"""
        run_dir = self.test_dir / "results" / run_id
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # Create metadata
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "test_questions": 10,
            "models": ["model1", "model2"]
        }
        with open(run_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create evaluation metrics
        eval_metrics = [
            {
                "model": "model1",
                "average_scores": {
                    "total_score": 8.5,
                    "procedural_accuracy": 2.2,
                    "legal_accuracy": 1.8,
                    "actionability": 1.9,
                    "mode_effectiveness": 1.4,
                    "strategic_caution": 0.4,
                    "citation_quality": 0.4,
                    "harm_prevention": 0.4,
                    "simple_tier": {"total_score": 9.0},
                    "standard_tier": {"total_score": 8.5},
                    "complex_tier": {"total_score": 8.0},
                    "crisis_tier": {"total_score": 7.5}
                }
            },
            {
                "model": "model2",
                "average_scores": {
                    "total_score": 7.2,
                    "procedural_accuracy": 1.8,
                    "legal_accuracy": 1.5,
                    "actionability": 1.6,
                    "mode_effectiveness": 1.2,
                    "strategic_caution": 0.3,
                    "citation_quality": 0.4,
                    "harm_prevention": 0.4,
                    "simple_tier": {"total_score": 8.0},
                    "standard_tier": {"total_score": 7.5},
                    "complex_tier": {"total_score": 6.5},
                    "crisis_tier": {"total_score": 6.0}
                }
            }
        ]
        with open(run_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(eval_metrics, f)
        
        # Create raw results
        for model in ["model1", "model2"]:
            results = []
            for i in range(10):
                results.append({
                    "question_id": f"Q{i+1}",
                    "question_text": f"Test question {i+1}",
                    "complexity_tier": "standard",
                    "response": f"Test response {i+1}" if i < 8 else "",
                    "latency": 2.5 + i * 0.1,
                    "cost_usd": 0.001,
                    "error": "timeout" if i == 9 else None
                })
            
            with open(run_dir / f"{model}_results.json", 'w') as f:
                json.dump(results, f)
        
        return run_dir


class TestTestResultsAnalyzer(TestAgentBase):
    """Test the Test Results Analyzer agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.analyzer = TestResultsAnalyzer()
        self.analyzer.project_root = self.test_dir
    
    def test_analyze_run(self):
        """Test run analysis"""
        run_id = "test_run_001"
        self.create_test_run_data(run_id)
        
        analysis = self.analyzer.analyze_run(run_id)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("model_rankings", analysis)
        self.assertEqual(len(analysis["model_rankings"]), 2)
        self.assertEqual(analysis["model_rankings"][0]["model"], "model1")
        self.assertEqual(analysis["model_rankings"][0]["total_score"], 8.5)
    
    def test_qualitative_insights(self):
        """Test qualitative insights generation"""
        run_id = "test_run_002"
        self.create_test_run_data(run_id)
        
        insights = self.analyzer.get_qualitative_insights(run_id)
        
        self.assertIsInstance(insights, dict)
        self.assertIn("summary", insights)
        self.assertIn("key_findings", insights)
        self.assertIn("model_strengths", insights)
    
    def test_cross_run_comparison(self):
        """Test cross-run comparison"""
        run1 = "test_run_003"
        run2 = "test_run_004"
        self.create_test_run_data(run1)
        self.create_test_run_data(run2)
        
        comparison = self.analyzer.compare_runs([run1, run2])
        
        self.assertIsInstance(comparison, dict)
        self.assertIn("run_summaries", comparison)
        self.assertEqual(len(comparison["run_summaries"]), 2)


class TestWorkflowOptimizer(TestAgentBase):
    """Test the Workflow Optimizer agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.optimizer = WorkflowOptimizer()
        self.optimizer.project_root = self.test_dir
        
        # Create test files
        (self.test_dir / "scripts").mkdir(exist_ok=True)
        test_file = self.test_dir / "scripts" / "test_file.py"
        test_file.write_text("# Test file\nprint('Hello')")
    
    def test_detect_changed_files(self):
        """Test file change detection"""
        # First run - all files are "new"
        changed = self.optimizer.detect_changed_files()
        self.assertIsInstance(changed, set)
        
        # Second run - no changes
        changed2 = self.optimizer.detect_changed_files()
        self.assertEqual(len(changed2), 0)
        
        # Modify file
        test_file = self.test_dir / "scripts" / "test_file.py"
        test_file.write_text("# Modified\nprint('Hello World')")
        
        # Third run - should detect change
        changed3 = self.optimizer.detect_changed_files()
        self.assertGreater(len(changed3), 0)
    
    def test_preflight_check(self):
        """Test preflight checks"""
        results = self.optimizer.run_preflight_check()
        
        self.assertIsInstance(results, dict)
        self.assertIn("status", results)
        self.assertIn("checks", results)
        self.assertIn("warnings", results)
        self.assertIn("estimated_time", results)
    
    def test_suggest_minimal_tests(self):
        """Test minimal test suggestions"""
        changed_files = {"scripts/llm_handler.py", "scripts/retrieval_setup.py"}
        suggestions = self.optimizer.suggest_minimal_tests(changed_files)
        
        self.assertIn("required_tests", suggestions)
        self.assertIn("recommended_tests", suggestions)
        self.assertIn("commands", suggestions)
        self.assertGreater(len(suggestions["required_tests"]), 0)


class TestAIIntegrationExpert(TestAgentBase):
    """Test the AI Integration Expert agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.expert = AIIntegrationExpert()
        self.expert.project_root = self.test_dir
    
    def test_analyze_failed_cases(self):
        """Test failed case analysis"""
        run_id = "test_run_005"
        self.create_test_run_data(run_id)
        
        failed_cases = self.expert.analyze_failed_cases(run_id)
        
        self.assertIsInstance(failed_cases, list)
        # Should find at least one failure (timeout case)
        self.assertGreater(len(failed_cases), 0)
    
    def test_generate_prompt_variations(self):
        """Test prompt variation generation"""
        variations = self.expert.generate_prompt_variations("tribal_notification")
        
        self.assertIsInstance(variations, list)
        self.assertGreater(len(variations), 0)
        
        for variation in variations:
            self.assertIn("name", variation)
            self.assertIn("description", variation)
            self.assertIn("modification", variation)


class TestCodeRefactorer(TestAgentBase):
    """Test the Code Refactorer agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.refactorer = CodeRefactorer()
        self.refactorer.project_root = self.test_dir
        
        # Create test Python file
        test_script = self.test_dir / "scripts" / "complex_script.py"
        test_script.write_text("""
def complex_function(a, b, c, d, e):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
    return 0

class TestClass:
    def __init__(self):
        self.value = 42
        
    def method_without_docstring(self):
        return self.value * 2
""")
    
    def test_analyze_python_file(self):
        """Test Python file analysis"""
        test_file = self.test_dir / "scripts" / "complex_script.py"
        metrics = self.refactorer.analyze_python_file(test_file)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("functions", metrics)
        self.assertIn("classes", metrics)
        self.assertIn("complexity", metrics)
        self.assertGreater(metrics["complexity"], 5)  # High complexity
    
    def test_suggest_refactorings(self):
        """Test refactoring suggestions"""
        analysis = self.refactorer.analyze_project(["scripts"])
        suggestions = self.refactorer.suggest_refactorings(analysis)
        
        self.assertIsInstance(suggestions, list)
        # Should suggest refactoring for complex function
        complex_suggestions = [s for s in suggestions if s["type"] == "extract_method"]
        self.assertGreater(len(complex_suggestions), 0)


class TestSystemArchitect(TestAgentBase):
    """Test the System Architect agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.architect = SystemArchitect()
        self.architect.project_root = self.test_dir
    
    def test_analyze_architecture(self):
        """Test architecture analysis"""
        analysis = self.architect.analyze_architecture()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("components", analysis)
        self.assertIn("dependencies", analysis)
        self.assertIn("data_flows", analysis)
        self.assertIn("metrics", analysis)
    
    def test_suggest_improvements(self):
        """Test improvement suggestions"""
        analysis = self.architect.analyze_architecture()
        suggestions = self.architect.suggest_improvements(analysis)
        
        self.assertIsInstance(suggestions, list)
        for suggestion in suggestions:
            self.assertIn("type", suggestion)
            self.assertIn("priority", suggestion)
            self.assertIn("suggestion", suggestion)


class TestProductStrategist(TestAgentBase):
    """Test the Product Strategist agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.strategist = ProductStrategist()
        self.strategist.project_root = self.test_dir
        
        # Mock test questions
        self.strategist.test_questions = self.create_mock_questions()
    
    def create_mock_questions(self):
        """Create mock test questions DataFrame"""
        import pandas as pd
        
        data = {
            'id': [f'Q{i}' for i in range(1, 21)],
            'question': [f'Test question {i}' for i in range(1, 21)],
            'category': ['procedural'] * 10 + ['eligibility'] * 5 + ['cost'] * 5,
            'complexity_tier': ['simple'] * 5 + ['standard'] * 10 + ['complex'] * 5
        }
        
        return pd.DataFrame(data)
    
    def test_analyze_question_coverage(self):
        """Test question coverage analysis"""
        coverage = self.strategist.analyze_question_coverage()
        
        self.assertIsInstance(coverage, dict)
        self.assertIn("total_questions", coverage)
        self.assertEqual(coverage["total_questions"], 20)
        self.assertIn("category_distribution", coverage)
        self.assertIn("pattern_coverage", coverage)
    
    def test_identify_feature_opportunities(self):
        """Test feature opportunity identification"""
        opportunities = self.strategist.identify_feature_opportunities()
        
        self.assertIsInstance(opportunities, list)
        for opportunity in opportunities:
            self.assertIn("name", opportunity)
            self.assertIn("description", opportunity)
            self.assertIn("score", opportunity)


class TestAnalyticsEngineer(TestAgentBase):
    """Test the Analytics Engineer agent"""
    
    def setUp(self):
        """Set up test instance"""
        self.engineer = AnalyticsEngineer()
        self.engineer.results_dir = self.test_dir / "results"
    
    def test_load_run_data(self):
        """Test run data loading"""
        run_id = "test_run_006"
        self.create_test_run_data(run_id)
        
        run_data = self.engineer.load_run_data(run_id)
        
        self.assertIsInstance(run_data, dict)
        self.assertIn("metadata", run_data)
        self.assertIn("raw_results", run_data)
        self.assertIn("evaluation_metrics", run_data)
    
    def test_analyze_performance_patterns(self):
        """Test performance pattern analysis"""
        run_id = "test_run_007"
        self.create_test_run_data(run_id)
        run_data = self.engineer.load_run_data(run_id)
        
        patterns = self.engineer.analyze_performance_patterns(run_data)
        
        self.assertIsInstance(patterns, dict)
        self.assertIn("latency_analysis", patterns)
        self.assertIn("accuracy_by_complexity", patterns)
        self.assertIn("error_patterns", patterns)
        self.assertIn("cost_efficiency", patterns)


class TestDashboardGenerator(TestAgentBase):
    """Test the Dashboard Generator"""
    
    def setUp(self):
        """Set up test instance"""
        self.generator = DashboardGenerator()
        self.generator.results_dir = self.test_dir / "results"
        self.generator.dashboards_dir = self.test_dir / "dashboards"
        self.generator.dashboards_dir.mkdir(exist_ok=True)
    
    def test_generate_test_results_dashboard(self):
        """Test dashboard generation for test results"""
        run_id = "test_run_008"
        self.create_test_run_data(run_id)
        
        dashboard_path = self.generator.generate_test_results_dashboard(run_id)
        
        self.assertIsNotNone(dashboard_path)
        self.assertTrue(Path(dashboard_path).exists())
        
        # Check HTML content
        with open(dashboard_path, 'r') as f:
            content = f.read()
            self.assertIn("Test Results Analysis", content)
            self.assertIn("model_comparison", content)
    
    def test_create_index_page(self):
        """Test index page creation"""
        index_path = self.generator.create_index_page()
        
        self.assertIsNotNone(index_path)
        self.assertTrue(Path(index_path).exists())


class TestAgentScheduler(TestAgentBase):
    """Test the Agent Scheduler"""
    
    def setUp(self):
        """Set up test instance"""
        self.scheduler = AgentScheduler()
        self.scheduler.scheduler_dir = self.test_dir / "scheduler"
        self.scheduler.scheduler_dir.mkdir(exist_ok=True)
        self.scheduler.jobs_file = self.scheduler.scheduler_dir / "test_jobs.json"
    
    def test_add_remove_job(self):
        """Test adding and removing jobs"""
        job = ScheduledJob(
            name="test_job",
            agent_command="echo 'test'",
            schedule_type="daily",
            schedule_time="10:00"
        )
        
        # Add job
        self.assertTrue(self.scheduler.add_job(job))
        self.assertIn("test_job", self.scheduler.jobs)
        
        # Remove job
        self.assertTrue(self.scheduler.remove_job("test_job"))
        self.assertNotIn("test_job", self.scheduler.jobs)
    
    def test_update_job(self):
        """Test updating job"""
        job = ScheduledJob(
            name="update_test",
            agent_command="echo 'test'",
            schedule_type="daily",
            schedule_time="10:00"
        )
        
        self.scheduler.add_job(job)
        self.assertTrue(self.scheduler.update_job("update_test", {"enabled": False}))
        
        updated_job = self.scheduler.jobs["update_test"]
        self.assertFalse(updated_job.enabled)


class TestPerformanceMonitor(TestAgentBase):
    """Test the Performance Monitor"""
    
    def setUp(self):
        """Set up test instance"""
        self.monitor = PerformanceMonitor()
        self.monitor.monitor_dir = self.test_dir / "monitoring"
        self.monitor.monitor_dir.mkdir(exist_ok=True)
        self.monitor.metrics_file = self.monitor.monitor_dir / "test_metrics.jsonl"
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle"""
        monitor_id = self.monitor.start_monitoring("test_agent", "test_operation")
        
        self.assertIsNotNone(monitor_id)
        self.assertIn(monitor_id, self.monitor.active_monitors)
        
        time.sleep(0.1)  # Simulate work
        
        self.monitor.stop_monitoring(monitor_id, success=True)
        self.assertNotIn(monitor_id, self.monitor.active_monitors)
        
        # Check that metric was saved
        self.assertTrue(self.monitor.metrics_file.exists())
    
    def test_context_manager(self):
        """Test monitoring context manager"""
        with monitor_agent("test_agent", "context_test", self.monitor) as m:
            m.set_metadata("test_key", "test_value")
            time.sleep(0.1)
        
        # Verify metric was recorded
        profile = self.monitor.agent_profiles.get("test_agent")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.total_runs, 1)
        self.assertEqual(profile.successful_runs, 1)


class TestConfigManager(TestAgentBase):
    """Test the Configuration Manager"""
    
    def setUp(self):
        """Set up test instance"""
        self.manager = ConfigManager(config_dir=self.test_dir / "configs")
    
    def test_agent_config_crud(self):
        """Test agent config create/read/update/delete"""
        # Get existing config
        config = self.manager.get_agent_config("test_analyzer")
        self.assertIsNotNone(config)
        self.assertIsInstance(config, AgentConfig)
        
        # Update config
        self.assertTrue(
            self.manager.update_agent_config(
                "test_analyzer",
                {"parameters.new_param": "test_value"}
            )
        )
        
        # Verify update
        updated_config = self.manager.get_agent_config("test_analyzer")
        self.assertEqual(updated_config.parameters.get("new_param"), "test_value")
    
    def test_profile_management(self):
        """Test configuration profiles"""
        # Apply development profile
        self.assertTrue(self.manager.apply_profile("development"))
        
        # Create custom profile
        custom_profile = {
            "global": {"log_level": "ERROR"},
            "agents": {
                "*": {"parameters.custom": True}
            }
        }
        
        self.assertTrue(
            self.manager.create_profile("test_profile", custom_profile)
        )
        self.assertIn("test_profile", self.manager.config_profiles)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid config
        result = self.manager.validate_config("test_analyzer")
        self.assertTrue(result["valid"])
        
        # Create invalid config
        self.manager.update_agent_config(
            "test_analyzer",
            {"resource_limits.timeout": 5}  # Too low
        )
        
        result = self.manager.validate_config("test_analyzer")
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)


class TestAgentPipeline(TestAgentBase):
    """Test the Agent Pipeline Runner"""
    
    def setUp(self):
        """Set up test instance"""
        self.pipeline = AgentPipeline()
        self.pipeline.project_root = self.test_dir
        self.pipeline.results_dir = self.test_dir / "agent_pipeline"
        self.pipeline.results_dir.mkdir(exist_ok=True)
        
        # Mock agents
        for agent_name in self.pipeline.agents:
            self.pipeline.agents[agent_name] = Mock()
    
    def test_run_pipeline(self):
        """Test pipeline execution"""
        # Mock agent results
        self.pipeline.agents["workflow_optimizer"].run_preflight_check.return_value = {
            "status": "ready",
            "checks": {},
            "warnings": []
        }
        self.pipeline.agents["workflow_optimizer"].detect_changed_files.return_value = set()
        self.pipeline.agents["workflow_optimizer"].suggest_minimal_tests.return_value = {
            "commands": ["test command"]
        }
        
        results = self.pipeline.run_pipeline("quick_check")
        
        self.assertIsInstance(results, dict)
        self.assertEqual(results["pipeline"], "quick_check")
        self.assertIn("results", results)
        self.assertIn("summary", results)
        self.assertIn("recommendations", results)
    
    def test_pipeline_report_generation(self):
        """Test pipeline report generation"""
        mock_results = {
            "pipeline": "test_pipeline",
            "timestamp": datetime.now().isoformat(),
            "agents_run": ["test_analyzer", "workflow_optimizer"],
            "results": {
                "test_analyzer": {"status": "success"},
                "workflow_optimizer": {"status": "success"}
            },
            "summary": {
                "total_agents_run": 2,
                "successful_agents": 2,
                "total_recommendations": 0,
                "high_priority_count": 0,
                "key_findings": []
            },
            "recommendations": []
        }
        
        report = self.pipeline.generate_pipeline_report(mock_results)
        
        self.assertIsInstance(report, str)
        self.assertIn("# Agent Pipeline Report", report)
        self.assertIn("test_pipeline", report)


class TestIntegration(TestAgentBase):
    """Integration tests across multiple agents"""
    
    def test_full_agent_workflow(self):
        """Test a complete workflow using multiple agents"""
        # Create test data
        run_id = "integration_test_001"
        self.create_test_run_data(run_id)
        
        # 1. Analyze test results
        analyzer = TestResultsAnalyzer()
        analyzer.project_root = self.test_dir
        analysis = analyzer.analyze_run(run_id)
        self.assertIsNotNone(analysis)
        
        # 2. Check workflow optimization
        optimizer = WorkflowOptimizer()
        optimizer.project_root = self.test_dir
        preflight = optimizer.run_preflight_check()
        self.assertEqual(preflight["status"], "error")  # No ChromaDB in test env
        
        # 3. Generate dashboard
        generator = DashboardGenerator()
        generator.results_dir = self.test_dir / "results"
        generator.dashboards_dir = self.test_dir / "dashboards"
        generator.dashboards_dir.mkdir(exist_ok=True)
        
        dashboard_path = generator.generate_test_results_dashboard(run_id)
        self.assertTrue(Path(dashboard_path).exists())
        
        # 4. Monitor performance
        monitor = PerformanceMonitor()
        monitor.monitor_dir = self.test_dir / "monitoring"
        monitor.monitor_dir.mkdir(exist_ok=True)
        
        with monitor_agent("integration_test", "full_workflow", monitor) as m:
            m.set_metadata("test_type", "integration")
            # Workflow completed
        
        health = monitor.get_system_health()
        self.assertIn("health_score", health)


def run_all_tests():
    """Run all agent tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTestResultsAnalyzer,
        TestWorkflowOptimizer,
        TestAIIntegrationExpert,
        TestCodeRefactorer,
        TestSystemArchitect,
        TestProductStrategist,
        TestAnalyticsEngineer,
        TestDashboardGenerator,
        TestAgentScheduler,
        TestPerformanceMonitor,
        TestConfigManager,
        TestAgentPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    report = f"""
# Agent Testing Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Tests**: {result.testsRun}
**Passed**: {result.testsRun - len(result.failures) - len(result.errors)}
**Failed**: {len(result.failures)}
**Errors**: {len(result.errors)}

## Test Results

"""
    
    if result.wasSuccessful():
        report += "✅ All tests passed!\n"
    else:
        report += "❌ Some tests failed.\n\n"
        
        if result.failures:
            report += "### Failures:\n"
            for test, traceback in result.failures:
                report += f"- {test}: {traceback.splitlines()[-1]}\n"
        
        if result.errors:
            report += "\n### Errors:\n"
            for test, traceback in result.errors:
                report += f"- {test}: {traceback.splitlines()[-1]}\n"
    
    # Save report
    report_path = Path("test_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nTest report saved to: {report_path}")
    
    return result.wasSuccessful()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Agent Testing Suite"
    )
    parser.add_argument(
        "--test",
        help="Run specific test class (e.g., TestWorkflowOptimizer)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test classes"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke tests only"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available test classes:")
        test_classes = [
            "TestTestResultsAnalyzer",
            "TestWorkflowOptimizer",
            "TestAIIntegrationExpert",
            "TestCodeRefactorer",
            "TestSystemArchitect",
            "TestProductStrategist",
            "TestAnalyticsEngineer",
            "TestDashboardGenerator",
            "TestAgentScheduler",
            "TestPerformanceMonitor",
            "TestConfigManager",
            "TestAgentPipeline",
            "TestIntegration"
        ]
        for test_class in test_classes:
            print(f"  - {test_class}")
    
    elif args.test:
        # Run specific test class
        test_class = globals().get(args.test)
        if test_class:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            print(f"Test class '{args.test}' not found")
    
    elif args.quick:
        # Run quick smoke tests
        print("Running quick smoke tests...")
        suite = unittest.TestSuite()
        
        # Add one test from each class
        for test_class_name in ["TestWorkflowOptimizer", "TestConfigManager", "TestAgentPipeline"]:
            test_class = globals()[test_class_name]
            suite.addTest(test_class('test_' + unittest.TestLoader().getTestCaseNames(test_class)[0]))
        
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()