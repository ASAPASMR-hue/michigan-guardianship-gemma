# Michigan Guardianship AI - Agent Operations Makefile
# Simplifies common agent operations and workflows

.PHONY: help quick-check full-analysis system-review health-check reports test setup clean

# Default target
help:
	@echo "Michigan Guardianship AI - Agent Operations"
	@echo ""
	@echo "Quick Operations:"
	@echo "  make quick-check       - Fast system validation"
	@echo "  make health-check      - System health status"
	@echo "  make golden-test       - Run golden question test"
	@echo ""
	@echo "Analysis Operations:"
	@echo "  make full-analysis RUN_ID=xxx  - Complete test analysis"
	@echo "  make system-review             - Architecture & code review"
	@echo "  make optimization RUN_ID=xxx   - AI & workflow optimization"
	@echo ""
	@echo "Reporting:"
	@echo "  make reports           - Generate all reports"
	@echo "  make dashboard RUN_ID=xxx - Generate dashboard"
	@echo "  make dashboard-index   - Update dashboard index"
	@echo ""
	@echo "Automation:"
	@echo "  make schedule-start    - Start agent scheduler"
	@echo "  make schedule-status   - Check scheduler status"
	@echo "  make monitor-demo      - Run monitoring demo"
	@echo ""
	@echo "Maintenance:"
	@echo "  make test             - Run all agent tests"
	@echo "  make test-quick       - Run quick smoke tests"
	@echo "  make setup            - Initialize agent environment"
	@echo "  make clean            - Clean temporary files"

# Environment setup
setup:
	@echo "Setting up agent environment..."
	@mkdir -p ../../agent_configs
	@mkdir -p ../../monitoring
	@mkdir -p ../../scheduler
	@mkdir -p ../../results/dashboards
	@mkdir -p ../../results/agent_pipeline
	@python config_manager.py --report > /dev/null 2>&1 || true
	@python agent_scheduler.py --add-defaults
	@echo "✅ Agent environment initialized"

# Quick system check
quick-check:
	@echo "Running quick system check..."
	@python run_agent_pipeline.py --pipeline quick_check

# Golden question test
golden-test:
	@echo "Running golden question test..."
	@python workflow_optimizer.py --golden-test

# System health check
health-check:
	@echo "Checking system health..."
	@python performance_monitor.py --health
	@echo ""
	@python config_manager.py --validate test_analyzer --validate workflow_optimizer

# Full analysis pipeline
full-analysis:
ifndef RUN_ID
	@echo "Error: RUN_ID is required"
	@echo "Usage: make full-analysis RUN_ID=run_20250128_1430"
	@exit 1
endif
	@echo "Running full analysis for $(RUN_ID)..."
	@python run_agent_pipeline.py --pipeline full_analysis --run-id $(RUN_ID)

# System review pipeline
system-review:
	@echo "Running system review..."
	@python run_agent_pipeline.py --pipeline system_review

# Optimization pipeline
optimization:
ifndef RUN_ID
	@echo "Error: RUN_ID is required"
	@echo "Usage: make optimization RUN_ID=run_20250128_1430"
	@exit 1
endif
	@echo "Running optimization for $(RUN_ID)..."
	@python run_agent_pipeline.py --pipeline optimization --run-id $(RUN_ID)

# Generate all reports
reports:
	@echo "Generating all reports..."
	@python performance_monitor.py --report
	@python config_manager.py --report
	@python system_architect.py --report
	@python product_strategist.py --report
	@echo "✅ All reports generated in respective directories"

# Generate dashboard
dashboard:
ifndef RUN_ID
	@echo "Error: RUN_ID is required"
	@echo "Usage: make dashboard RUN_ID=run_20250128_1430"
	@exit 1
endif
	@echo "Generating dashboard for $(RUN_ID)..."
	@python dashboard_generator.py --test-results $(RUN_ID) --open

# Update dashboard index
dashboard-index:
	@echo "Updating dashboard index..."
	@python dashboard_generator.py --index --open

# Start scheduler
schedule-start:
	@echo "Starting agent scheduler..."
	@python agent_scheduler.py --start
	@echo "Scheduler started. Check status with: make schedule-status"

# Check scheduler status
schedule-status:
	@python agent_scheduler.py --status

# List scheduled jobs
schedule-list:
	@python agent_scheduler.py --list-jobs

# Run monitoring demo
monitor-demo:
	@echo "Running performance monitoring demo..."
	@python performance_monitor.py --demo

# Run all tests
test:
	@echo "Running comprehensive agent tests..."
	@python test_all_agents.py

# Run quick tests
test-quick:
	@echo "Running quick smoke tests..."
	@python test_all_agents.py --quick

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find ../../monitoring -name "*.jsonl" -mtime +30 -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Advanced operations
.PHONY: compare-runs profile-apply validate-all monitor-cleanup

# Compare multiple runs
compare-runs:
ifndef RUNS
	@echo "Error: RUNS is required"
	@echo "Usage: make compare-runs RUNS='run1 run2 run3'"
	@exit 1
endif
	@echo "Comparing runs: $(RUNS)"
	@python analytics_engineer.py --compare $(RUNS)

# Apply configuration profile
profile-apply:
ifndef PROFILE
	@echo "Error: PROFILE is required"
	@echo "Usage: make profile-apply PROFILE=production"
	@exit 1
endif
	@echo "Applying profile: $(PROFILE)"
	@python config_manager.py --profile $(PROFILE)

# Validate all agent configurations
validate-all:
	@echo "Validating all agent configurations..."
	@for agent in test_analyzer workflow_optimizer ai_expert code_refactorer system_architect product_strategist analytics_engineer; do \
		echo -n "Validating $$agent... "; \
		python config_manager.py --validate $$agent > /dev/null 2>&1 && echo "✅" || echo "❌"; \
	done

# Clean old monitoring data
monitor-cleanup:
ifndef DAYS
	DAYS=30
endif
	@echo "Cleaning monitoring data older than $(DAYS) days..."
	@python performance_monitor.py --cleanup $(DAYS)

# Development helpers
.PHONY: dev-monitor dev-scheduler dev-pipeline

# Run with performance monitoring
dev-monitor:
ifndef CMD
	@echo "Error: CMD is required"
	@echo "Usage: make dev-monitor CMD='python my_script.py'"
	@exit 1
endif
	@echo "Running with performance monitoring: $(CMD)"
	@python -c "from performance_monitor import monitor_agent; \
		with monitor_agent('dev_test', 'command', None) as m: \
		    import subprocess; \
		    subprocess.run('$(CMD)', shell=True)"

# Interactive scheduler management
dev-scheduler:
	@python -i -c "from agent_scheduler import AgentScheduler; \
		scheduler = AgentScheduler(); \
		print('Scheduler loaded as: scheduler'); \
		print('Try: scheduler.get_status()')"

# Interactive pipeline testing
dev-pipeline:
	@python -i -c "from run_agent_pipeline import AgentPipeline; \
		pipeline = AgentPipeline(); \
		print('Pipeline loaded as: pipeline'); \
		print('Try: pipeline.list_pipelines()')"

# Generate comprehensive status report
.PHONY: status-report

status-report:
	@echo "Michigan Guardianship AI - System Status Report"
	@echo "=============================================="
	@echo ""
	@echo "System Health:"
	@python performance_monitor.py --health | grep -E "(Score|CPU|Memory|Disk)" || echo "No health data"
	@echo ""
	@echo "Active Configurations:"
	@python config_manager.py --get test_analyzer | grep -E "(enabled|version)" || echo "No config data"
	@echo ""
	@echo "Recent Agent Runs:"
	@ls -la ../../results/ 2>/dev/null | tail -5 || echo "No recent runs"
	@echo ""
	@echo "Scheduler Status:"
	@python agent_scheduler.py --status | grep -E "(Status|Jobs)" || echo "Scheduler not running"
	@echo ""
	@echo "Dashboard Count:"
	@ls ../../results/dashboards/*.html 2>/dev/null | wc -l | xargs echo "Total dashboards:"

# Installation check
.PHONY: check-deps

check-deps:
	@echo "Checking agent dependencies..."
	@python -c "import plotly" && echo "✅ plotly" || echo "❌ plotly (run: pip install plotly)"
	@python -c "import jinja2" && echo "✅ jinja2" || echo "❌ jinja2 (run: pip install jinja2)"
	@python -c "import schedule" && echo "✅ schedule" || echo "❌ schedule (run: pip install schedule)"
	@python -c "import psutil" && echo "✅ psutil" || echo "❌ psutil (run: pip install psutil)"
	@python -c "import matplotlib" && echo "✅ matplotlib" || echo "❌ matplotlib"
	@python -c "import seaborn" && echo "✅ seaborn" || echo "❌ seaborn"
	@python -c "import graphviz" && echo "✅ graphviz" || echo "❌ graphviz"