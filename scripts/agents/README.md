# Michigan Guardianship AI - Intelligent Agents

This directory contains specialized agents that enhance the testing, analysis, and optimization of the Michigan Guardianship AI system.

## Available Agents

### 1. Test Results Analyzer
**Purpose**: Advanced analysis of test results with qualitative insights

**Key Features**:
- Cross-run comparisons
- Failure pattern identification
- Executive summaries
- Strategic recommendations

**Usage**:
```bash
# Analyze a single run
python scripts/agents/test_results_analyzer.py --run_ids run_20250128_1430

# Compare multiple runs
python scripts/agents/test_results_analyzer.py --run_ids run_20250128_1430 run_20250201_0900 --compare

# Generate detailed report
python scripts/agents/test_results_analyzer.py --run_ids run_20250128_1430 --output report.md
```

### 2. Workflow Optimizer
**Purpose**: Streamline testing workflows and provide faster feedback loops

**Key Features**:
- File change detection
- Minimal test suggestions
- Preflight checks
- Golden question testing

**Usage**:
```bash
# Check what changed and get recommendations
python scripts/agents/workflow_optimizer.py --report

# Run preflight checks
python scripts/agents/workflow_optimizer.py --check

# Quick golden test
python scripts/agents/workflow_optimizer.py --golden-test

# Test specific model
python scripts/agents/workflow_optimizer.py --golden-test --model "meta-llama/llama-3.3-70b-instruct:free"
```

### 3. AI Integration Expert
**Purpose**: Optimize prompts and AI techniques based on test results

**Key Features**:
- Failed case analysis
- Prompt variation generation
- A/B testing of prompts
- Optimization recommendations

**Usage**:
```bash
# Analyze failed cases
python scripts/agents/ai_integration_expert.py --analyze run_20250128_1430

# Optimize prompts for failures
python scripts/agents/ai_integration_expert.py --analyze run_20250128_1430 --optimize

# Generate optimization report
python scripts/agents/ai_integration_expert.py --analyze run_20250128_1430 --report
```

## Agent Architecture

Each agent follows a consistent pattern:
1. **Analysis**: Examine data and identify patterns
2. **Recommendation**: Suggest improvements
3. **Testing**: Validate recommendations
4. **Reporting**: Generate actionable insights

## Workflow Examples

### Quick Development Cycle
```bash
# 1. Make code changes
vim scripts/adaptive_retrieval.py

# 2. Check what needs testing
python scripts/agents/workflow_optimizer.py --report

# 3. Run recommended tests
python scripts/agents/workflow_optimizer.py --golden-test

# 4. If successful, run full test on one model
python scripts/run_full_evaluation.py --models "mistralai/mistral-nemo:free"
```

### Post-Evaluation Analysis
```bash
# 1. Run full evaluation
python scripts/run_full_evaluation.py

# 2. Basic analysis
python scripts/analyze_results.py --run_id run_20250128_1430

# 3. Advanced analysis
python scripts/agents/test_results_analyzer.py --run_ids run_20250128_1430

# 4. Optimize based on failures
python scripts/agents/ai_integration_expert.py --analyze run_20250128_1430 --report
```

### Continuous Improvement
```bash
# 1. Compare runs over time
python scripts/agents/test_results_analyzer.py \
  --run_ids run_20250128_1430 run_20250201_0900 run_20250215_1000 \
  --compare

# 2. Track improvements
# The agent will show performance changes across runs
```

### 4. Code Refactorer
**Purpose**: Analyze code quality and suggest improvements

**Key Features**:
- Code complexity analysis
- Identify refactoring opportunities
- Docstring coverage analysis
- Type hint recommendations

**Usage**:
```bash
# Analyze code quality
python scripts/agents/code_refactorer.py --analyze

# Generate refactoring report
python scripts/agents/code_refactorer.py --report

# Show specific suggestions
python scripts/agents/code_refactorer.py --suggest
```

### 5. System Architect
**Purpose**: Analyze system architecture and component relationships

**Key Features**:
- Component dependency mapping
- Data flow analysis
- Interface documentation
- Architecture health scoring

**Usage**:
```bash
# Analyze architecture
python scripts/agents/system_architect.py --analyze

# Generate architecture report
python scripts/agents/system_architect.py --report

# Create architecture diagram
python scripts/agents/system_architect.py --diagram

# Show improvement suggestions
python scripts/agents/system_architect.py --suggest
```

### 6. Product Strategist
**Purpose**: Identify feature gaps and strategic opportunities

**Key Features**:
- Question coverage analysis
- Knowledge base completeness
- User persona mapping
- Feature opportunity scoring

**Usage**:
```bash
# Analyze question coverage
python scripts/agents/product_strategist.py --coverage

# Check knowledge base
python scripts/agents/product_strategist.py --kb

# Identify feature opportunities
python scripts/agents/product_strategist.py --features

# Generate strategy report
python scripts/agents/product_strategist.py --report
```

### 7. Analytics Engineer
**Purpose**: Deep performance analytics and insights

**Key Features**:
- Latency pattern analysis
- Performance correlation
- Cost efficiency metrics
- Visual analytics generation

**Usage**:
```bash
# Analyze run performance
python scripts/agents/analytics_engineer.py --analyze run_20250128_1430

# Generate visualizations
python scripts/agents/analytics_engineer.py --visualize run_20250128_1430

# Create analytics report
python scripts/agents/analytics_engineer.py --report run_20250128_1430

# Compare multiple runs
python scripts/agents/analytics_engineer.py --compare run_1 run_2 run_3
```

## Agent Pipeline Runner

The pipeline runner allows you to chain multiple agents together for comprehensive analysis.

### Available Pipelines

1. **quick_check**: Fast validation of changes and test recommendations
   - Agents: workflow_optimizer, test_analyzer

2. **full_analysis**: Comprehensive test results and performance analysis
   - Agents: test_analyzer, analytics_engineer, ai_expert

3. **system_review**: Architecture, code quality, and product strategy review
   - Agents: system_architect, code_refactorer, product_strategist

4. **optimization**: AI prompt and workflow optimization
   - Agents: ai_expert, workflow_optimizer

5. **comprehensive**: Run all available agents

### Usage Examples

```bash
# List available pipelines
python scripts/agents/run_agent_pipeline.py --list

# Run quick check pipeline
python scripts/agents/run_agent_pipeline.py --pipeline quick_check

# Run full analysis for a specific test run
python scripts/agents/run_agent_pipeline.py --pipeline full_analysis --run-id run_20250128_1430

# Run custom agent sequence
python scripts/agents/run_agent_pipeline.py --agents code_refactorer system_architect

# Generate report from saved results
python scripts/agents/run_agent_pipeline.py --report results/agent_pipeline/comprehensive_20250128_143000.json
```

## Development Guide

To create a new agent:

1. **Inherit base pattern**:
```python
class NewAgent:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        
    def analyze(self):
        # Examine data
        pass
        
    def recommend(self):
        # Generate suggestions
        pass
        
    def test(self):
        # Validate recommendations
        pass
        
    def report(self):
        # Create actionable insights
        pass
```

2. **Follow conventions**:
- Use `log_step()` for progress updates
- Return structured data (dicts/lists)
- Generate markdown reports
- Include command-line interface

3. **Document thoroughly**:
- Clear docstrings
- Usage examples
- Expected inputs/outputs

## Additional Tools

### Dashboard Generator
**Purpose**: Create interactive HTML dashboards from results

**Features**:
- Plotly-based visualizations
- Unified reporting
- Index page generation

**Usage**:
```bash
python dashboard_generator.py --test-results run_20250128_1430 --open
```

### Agent Scheduler
**Purpose**: Automate periodic agent runs

**Features**:
- Cron-like scheduling
- Job management
- Failure recovery

**Usage**:
```bash
python agent_scheduler.py --add-defaults
python agent_scheduler.py --start --daemon
```

### Performance Monitor
**Purpose**: Track agent execution performance

**Features**:
- Real-time monitoring
- Alert generation
- Health scoring

**Usage**:
```bash
python performance_monitor.py --report
python performance_monitor.py --health
```

### Configuration Manager
**Purpose**: Centralized configuration management

**Features**:
- Profile management
- Validation
- History tracking

**Usage**:
```bash
python config_manager.py --profile production
python config_manager.py --validate all
```

## Utilities

### Makefile
Simplifies common operations:
```bash
make help              # Show all commands
make quick-check       # Fast validation
make full-analysis RUN_ID=xxx  # Complete analysis
make reports           # Generate all reports
```

### Agent Utils
Common utility functions:
```bash
python agent_utils.py --latest-run    # Get latest run ID
python agent_utils.py --git-info      # Show git information
python agent_utils.py --demo          # Run utility demos
```

### Quick Start
Interactive setup guide:
```bash
python quick_start.py  # Launch interactive guide
```

### Test Suite
Comprehensive testing:
```bash
python test_all_agents.py         # Run all tests
python test_all_agents.py --quick # Quick smoke tests
python test_all_agents.py --list  # List test classes
```

## Tips

- **Start with Quick Start** if you're new to the system
- **Use the Makefile** for common operations
- **Start with Workflow Optimizer** for quick feedback during development
- **Use Test Results Analyzer** for comprehensive post-run analysis
- **Apply AI Integration Expert** when specific patterns of failure emerge
- **Chain agents** for maximum insight (optimize → test → analyze → repeat)
- **Monitor performance** regularly to catch issues early
- **Use configuration profiles** for different environments