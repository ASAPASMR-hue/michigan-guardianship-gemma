# Intelligent Agents Overview

## Summary

The Michigan Guardianship AI project now includes a comprehensive suite of intelligent agents that provide advanced analysis, optimization, and strategic insights. These agents work independently or can be chained together through the pipeline runner for comprehensive analysis workflows.

## Complete Agent Inventory

### 1. Test Results Analyzer (`test_results_analyzer.py`)
**Purpose**: Advanced analysis of test results with qualitative insights

**Capabilities**:
- Cross-run performance comparisons
- Failure pattern identification  
- Model-specific strength analysis (e.g., "excels at ICWA questions")
- Executive summary generation with strategic recommendations
- Qualitative trend identification beyond raw metrics

**Best Used For**:
- Post-evaluation deep analysis
- Identifying model-specific strengths and weaknesses
- Tracking improvements over time
- Strategic deployment decisions

### 2. Workflow Optimizer (`workflow_optimizer.py`)
**Purpose**: Streamline testing workflows and provide faster feedback loops

**Capabilities**:
- File change detection using checksums
- Minimal test set recommendations based on changes
- Preflight environment checks
- Golden question set for quick validation (5 critical questions)
- Time estimates for different test scenarios

**Best Used For**:
- Quick validation during development
- Pre-commit testing
- Rapid iteration cycles
- Environment validation before full runs

### 3. AI Integration Expert (`ai_integration_expert.py`)
**Purpose**: Optimize prompts and AI techniques based on test results

**Capabilities**:
- Failed test case analysis
- Prompt variation generation for specific issues
- A/B testing of prompt modifications
- Focus on common failure patterns (ICWA, procedural accuracy)
- Automated prompt optimization recommendations

**Best Used For**:
- Improving model performance on specific question types
- Addressing systematic failures
- Prompt engineering optimization
- Compliance improvement (especially ICWA)

### 4. Code Refactorer (`code_refactorer.py`)
**Purpose**: Analyze code quality and suggest refactoring improvements

**Capabilities**:
- Cyclomatic complexity analysis
- Function and class metrics
- Docstring coverage reporting
- Type hint analysis
- Common issue detection (long lines, hardcoded values, missing error handling)
- Refactoring prioritization

**Best Used For**:
- Code quality assessment
- Technical debt identification
- Maintainability improvements
- Pre-release code reviews

### 5. System Architect (`system_architect.py`)
**Purpose**: Analyze system architecture and component relationships

**Capabilities**:
- Component dependency mapping
- Data flow visualization
- Interface completeness analysis
- Architecture health scoring
- Coupling metrics calculation
- Architecture diagram generation (requires graphviz)

**Best Used For**:
- System design reviews
- Identifying architectural improvements
- Documentation generation
- Dependency analysis

### 6. Product Strategist (`product_strategist.py`)
**Purpose**: Identify feature gaps and strategic product opportunities

**Capabilities**:
- Question coverage analysis across categories
- Knowledge base completeness assessment
- User persona definition and mapping
- Feature opportunity scoring
- Competitive landscape analysis
- Strategic roadmap generation

**Best Used For**:
- Product planning
- Feature prioritization
- Gap analysis
- User needs assessment

### 7. Analytics Engineer (`analytics_engineer.py`)
**Purpose**: Deep performance analytics and correlation analysis

**Capabilities**:
- Latency pattern analysis (mean, median, P95, P99)
- Performance degradation by complexity
- Error pattern identification
- Cost efficiency calculations
- Question difficulty analysis
- Correlation analysis between metrics
- Visualization generation (requires matplotlib/seaborn)

**Best Used For**:
- Performance optimization
- Cost-benefit analysis
- Identifying problematic questions
- Model selection decisions

## Agent Pipeline Runner (`run_agent_pipeline.py`)

The pipeline runner orchestrates multiple agents for comprehensive analysis workflows.

### Predefined Pipelines

1. **quick_check**
   - Agents: workflow_optimizer → test_analyzer
   - Purpose: Fast validation and test recommendations
   - Time: ~1 minute

2. **full_analysis**
   - Agents: test_analyzer → analytics_engineer → ai_expert
   - Purpose: Comprehensive test results and optimization
   - Time: ~5-10 minutes

3. **system_review**
   - Agents: system_architect → code_refactorer → product_strategist
   - Purpose: Full system health check
   - Time: ~5-10 minutes

4. **optimization**
   - Agents: ai_expert → workflow_optimizer
   - Purpose: Focused improvement workflow
   - Time: ~3-5 minutes

5. **comprehensive**
   - Agents: All available agents
   - Purpose: Complete system analysis
   - Time: ~15-20 minutes

### Pipeline Features

- **Sequential Execution**: Agents run in order with results passed forward
- **Error Handling**: Continues pipeline even if individual agents fail
- **Consolidated Reporting**: Unified markdown report with all findings
- **Recommendation Aggregation**: Collects and prioritizes all recommendations
- **Results Persistence**: JSON output for later analysis

## Usage Patterns

### During Development
```bash
# Quick check after code changes
python scripts/agents/workflow_optimizer.py --golden-test

# If issues found, get optimization suggestions
python scripts/agents/ai_integration_expert.py --analyze last_run --optimize
```

### After Test Run
```bash
# Full analysis pipeline
python scripts/agents/run_agent_pipeline.py --pipeline full_analysis --run-id run_20250128_1430

# Generate all visualizations
python scripts/agents/analytics_engineer.py --visualize run_20250128_1430
```

### System Review
```bash
# Comprehensive system health check
python scripts/agents/run_agent_pipeline.py --pipeline system_review

# Generate architecture diagram
python scripts/agents/system_architect.py --diagram
```

### Strategic Planning
```bash
# Product strategy analysis
python scripts/agents/product_strategist.py --report

# Cost-benefit analysis
python scripts/agents/analytics_engineer.py --report run_20250128_1430
```

## Key Benefits

1. **Faster Development Cycles**: Workflow optimizer reduces feedback time from hours to minutes
2. **Systematic Improvement**: AI expert identifies and fixes specific failure patterns
3. **Code Quality**: Refactorer maintains high code standards
4. **Strategic Insights**: Product strategist guides feature development
5. **Cost Optimization**: Analytics engineer identifies cost-saving opportunities
6. **Comprehensive Analysis**: Pipeline runner provides holistic system view

## Implementation Notes

### Dependencies
- Core: pandas, numpy, pyyaml
- Visualization: matplotlib, seaborn, graphviz
- Already included in project requirements.txt

### Performance
- Agents are designed to be lightweight and fast
- Most complete in seconds to minutes
- Visualization generation may take longer for large datasets

### Extensibility
- New agents can be added by following the base pattern
- Pipeline runner automatically discovers new agents
- Custom pipelines can be created on the fly

## Future Enhancements

1. **Parallel Execution**: Run independent agents concurrently
2. **Web Dashboard**: Interactive visualization of agent results
3. **Scheduled Runs**: Automated daily/weekly analysis
4. **Alert System**: Notify on performance degradation
5. **ML-Based Insights**: Use historical data for predictive analysis

## Conclusion

The intelligent agents transform the Michigan Guardianship AI system from a static implementation to a continuously improving, self-analyzing system. By providing automated analysis, optimization, and strategic insights, these agents enable rapid development cycles and data-driven decision making.