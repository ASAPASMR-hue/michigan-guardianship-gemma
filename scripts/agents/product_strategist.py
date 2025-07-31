#!/usr/bin/env python3
"""
Product Strategist Agent
Analyzes feature gaps, user needs, and strategic product opportunities
"""

import os
import sys
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.log_step import log_step


class ProductStrategist:
    """Analyzes product strategy and identifies feature opportunities"""
    
    def __init__(self):
        """Initialize product strategist"""
        self.project_root = Path(__file__).parent.parent.parent
        self.test_questions = self._load_test_questions()
        self.kb_coverage = {}
        self.feature_gaps = []
        self.user_personas = self._define_user_personas()
        
    def _load_test_questions(self) -> pd.DataFrame:
        """Load test questions for analysis"""
        csv_paths = [
            self.project_root / "data/Synthetic Test Questions.xlsx",
            Path("/Users/claytoncanady/Library/michigan-guardianship-ai/Synthetic Test Questions - Sheet1.csv")
        ]
        
        for path in csv_paths:
            if path.exists():
                try:
                    if path.suffix == '.xlsx':
                        return pd.read_excel(path)
                    else:
                        return pd.read_csv(path)
                except Exception as e:
                    log_step(f"Error loading test questions: {e}", level="error")
        
        # Create empty dataframe if no file found
        return pd.DataFrame(columns=['id', 'question', 'category', 'complexity_tier'])
    
    def _define_user_personas(self) -> Dict[str, Dict[str, Any]]:
        """Define user personas for Michigan guardianship system"""
        return {
            "Overwhelmed Parent": {
                "description": "Parent suddenly needing guardianship due to crisis",
                "needs": ["quick answers", "simple explanations", "clear next steps"],
                "pain_points": ["legal jargon", "complex procedures", "time pressure"],
                "priority_features": ["crisis mode", "step-by-step guidance", "form help"]
            },
            "Grandparent Caregiver": {
                "description": "Grandparent seeking formal guardianship of grandchild",
                "needs": ["procedural clarity", "cost information", "timeline expectations"],
                "pain_points": ["technology barriers", "court intimidation", "financial concerns"],
                "priority_features": ["simple UI", "cost calculator", "hearing preparation"]
            },
            "Legal Professional": {
                "description": "Attorney or paralegal assisting with guardianship",
                "needs": ["accurate citations", "comprehensive coverage", "efficiency tools"],
                "pain_points": ["incomplete information", "outdated references", "client education"],
                "priority_features": ["citation verification", "bulk processing", "client handouts"]
            },
            "Social Worker": {
                "description": "Social worker guiding families through the process",
                "needs": ["resource links", "multi-language support", "special circumstances"],
                "pain_points": ["ICWA complexity", "resource availability", "family dynamics"],
                "priority_features": ["ICWA guidance", "resource directory", "case notes"]
            }
        }
    
    def analyze_question_coverage(self) -> Dict[str, Any]:
        """Analyze coverage of different question types"""
        log_step("Analyzing question coverage...")
        
        if self.test_questions.empty:
            return {"error": "No test questions available"}
        
        # Categorize questions
        category_counts = self.test_questions['category'].value_counts().to_dict()
        complexity_counts = self.test_questions['complexity_tier'].value_counts().to_dict()
        
        # Identify question patterns
        question_patterns = {
            "procedural": r"(how|what|when|where).*(file|submit|court|form)",
            "eligibility": r"(who|can|eligible|qualify|standing)",
            "cost": r"(cost|fee|pay|afford|waiver)",
            "timeline": r"(how long|when|timeline|duration|days)",
            "icwa": r"(indian|native|tribal|icwa|mifpa)",
            "emergency": r"(emergency|urgent|immediate|crisis)",
            "modification": r"(change|modify|terminate|end)",
            "rights": r"(rights|visitation|parenting time|contact)"
        }
        
        pattern_counts = {}
        for pattern_name, pattern in question_patterns.items():
            matches = self.test_questions['question'].str.contains(pattern, case=False, na=False)
            pattern_counts[pattern_name] = matches.sum()
        
        # Identify gaps
        expected_min_counts = {
            "procedural": 30,
            "eligibility": 20,
            "cost": 15,
            "timeline": 15,
            "icwa": 10,
            "emergency": 10,
            "modification": 10,
            "rights": 15
        }
        
        coverage_gaps = []
        for pattern, expected in expected_min_counts.items():
            actual = pattern_counts.get(pattern, 0)
            if actual < expected:
                coverage_gaps.append({
                    "type": pattern,
                    "expected": expected,
                    "actual": actual,
                    "gap": expected - actual
                })
        
        return {
            "total_questions": len(self.test_questions),
            "category_distribution": category_counts,
            "complexity_distribution": complexity_counts,
            "pattern_coverage": pattern_counts,
            "coverage_gaps": coverage_gaps
        }
    
    def analyze_kb_completeness(self) -> Dict[str, Any]:
        """Analyze knowledge base completeness"""
        log_step("Analyzing knowledge base completeness...")
        
        kb_dir = self.project_root / "kb_files"
        if not kb_dir.exists():
            return {"error": "Knowledge base directory not found"}
        
        kb_analysis = {
            "total_files": 0,
            "total_size_mb": 0,
            "file_types": defaultdict(int),
            "content_analysis": {},
            "missing_topics": []
        }
        
        # Expected topics in a complete guardianship KB
        expected_topics = {
            "forms": ["PC 651", "PC 650", "MC 20", "PC 652", "PC 678"],
            "procedures": ["filing", "service", "hearing", "notification"],
            "requirements": ["standing", "eligibility", "consent", "investigation"],
            "special_cases": ["ICWA", "emergency", "limited", "temporary"],
            "post_order": ["modification", "termination", "annual review"],
            "resources": ["legal aid", "court locations", "fee waiver", "interpreters"]
        }
        
        # Analyze each KB file
        all_content = ""
        for kb_file in kb_dir.rglob("*"):
            if kb_file.is_file():
                kb_analysis["total_files"] += 1
                kb_analysis["total_size_mb"] += kb_file.stat().st_size / (1024 * 1024)
                kb_analysis["file_types"][kb_file.suffix] += 1
                
                try:
                    content = kb_file.read_text(errors='ignore')
                    all_content += content.lower() + "\n"
                except:
                    pass
        
        # Check for expected topics
        for category, topics in expected_topics.items():
            found_topics = []
            missing_topics = []
            
            for topic in topics:
                if topic.lower() in all_content:
                    found_topics.append(topic)
                else:
                    missing_topics.append(topic)
            
            kb_analysis["content_analysis"][category] = {
                "found": found_topics,
                "missing": missing_topics,
                "coverage": len(found_topics) / len(topics) if topics else 0
            }
            
            if missing_topics:
                kb_analysis["missing_topics"].extend([
                    {"category": category, "topic": topic} for topic in missing_topics
                ])
        
        return kb_analysis
    
    def identify_feature_opportunities(self) -> List[Dict[str, Any]]:
        """Identify strategic feature opportunities"""
        log_step("Identifying feature opportunities...")
        
        opportunities = []
        
        # Analyze current capabilities
        existing_features = self._analyze_existing_features()
        
        # Feature ideas based on user personas
        persona_features = {
            "Interactive Form Wizard": {
                "description": "Step-by-step form completion with validation",
                "personas": ["Overwhelmed Parent", "Grandparent Caregiver"],
                "impact": "high",
                "complexity": "medium",
                "rationale": "Reduces form errors and anxiety"
            },
            "Timeline Calculator": {
                "description": "Personalized timeline based on case specifics",
                "personas": ["Overwhelmed Parent", "Legal Professional"],
                "impact": "high",
                "complexity": "low",
                "rationale": "Sets clear expectations and deadlines"
            },
            "Cost Estimator": {
                "description": "Calculate total costs including filing, service, etc.",
                "personas": ["Grandparent Caregiver", "Social Worker"],
                "impact": "medium",
                "complexity": "low",
                "rationale": "Financial planning and fee waiver guidance"
            },
            "ICWA Compliance Checker": {
                "description": "Automated ICWA requirement verification",
                "personas": ["Legal Professional", "Social Worker"],
                "impact": "high",
                "complexity": "high",
                "rationale": "Prevents costly ICWA violations"
            },
            "Multi-language Support": {
                "description": "Spanish, Arabic, and other language translations",
                "personas": ["Social Worker", "Grandparent Caregiver"],
                "impact": "high",
                "complexity": "medium",
                "rationale": "Serves diverse Genesee County population"
            },
            "Document Assembly": {
                "description": "Auto-generate completed forms from Q&A",
                "personas": ["Legal Professional", "Overwhelmed Parent"],
                "impact": "very_high",
                "complexity": "high",
                "rationale": "Dramatic time savings and accuracy"
            },
            "Court Appearance Prep": {
                "description": "What to expect, what to bring, practice questions",
                "personas": ["Grandparent Caregiver", "Overwhelmed Parent"],
                "impact": "medium",
                "complexity": "low",
                "rationale": "Reduces court anxiety and improves outcomes"
            },
            "Case Status Tracker": {
                "description": "Track filing status and next steps",
                "personas": ["All"],
                "impact": "medium",
                "complexity": "medium",
                "rationale": "Reduces confusion and court inquiries"
            }
        }
        
        # Score and prioritize features
        for feature_name, details in persona_features.items():
            if feature_name not in existing_features:
                score = self._calculate_feature_score(details)
                opportunities.append({
                    "name": feature_name,
                    **details,
                    "score": score,
                    "status": "proposed"
                })
        
        # Sort by score
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        
        return opportunities
    
    def _analyze_existing_features(self) -> Set[str]:
        """Analyze existing features in the codebase"""
        existing = set()
        
        # Check for specific feature implementations
        feature_indicators = {
            "Adaptive Retrieval": "adaptive_retrieval.py",
            "Hallucination Detection": "validator_setup.py",
            "Multi-Model Support": "llm_handler.py",
            "Complexity Classification": "train_complexity_classifier.py"
        }
        
        scripts_dir = self.project_root / "scripts"
        for feature, file_name in feature_indicators.items():
            if (scripts_dir / file_name).exists():
                existing.add(feature)
        
        return existing
    
    def _calculate_feature_score(self, feature: Dict[str, Any]) -> float:
        """Calculate priority score for a feature"""
        # Impact weights
        impact_scores = {
            "very_high": 5,
            "high": 4,
            "medium": 3,
            "low": 2
        }
        
        # Complexity weights (inverse - easier is better)
        complexity_scores = {
            "low": 3,
            "medium": 2,
            "high": 1
        }
        
        # Persona coverage weight
        persona_weight = len(feature.get("personas", [])) * 0.5
        
        impact = impact_scores.get(feature.get("impact", "medium"), 3)
        complexity = complexity_scores.get(feature.get("complexity", "medium"), 2)
        
        return impact * complexity + persona_weight
    
    def analyze_competitive_landscape(self) -> Dict[str, Any]:
        """Analyze competitive landscape and best practices"""
        # This would normally involve market research
        # For now, return strategic insights
        
        return {
            "competitors": [
                {
                    "name": "Generic Legal AI",
                    "strengths": ["Broad coverage", "Multi-state"],
                    "weaknesses": ["Not Michigan-specific", "No ICWA expertise"],
                    "opportunities": ["Specialize in Michigan", "ICWA excellence"]
                },
                {
                    "name": "Court Self-Help Centers",
                    "strengths": ["Official", "Free"],
                    "weaknesses": ["Limited hours", "No AI assistance"],
                    "opportunities": ["24/7 availability", "Intelligent guidance"]
                }
            ],
            "best_practices": [
                "Plain language explanations",
                "Mobile-first design",
                "Offline capability for court",
                "Integration with e-filing",
                "Accessibility compliance"
            ],
            "differentiation": [
                "Genesee County expertise",
                "Zero hallucination guarantee",
                "ICWA specialization",
                "Multi-model verification"
            ]
        }
    
    def generate_product_strategy_report(self) -> str:
        """Generate comprehensive product strategy report"""
        # Run analyses
        coverage = self.analyze_question_coverage()
        kb_analysis = self.analyze_kb_completeness()
        opportunities = self.identify_feature_opportunities()
        competitive = self.analyze_competitive_landscape()
        
        report = f"""# Product Strategy Report

**Project**: Michigan Guardianship AI
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

This report analyzes the current product state, identifies strategic opportunities, and recommends a roadmap for enhancing the Michigan Guardianship AI system to better serve Genesee County families.

## Question Coverage Analysis

### Current State
- **Total Questions**: {coverage.get('total_questions', 0)}
- **Categories Covered**: {len(coverage.get('category_distribution', {}))}
- **Complexity Tiers**: {len(coverage.get('complexity_distribution', {}))}

### Coverage by Pattern
"""
        
        if 'pattern_coverage' in coverage:
            report += "| Pattern | Count | Status |\n"
            report += "|---------|--------|--------|\n"
            for pattern, count in coverage['pattern_coverage'].items():
                status = "✅" if count >= 10 else "⚠️"
                report += f"| {pattern.title()} | {count} | {status} |\n"
        
        if 'coverage_gaps' in coverage:
            report += "\n### Identified Gaps\n"
            for gap in coverage['coverage_gaps']:
                report += f"- **{gap['type'].title()}**: Need {gap['gap']} more questions\n"
        
        report += f"""

## Knowledge Base Analysis

### Repository Status
- **Total Files**: {kb_analysis.get('total_files', 0)}
- **Total Size**: {kb_analysis.get('total_size_mb', 0):.1f} MB
- **File Types**: {dict(kb_analysis.get('file_types', {}))}

### Content Coverage
"""
        
        if 'content_analysis' in kb_analysis:
            report += "| Category | Coverage | Missing Topics |\n"
            report += "|----------|----------|----------------|\n"
            for category, analysis in kb_analysis['content_analysis'].items():
                missing_count = len(analysis['missing'])
                report += f"| {category.title()} | {analysis['coverage']:.0%} | {missing_count} items |\n"
        
        report += "\n## User Personas\n\n"
        
        for persona_name, persona in self.user_personas.items():
            report += f"### {persona_name}\n"
            report += f"**Description**: {persona['description']}\n\n"
            report += f"**Key Needs**: {', '.join(persona['needs'])}\n\n"
            report += f"**Pain Points**: {', '.join(persona['pain_points'])}\n\n"
        
        report += "## Strategic Feature Opportunities\n\n"
        
        report += "### Top Priority Features\n\n"
        
        for i, feature in enumerate(opportunities[:5], 1):
            report += f"#### {i}. {feature['name']}\n"
            report += f"**Description**: {feature['description']}\n\n"
            report += f"**Target Users**: {', '.join(feature['personas'])}\n\n"
            report += f"**Impact**: {feature['impact']} | **Complexity**: {feature['complexity']}\n\n"
            report += f"**Rationale**: {feature['rationale']}\n\n"
            report += f"**Priority Score**: {feature['score']:.1f}\n\n"
        
        report += "## Competitive Analysis\n\n"
        
        report += "### Key Differentiators\n"
        for diff in competitive['differentiation']:
            report += f"- {diff}\n"
        
        report += "\n### Best Practices to Adopt\n"
        for practice in competitive['best_practices'][:5]:
            report += f"- {practice}\n"
        
        report += """

## Recommended Roadmap

### Phase 1: Foundation (Month 1-2)
1. **Complete KB Coverage**: Add missing ICWA and emergency topics
2. **Enhance Question Set**: Fill identified gaps in test coverage
3. **API Documentation**: Prepare for external integrations

### Phase 2: Core Features (Month 3-4)
1. **Interactive Form Wizard**: Reduce user errors
2. **Timeline Calculator**: Set clear expectations
3. **Multi-language Support**: Serve diverse population

### Phase 3: Advanced Features (Month 5-6)
1. **Document Assembly**: Auto-generate court forms
2. **ICWA Compliance Checker**: Prevent violations
3. **Case Status Tracker**: Reduce confusion

### Phase 4: Scale & Optimize (Month 7+)
1. **Performance Optimization**: Sub-second responses
2. **Analytics Dashboard**: Usage insights
3. **Partner Integrations**: Courts, legal aid

## Success Metrics

### User Satisfaction
- Target: 90%+ user satisfaction rating
- Measure: Post-interaction surveys

### Accuracy & Compliance
- Target: 99.5%+ procedural accuracy
- Measure: Expert review sampling

### Adoption
- Target: 1000+ monthly active users by Month 6
- Measure: Analytics tracking

### Impact
- Target: 50% reduction in filing errors
- Measure: Court feedback

## Investment Requirements

### Technical
- 2 Full-stack developers
- 1 ML engineer
- 1 Legal domain expert

### Timeline
- 6-month initial roadmap
- 12-month full feature set

### Budget Estimate
- Development: $300-400k
- Infrastructure: $50k/year
- Maintenance: $100k/year

## Risks & Mitigation

1. **Legal Liability**: Maintain disclaimers, attorney review
2. **Technical Debt**: Regular refactoring sprints
3. **User Adoption**: Community partnerships, marketing
4. **Sustainability**: Grant funding, court partnerships

## Conclusion

The Michigan Guardianship AI system has strong foundations. By focusing on user-centric features, addressing coverage gaps, and maintaining our commitment to accuracy, we can create the definitive resource for Genesee County families navigating guardianship.

**Next Steps**:
1. Review and prioritize feature list with stakeholders
2. Secure funding for Phase 1-2 development
3. Establish partnerships with courts and legal aid
4. Begin KB enhancement immediately
"""
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Product Strategist - Analyze product opportunities"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Analyze question coverage"
    )
    parser.add_argument(
        "--kb",
        action="store_true",
        help="Analyze knowledge base completeness"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Identify feature opportunities"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate full strategy report"
    )
    
    args = parser.parse_args()
    
    strategist = ProductStrategist()
    
    if args.coverage:
        coverage = strategist.analyze_question_coverage()
        print("Question Coverage Analysis:")
        print(f"Total questions: {coverage.get('total_questions', 0)}")
        print("\nPattern coverage:")
        for pattern, count in coverage.get('pattern_coverage', {}).items():
            print(f"  {pattern}: {count}")
        
        if coverage.get('coverage_gaps'):
            print("\nCoverage gaps:")
            for gap in coverage['coverage_gaps']:
                print(f"  {gap['type']}: need {gap['gap']} more")
    
    elif args.kb:
        kb_analysis = strategist.analyze_kb_completeness()
        print("Knowledge Base Analysis:")
        print(f"Total files: {kb_analysis.get('total_files', 0)}")
        print(f"Total size: {kb_analysis.get('total_size_mb', 0):.1f} MB")
        
        if 'missing_topics' in kb_analysis:
            print(f"\nMissing topics: {len(kb_analysis['missing_topics'])}")
            for topic in kb_analysis['missing_topics'][:5]:
                print(f"  - {topic['category']}: {topic['topic']}")
    
    elif args.features:
        opportunities = strategist.identify_feature_opportunities()
        print("Top Feature Opportunities:\n")
        for i, feature in enumerate(opportunities[:5], 1):
            print(f"{i}. {feature['name']} (score: {feature['score']:.1f})")
            print(f"   {feature['description']}")
            print(f"   Impact: {feature['impact']} | Complexity: {feature['complexity']}")
            print()
    
    elif args.report:
        report = strategist.generate_product_strategy_report()
        print(report)
        
        # Save report
        report_path = Path("results/product_strategy_report.md")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    else:
        # Default: show summary
        print("Product Strategy Summary:")
        print("\nRun with specific flags:")
        print("  --coverage   : Analyze question coverage")
        print("  --kb        : Analyze knowledge base")
        print("  --features  : Show feature opportunities")
        print("  --report    : Generate full report")


if __name__ == "__main__":
    main()