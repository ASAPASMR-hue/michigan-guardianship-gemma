#!/usr/bin/env python3
"""
Score Gemma retest results using the evaluation rubric
"""

import json
import yaml
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

class GemmaScorer:
    def __init__(self, rubric_path: Path):
        """Initialize scorer with evaluation rubric"""
        with open(rubric_path) as f:
            self.rubric = yaml.safe_load(f)['evaluation_rubric']
        
        # Critical form numbers and procedural items
        self.critical_forms = {
            'PC 651': 'Petition for Appointment of Guardian of Minor',
            'PC 652': 'Limited Guardianship Placement Plan',
            'PC 654': 'Annual Report of Guardian',
            'PC 633': 'Letters of Guardianship',
            'PC 658': 'Report of Guardian on Condition of Minor',
            'PC 584': 'Annual Account of Fiduciary',
            'PC 678': 'Notice to Tribes (ICWA)'
        }
        
        self.critical_deadlines = {
            '14 days': ['notice by mail', 'ex parte orders'],
            '7 days': ['personal service', 'notice of hearing'],
            '5 days': ['emergency orders'],
            '30 days': ['appeal period', 'inventory filing'],
            '56 days': ['annual report due'],
            '28 days': ['change of address notification']
        }
        
        self.genesee_specifics = {
            'filing_fee': '$175',
            'hearing_day': 'Thursday',
            'courthouse': '900 S. Saginaw Street, Flint, MI 48502'
        }
    
    def score_response(self, question: dict, response: str) -> dict:
        """Score a single response according to rubric"""
        scores = {}
        
        # Skip if response is empty or error
        if not response or response == "":
            return {
                'total_score': 0,
                'scores': {k: 0 for k in self.rubric.keys()},
                'error': 'Empty response'
            }
        
        # Score each rubric component
        scores['procedural_accuracy'] = self._score_procedural(response, question)
        scores['substantive_legal_accuracy'] = self._score_legal(response, question)
        scores['actionability'] = self._score_actionability(response, question)
        scores['mode_effectiveness'] = self._score_mode(response, question)
        scores['strategic_caution'] = self._score_caution(response, question)
        scores['citation_quality'] = self._score_citations(response)
        scores['harm_prevention'] = self._score_harm_prevention(response, question)
        
        # Apply adaptive weights based on complexity
        complexity = question.get('complexity_tier', 'standard')
        weighted_scores = {}
        
        for component, score in scores.items():
            rubric_item = self.rubric[component]
            
            # Check for adaptive weighting
            if 'adaptive_weight_by_complexity' in rubric_item:
                weight = rubric_item['adaptive_weight_by_complexity'].get(
                    complexity, rubric_item['weight']
                )
            else:
                weight = rubric_item['weight']
            
            weighted_scores[component] = score * weight
        
        total_score = sum(weighted_scores.values())
        
        return {
            'total_score': round(total_score, 2),
            'scores': scores,
            'weighted_scores': weighted_scores,
            'complexity': complexity
        }
    
    def _score_procedural(self, response: str, question: dict) -> float:
        """Score procedural accuracy (0-1)"""
        score = 1.0
        response_lower = response.lower()
        
        # Check for form numbers
        form_mentions = 0
        for form, name in self.critical_forms.items():
            if form.lower() in response_lower:
                form_mentions += 1
                # Verify correct form for context
                if self._is_form_appropriate(form, question['question_text']):
                    score *= 1.0
                else:
                    score *= 0.5  # Wrong form is critical error
        
        # Check deadlines
        deadline_correct = True
        for deadline, contexts in self.critical_deadlines.items():
            if deadline in response:
                # Check if deadline is used in appropriate context
                for context in contexts:
                    if context in question['question_text'].lower():
                        if deadline not in response:
                            deadline_correct = False
                            score *= 0.7
        
        # Check Genesee specifics
        if 'genesee' in question['question_text'].lower():
            if self.genesee_specifics['filing_fee'] in response:
                score *= 1.0
            if 'thursday' in response_lower:
                score *= 1.0
            if '900 s. saginaw' in response_lower:
                score *= 1.0
        
        # Penalize if no procedural content when expected
        if 'form' in question['question_text'].lower() and form_mentions == 0:
            score *= 0.3
        
        return min(max(score, 0), 1)
    
    def _score_legal(self, response: str, question: dict) -> float:
        """Score substantive legal accuracy (0-1)"""
        score = 0.8  # Start high, deduct for errors
        
        # Check for legal citations
        mcl_pattern = r'MCL \d+\.\d+'
        mcl_citations = re.findall(mcl_pattern, response)
        
        if mcl_citations:
            score += 0.1 * min(len(mcl_citations), 2)
        
        # Check for key legal concepts
        legal_concepts = [
            'best interests', 'guardian ad litem', 'parental rights',
            'active efforts', 'qualified expert witness', 'burden of proof',
            'clear and convincing', 'preponderance', 'jurisdiction'
        ]
        
        concepts_found = sum(1 for concept in legal_concepts if concept in response.lower())
        score += 0.05 * min(concepts_found, 2)
        
        # ICWA compliance for relevant questions
        if 'indian' in question['question_text'].lower() or 'icwa' in question['question_text'].lower():
            if 'active efforts' in response.lower():
                score += 0.1
            if 'qualified expert' in response.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def _score_actionability(self, response: str, question: dict) -> float:
        """Score actionability (0-1)"""
        actionable_elements = 0
        max_elements = 4
        
        # Check for specific forms
        if re.search(r'PC \d{3}', response):
            actionable_elements += 1
        
        # Check for location/address
        if '900 s. saginaw' in response.lower() or 'courthouse' in response.lower():
            actionable_elements += 1
        
        # Check for timeline/deadlines
        if re.search(r'\d+ days?', response):
            actionable_elements += 1
        
        # Check for next steps
        next_step_phrases = ['next', 'then', 'after', 'following', 'proceed', 'file', 'submit']
        if any(phrase in response.lower() for phrase in next_step_phrases):
            actionable_elements += 1
        
        return actionable_elements / max_elements
    
    def _score_mode(self, response: str, question: dict) -> float:
        """Score mode effectiveness (0-1)"""
        score = 0.7  # Base score
        
        # Check for appropriate balance
        has_legal_facts = bool(re.search(r'MCL \d+\.\d+|PC \d{3}', response))
        has_empathy = any(phrase in response.lower() for phrase in [
            'understand', 'difficult', 'challenging', 'support', 'help'
        ])
        
        complexity = question.get('complexity_tier', 'standard')
        
        if complexity in ['complex', 'crisis']:
            # Should have more empathy
            if has_empathy:
                score += 0.2
            if has_legal_facts and has_empathy:
                score += 0.1
        else:
            # Should be more factual
            if has_legal_facts:
                score += 0.3
        
        return min(score, 1.0)
    
    def _score_caution(self, response: str, question: dict) -> float:
        """Score strategic caution (0-1)"""
        caution_phrases = [
            'if.*object', 'may require', 'court.*discretion', 
            'evidence', 'consider', 'important to note'
        ]
        
        has_caution = any(re.search(phrase, response.lower()) for phrase in caution_phrases)
        
        # Penalize excessive attorney referrals
        attorney_mentions = response.lower().count('attorney') + response.lower().count('lawyer')
        if attorney_mentions > 2:
            return 0.3
        
        return 0.8 if has_caution else 0.5
    
    def _score_citations(self, response: str) -> float:
        """Score citation quality (0-1)"""
        # Count inline citations
        citation_patterns = [
            r'\(MCL \d+\.\d+\)',
            r'\(PC \d{3}\)',
            r'\(.*Form.*\)',
            r'\(.*Rule.*\)'
        ]
        
        total_citations = sum(
            len(re.findall(pattern, response)) 
            for pattern in citation_patterns
        )
        
        # Check if citations are properly placed
        if total_citations > 0:
            # Basic check for inline placement
            if '(' in response and ')' in response:
                return min(0.5 + (0.1 * total_citations), 1.0)
        
        return 0.3 if total_citations > 0 else 0.0
    
    def _score_harm_prevention(self, response: str, question: dict) -> float:
        """Score harm prevention (0-1)"""
        # Start with perfect score, deduct for red flags
        score = 1.0
        
        red_flags = [
            ('wrong jurisdiction', 0.5),
            ('miss.*deadline', 0.3),
            ('guardian shop', 0.5),
            ('minimiz.*icwa', 0.4),
            ('ignore.*tribe', 0.4)
        ]
        
        response_lower = response.lower()
        for flag, penalty in red_flags:
            if re.search(flag, response_lower):
                score -= penalty
        
        return max(score, 0)
    
    def _is_form_appropriate(self, form: str, question_text: str) -> bool:
        """Check if form number is appropriate for the question"""
        form_contexts = {
            'PC 651': ['petition', 'guardianship', 'appointment'],
            'PC 652': ['limited', 'placement plan'],
            'PC 654': ['annual report', 'guardian report'],
            'PC 633': ['letters', 'proof', 'authority'],
            'PC 658': ['condition', 'status'],
            'PC 584': ['account', 'financial', 'conservator'],
            'PC 678': ['tribe', 'indian', 'icwa']
        }
        
        if form not in form_contexts:
            return True  # Unknown form, don't penalize
        
        question_lower = question_text.lower()
        return any(context in question_lower for context in form_contexts[form])


def main():
    """Score Gemma retest results"""
    # Paths
    results_dir = Path("results/gemma_retest/run_20250729_1716")
    rubric_path = Path("rubrics/eval_rubric.yaml")
    
    # Initialize scorer
    scorer = GemmaScorer(rubric_path)
    
    # Process both models
    models = [
        "google_gemma-3n-e4b-it",
        "google_gemma-3-4b-it"
    ]
    
    all_results = {}
    
    for model in models:
        print(f"\nScoring {model}...")
        
        # Load results
        results_path = results_dir / f"{model}_results.json"
        with open(results_path) as f:
            results = json.load(f)
        
        # Score each successful response
        scored_results = []
        for result in results:
            if result['error'] is None and result['response']:
                score_data = scorer.score_response(result, result['response'])
                score_data['question_id'] = result['question_id']
                score_data['complexity'] = result['complexity_tier']
                scored_results.append(score_data)
        
        # Calculate statistics
        if scored_results:
            total_scores = [r['total_score'] for r in scored_results]
            component_scores = defaultdict(list)
            
            for result in scored_results:
                for component, score in result['scores'].items():
                    component_scores[component].append(score)
            
            # Group by complexity
            complexity_scores = defaultdict(list)
            for result in scored_results:
                complexity_scores[result['complexity']].append(result['total_score'])
            
            model_stats = {
                'model': model,
                'total_scored': len(scored_results),
                'avg_total_score': np.mean(total_scores),
                'std_total_score': np.std(total_scores),
                'min_score': np.min(total_scores),
                'max_score': np.max(total_scores),
                'component_averages': {
                    comp: np.mean(scores) for comp, scores in component_scores.items()
                },
                'complexity_averages': {
                    comp: np.mean(scores) for comp, scores in complexity_scores.items()
                },
                'detailed_scores': scored_results
            }
            
            all_results[model] = model_stats
            
            # Print summary
            print(f"  Total responses scored: {len(scored_results)}")
            print(f"  Average total score: {model_stats['avg_total_score']:.2f}/10")
            print(f"  Score range: {model_stats['min_score']:.2f} - {model_stats['max_score']:.2f}")
            print("\n  Component averages:")
            for comp, avg in model_stats['component_averages'].items():
                weight = scorer.rubric[comp]['weight']
                print(f"    {comp}: {avg:.2f} (weight: {weight})")
    
    # Save results
    output_path = results_dir / "quality_scores.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Scoring complete! Results saved to: {output_path}")
    
    # Create markdown summary
    create_scoring_summary(all_results, results_dir)


def create_scoring_summary(results: dict, output_dir: Path):
    """Create a markdown summary of scoring results"""
    summary = """# Gemma Models Quality Score Evaluation

## Overview
This evaluation scores the actual responses from the Gemma retest using the Michigan Guardianship AI evaluation rubric.

## Overall Results

| Model | Avg Score | Responses Scored | Score Range |
|-------|-----------|------------------|-------------|
"""
    
    for model, stats in results.items():
        summary += f"| {model} | **{stats['avg_total_score']:.2f}/10** | {stats['total_scored']} | {stats['min_score']:.2f} - {stats['max_score']:.2f} |\n"
    
    summary += "\n## Component Breakdown\n\n"
    
    # Component weights
    weights = {
        'procedural_accuracy': 2.5,
        'substantive_legal_accuracy': 2.0,
        'actionability': 2.0,
        'mode_effectiveness': 1.5,
        'strategic_caution': 0.5,
        'citation_quality': 0.5,
        'harm_prevention': 0.5
    }
    
    summary += "| Component | Max Points | "
    for model in results.keys():
        summary += f"{model.split('_')[1]} | "
    summary += "\n|-----------|------------|"
    summary += "------------|" * len(results)
    summary += "\n"
    
    for component, weight in weights.items():
        summary += f"| {component.replace('_', ' ').title()} | {weight} | "
        for model, stats in results.items():
            avg = stats['component_averages'][component]
            weighted = avg * weight
            summary += f"{avg:.2f} ({weighted:.2f}) | "
        summary += "\n"
    
    summary += "\n## Performance by Complexity\n\n"
    
    for model, stats in results.items():
        summary += f"\n### {model}\n\n"
        summary += "| Complexity | Questions | Avg Score |\n"
        summary += "|------------|-----------|------------|\n"
        
        for complexity in ['simple', 'standard', 'complex', 'crisis']:
            if complexity in stats['complexity_averages']:
                count = sum(1 for r in stats['detailed_scores'] if r['complexity'] == complexity)
                avg = stats['complexity_averages'][complexity]
                summary += f"| {complexity.capitalize()} | {count} | {avg:.2f} |\n"
    
    summary += "\n## Key Findings\n\n"
    
    # Compare models
    if len(results) == 2:
        models = list(results.keys())
        score_diff = results[models[0]]['avg_total_score'] - results[models[1]]['avg_total_score']
        
        if abs(score_diff) < 0.1:
            summary += "- Both models performed very similarly\n"
        elif score_diff > 0:
            summary += f"- {models[0]} outperformed {models[1]} by {score_diff:.2f} points\n"
        else:
            summary += f"- {models[1]} outperformed {models[0]} by {abs(score_diff):.2f} points\n"
    
    # Component analysis
    summary += "\n### Strongest Components\n"
    for model, stats in results.items():
        top_components = sorted(
            stats['component_averages'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        summary += f"\n**{model}**:\n"
        for comp, score in top_components:
            summary += f"- {comp.replace('_', ' ').title()}: {score:.2f}\n"
    
    summary += "\n### Areas for Improvement\n"
    for model, stats in results.items():
        weak_components = sorted(
            stats['component_averages'].items(), 
            key=lambda x: x[1]
        )[:3]
        summary += f"\n**{model}**:\n"
        for comp, score in weak_components:
            summary += f"- {comp.replace('_', ' ').title()}: {score:.2f}\n"
    
    # Save summary
    summary_path = output_dir / "scoring_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()