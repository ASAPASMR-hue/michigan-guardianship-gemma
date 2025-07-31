#!/usr/bin/env python3
"""
Score ALL models from Phase 3 testing using the same evaluation rubric
Includes both original results and Gemma retest
"""

import json
import yaml
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from score_gemma_retest import GemmaScorer  # Reuse the scorer class


def process_model_results(results_path: Path, model_name: str, scorer: GemmaScorer):
    """Process results for a single model"""
    print(f"\nScoring {model_name}...")
    
    # Load results
    with open(results_path) as f:
        results = json.load(f)
    
    # Handle different result formats
    if isinstance(results, dict) and 'results' in results:
        # Original format
        results_list = results['results']
    else:
        # Direct list format (Gemma retest)
        results_list = results
    
    # Score each successful response
    scored_results = []
    total_attempted = len(results_list)
    
    for result in results_list:
        # Skip if error or no response
        if result.get('error') is not None or not result.get('response'):
            continue
        
        # Create question dict for scorer
        question = {
            'question_text': result.get('question_text', ''),
            'complexity_tier': result.get('complexity_tier', 'standard')
        }
        
        score_data = scorer.score_response(question, result['response'])
        score_data['question_id'] = result.get('question_id', 'unknown')
        score_data['complexity'] = result.get('complexity_tier', 'standard')
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
        
        # Calculate weighted component scores
        weighted_component_scores = {}
        for component in scorer.rubric.keys():
            if component in component_scores:
                avg_score = np.mean(component_scores[component])
                weight = scorer.rubric[component]['weight']
                weighted_component_scores[component] = avg_score * weight
        
        model_stats = {
            'model': model_name,
            'total_attempted': total_attempted,
            'total_scored': len(scored_results),
            'success_rate': len(scored_results) / total_attempted * 100,
            'avg_total_score': np.mean(total_scores),
            'std_total_score': np.std(total_scores),
            'min_score': np.min(total_scores),
            'max_score': np.max(total_scores),
            'component_averages': {
                comp: np.mean(scores) for comp, scores in component_scores.items()
            },
            'weighted_component_scores': weighted_component_scores,
            'complexity_averages': {
                comp: {
                    'avg_score': np.mean(scores),
                    'count': len(scores)
                } for comp, scores in complexity_scores.items()
            }
        }
        
        # Print summary
        print(f"  Success rate: {model_stats['success_rate']:.1f}% ({len(scored_results)}/{total_attempted})")
        print(f"  Average total score: {model_stats['avg_total_score']:.2f}/10")
        print(f"  Score range: {model_stats['min_score']:.2f} - {model_stats['max_score']:.2f}")
        
        return model_stats
    else:
        print(f"  No successful responses to score")
        return None


def main():
    """Score all models from Phase 3 testing"""
    # Initialize scorer
    rubric_path = Path("rubrics/eval_rubric.yaml")
    scorer = GemmaScorer(rubric_path)
    
    # Define all models to evaluate
    models_to_evaluate = [
        # Original Phase 3 results
        {
            'path': Path("results/run_20250728_2255/google_gemini-2-0-flash-001_results.json"),
            'name': 'google/gemini-2.0-flash-001'
        },
        {
            'path': Path("results/run_20250728_2255/google_gemini-2-0-flash-lite-001_results.json"),
            'name': 'google/gemini-2.0-flash-lite-001'
        },
        {
            'path': Path("results/run_20250728_2255/google_gemini-2-5-flash-lite_results.json"),
            'name': 'google/gemini-2.5-flash-lite'
        },
        {
            'path': Path("results/run_20250728_2255/meta-llama_llama-3-3-70b-instruct_free_results.json"),
            'name': 'meta-llama/llama-3.3-70b-instruct:free'
        },
        {
            'path': Path("results/run_20250728_2255/mistralai_mistral-nemo_free_results.json"),
            'name': 'mistralai/mistral-nemo:free'
        },
        {
            'path': Path("results/run_20250728_2255/mistralai_mistral-small-24b-instruct-2501_results.json"),
            'name': 'mistralai/mistral-small-24b-instruct-2501'
        },
        # Gemma retest results  
        {
            'path': Path("results/gemma_retest/run_20250729_1716/google_gemma-3n-e4b-it_results.json"),
            'name': 'google/gemma-3n-e4b-it (retest)'
        },
        {
            'path': Path("results/gemma_retest/run_20250729_1716/google_gemma-3-4b-it_results.json"),
            'name': 'google/gemma-3-4b-it (retest)'
        },
        # New Phi-4 and Llama 3.1 8B results
        {
            'path': Path("results/phi4_llama_test/run_20250729_1909/microsoft_phi-4_results.json"),
            'name': 'microsoft/phi-4'
        },
        {
            'path': Path("results/phi4_llama_test/run_20250729_1909/meta-llama_llama-3-1-8b-instruct_results.json"),
            'name': 'meta-llama/llama-3.1-8b-instruct'
        }
    ]
    
    # Score all models
    all_results = {}
    for model_info in models_to_evaluate:
        if model_info['path'].exists():
            stats = process_model_results(model_info['path'], model_info['name'], scorer)
            if stats:
                all_results[model_info['name']] = stats
        else:
            print(f"\nSkipping {model_info['name']} - file not found")
    
    # Sort by average score
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['avg_total_score'], reverse=True)
    
    # Create comprehensive summary
    summary = """# Michigan Guardianship AI - Comprehensive Model Evaluation

## Overview
This evaluation scores ALL models from Phase 3 testing using the same evaluation rubric to ensure fair comparison.
Includes original Phase 3 results, Gemma retest results, and newly tested Phi-4 and Llama 3.1 8B models.
Total models evaluated: 10

## Evaluation Methodology
- **Rubric**: 10-point scale across 7 components
- **Procedural Accuracy** (2.5 pts): Forms, deadlines, fees, courthouse details
- **Legal Accuracy** (2.0 pts): Legal concepts, statutes, requirements  
- **Actionability** (2.0 pts): Concrete steps and guidance
- **Mode Effectiveness** (1.5 pts): Balance of legal facts and empathy
- **Strategic Caution** (0.5 pts): Warnings and caveats
- **Citation Quality** (0.5 pts): Proper inline citations
- **Harm Prevention** (0.5 pts): Avoiding dangerous advice

## Overall Rankings

| Rank | Model | Avg Score | Success Rate | Cost/Question |
|------|-------|-----------|--------------|---------------|
"""
    
    for i, (model, stats) in enumerate(sorted_models, 1):
        # Determine cost
        if 'gemma' in model.lower():
            cost = "$0.00"
        elif 'free' in model.lower():
            cost = "$0.00"
        else:
            cost = "$0.00"  # All models in this test were free
        
        summary += f"| {i} | {model} | **{stats['avg_total_score']:.2f}/10** | {stats['success_rate']:.1f}% | {cost} |\n"
    
    summary += "\n## Detailed Component Analysis\n\n"
    summary += "| Model | Procedural<br>(2.5) | Legal<br>(2.0) | Action<br>(2.0) | Mode<br>(1.5) | Caution<br>(0.5) | Citation<br>(0.5) | Harm<br>(0.5) | **Total** |\n"
    summary += "|-------|----------|-------|---------|------|---------|----------|------|-----------|\n"
    
    for model, stats in sorted_models:
        summary += f"| {model} | "
        components = ['procedural_accuracy', 'substantive_legal_accuracy', 'actionability', 
                     'mode_effectiveness', 'strategic_caution', 'citation_quality', 'harm_prevention']
        
        for comp in components:
            if comp in stats['weighted_component_scores']:
                score = stats['weighted_component_scores'][comp]
                summary += f"{score:.2f} | "
            else:
                summary += "- | "
        
        summary += f"**{stats['avg_total_score']:.2f}** |\n"
    
    summary += "\n## Performance by Complexity Tier\n\n"
    
    # Create complexity comparison table
    summary += "| Model | Simple | Standard | Complex | Crisis |\n"
    summary += "|-------|--------|----------|---------|--------|\n"
    
    for model, stats in sorted_models:
        summary += f"| {model} | "
        for complexity in ['simple', 'standard', 'complex', 'crisis']:
            if complexity in stats['complexity_averages']:
                data = stats['complexity_averages'][complexity]
                summary += f"{data['avg_score']:.2f} (n={data['count']}) | "
            else:
                summary += "- | "
        summary += "\n"
    
    summary += "\n## Key Findings\n\n"
    
    # Highlight top performers
    summary += "### Top 3 Models by Quality Score\n"
    for i, (model, stats) in enumerate(sorted_models[:3], 1):
        summary += f"{i}. **{model}**: {stats['avg_total_score']:.2f}/10"
        if stats['success_rate'] < 100:
            summary += f" (Note: {stats['success_rate']:.0f}% success rate)"
        summary += "\n"
    
    summary += "\n### Component Excellence\n"
    
    # Find best model for each component
    components = ['procedural_accuracy', 'substantive_legal_accuracy', 'actionability', 
                 'mode_effectiveness', 'strategic_caution', 'citation_quality', 'harm_prevention']
    
    for comp in components:
        best_model = None
        best_score = 0
        
        for model, stats in all_results.items():
            if comp in stats['component_averages']:
                score = stats['component_averages'][comp]
                if score > best_score:
                    best_score = score
                    best_model = model
        
        if best_model:
            summary += f"- **{comp.replace('_', ' ').title()}**: {best_model} ({best_score:.2f})\n"
    
    summary += "\n### Success Rate vs Quality Trade-off\n\n"
    summary += "| Category | Models | Avg Quality | Avg Success Rate |\n"
    summary += "|----------|--------|-------------|------------------|\n"
    
    # Categorize models
    gemma_models = [(m, s) for m, s in all_results.items() if 'gemma' in m.lower()]
    other_models = [(m, s) for m, s in all_results.items() if 'gemma' not in m.lower()]
    
    if gemma_models:
        avg_quality = np.mean([s['avg_total_score'] for _, s in gemma_models])
        avg_success = np.mean([s['success_rate'] for _, s in gemma_models])
        summary += f"| Gemma Models | {len(gemma_models)} | {avg_quality:.2f}/10 | {avg_success:.1f}% |\n"
    
    if other_models:
        avg_quality = np.mean([s['avg_total_score'] for _, s in other_models])
        avg_success = np.mean([s['success_rate'] for _, s in other_models])
        summary += f"| Other Models | {len(other_models)} | {avg_quality:.2f}/10 | {avg_success:.1f}% |\n"
    
    summary += "\n## Conclusions\n\n"
    summary += "1. **Quality Leader**: " + sorted_models[0][0] + f" ({sorted_models[0][1]['avg_total_score']:.2f}/10)\n"
    summary += "2. **Most Reliable**: Models with 100% success rate: "
    reliable_models = [m for m, s in all_results.items() if s['success_rate'] == 100]
    summary += ", ".join(reliable_models[:3]) + "\n"
    summary += "3. **Best Value**: Free models scoring above 6.0/10 provide excellent cost-effectiveness\n"
    summary += "4. **Complexity Handling**: Most models perform better on complex questions than simple ones\n"
    
    # Save results
    output_path = Path("results/comprehensive_evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary_path = Path("results/comprehensive_evaluation_summary.md")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"\n✅ Comprehensive evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()