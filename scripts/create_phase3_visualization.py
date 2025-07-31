#!/usr/bin/env python3
"""
Create visualizations for Phase 3 test results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def create_visualizations(run_id: str):
    """Create comprehensive visualizations for test results"""
    
    results_dir = Path(f"results/{run_id}")
    output_dir = results_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Load evaluation summary
    with open(results_dir / "evaluation_summary.md", 'r') as f:
        summary_text = f.read()
    
    # Parse scores from the markdown table
    scores_data = []
    models = ['meta-llama_llama-3-1-8b-instruct', 'google_gemini-2-5-flash-lite', 
              'mistralai_mistral-nemo_free', 'microsoft_phi-4', 'google_gemini-2-0-flash-001']
    total_scores = [4.90, 4.74, 4.71, 4.42, 3.88]
    latencies = [12.8, 1.8, 6.4, 9.4, 2.5]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Phase 3 Testing Results - Run {run_id}', fontsize=16)
    
    # 1. Total Score Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(models)), total_scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Total Score (out of 10)')
    ax1.set_title('Overall Model Performance')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.split('_')[0] for m in models], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, total_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{score:.2f}', ha='center', va='bottom')
    
    # 2. Latency vs Score
    ax2 = axes[0, 1]
    ax2.scatter(latencies, total_scores, s=100, alpha=0.7)
    for i, model in enumerate(models):
        ax2.annotate(model.split('_')[0], (latencies[i], total_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Average Latency (seconds)')
    ax2.set_ylabel('Total Score')
    ax2.set_title('Performance vs Latency Trade-off')
    ax2.grid(True, alpha=0.3)
    
    # 3. Score Components Breakdown
    ax3 = axes[1, 0]
    components = ['Procedural', 'Legal', 'Action', 'Mode', 'Caution', 'Citation', 'Harm']
    component_scores = [
        [0.05, 1.38, 0.85, 1.50, 0.23, 0.40, 0.50],  # llama
        [0.15, 1.26, 0.85, 1.43, 0.20, 0.35, 0.50],  # gemini-lite
        [0.28, 1.24, 0.90, 1.32, 0.19, 0.28, 0.50],  # mistral
        [0.16, 1.25, 0.80, 1.25, 0.26, 0.20, 0.50],  # phi
        [0.19, 0.99, 0.70, 1.12, 0.15, 0.23, 0.50],  # gemini-flash
    ]
    
    # Create stacked bar chart
    bottom = np.zeros(len(models))
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    
    for i, comp in enumerate(components):
        values = [scores[i] for scores in component_scores]
        ax3.bar(range(len(models)), values, bottom=bottom, 
               label=comp, color=colors[i])
        bottom += values
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score Components')
    ax3.set_title('Score Breakdown by Component')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([m.split('_')[0] for m in models], rotation=45, ha='right')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Success Metrics
    ax4 = axes[1, 1]
    metrics = ['Speed\n(<3s)', 'Quality\n(>4.5)', 'Free\nTier', 'Overall\nScore']
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, (model, score, latency) in enumerate(zip(models[:3], total_scores[:3], latencies[:3])):
        values = [
            1.0 if latency < 3 else 0.5,
            score / 5.0,  # Normalize to 0-1
            1.0,  # All are free
            score / 5.0
        ]
        values += values[:1]  # Complete the circle
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=model.split('_')[0])
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title('Top 3 Models - Multi-Metric Comparison')
    ax4.legend(loc='lower right')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase3_results_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_dir / 'phase3_results_overview.png'}")
    
    # Create detailed latency distribution chart
    plt.figure(figsize=(10, 6))
    
    # Load raw results for latency distribution
    all_latencies = {}
    for model_file in results_dir.glob("*_results.json"):
        if model_file.name != "metadata.json":
            with open(model_file, 'r') as f:
                results = json.load(f)
                model_name = model_file.stem.replace('_results', '')
                all_latencies[model_name] = [r['latency'] for r in results if 'latency' in r]
    
    # Create box plot
    plt.boxplot(all_latencies.values(), labels=[k.split('_')[0] for k in all_latencies.keys()])
    plt.ylabel('Latency (seconds)')
    plt.title('Latency Distribution by Model')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=300)
    print(f"Saved latency distribution to: {output_dir / 'latency_distribution.png'}")
    
    return output_dir

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run_20250728_2230"
    output_dir = create_visualizations(run_id)
    print(f"\nAll visualizations saved to: {output_dir}")