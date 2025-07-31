#!/usr/bin/env python3
"""
ragas_evaluation.py - Add RAGAS metrics to evaluation pipeline
Phase 2: Step 5 - Integrate RAGAS (Retrieval Augmented Generation Assessment)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.log_step import log_step
from scripts.adaptive_retrieval import AdaptiveHybridRetriever

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "phase2_ragas.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    """
    Implements RAGAS metrics for RAG evaluation
    Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_data_path = self.project_root / "guardianship_qa_cleaned - rubric_determining-2.csv"
        self.retriever = AdaptiveHybridRetriever()
        
    def load_test_data(self, n_samples=50) -> List[Dict]:
        """Load test questions with ground truth answers"""
        df = pd.read_csv(self.test_data_path)
        
        # Sample stratified by complexity if available
        if 'complexity' in df.columns:
            samples = []
            for complexity in df['complexity'].unique():
                complexity_df = df[df['complexity'] == complexity]
                n_tier = min(n_samples // 3, len(complexity_df))
                samples.append(complexity_df.sample(n_tier))
            df_sample = pd.concat(samples)
        else:
            df_sample = df.sample(min(n_samples, len(df)))
        
        test_data = []
        for _, row in df_sample.iterrows():
            test_data.append({
                'question': row['question'],
                'ground_truth': row.get('Ideal Answer', ''),
                'complexity': row.get('complexity', 'standard')
            })
        
        logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate faithfulness: How well the answer is grounded in retrieved contexts
        Simple implementation using keyword overlap
        """
        if not answer or not contexts:
            return 0.0
        
        answer_words = set(answer.lower().split())
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        answer_words -= stop_words
        context_words -= stop_words
        
        if not answer_words:
            return 1.0  # If only stop words, consider it faithful
        
        # Calculate overlap
        overlap = answer_words.intersection(context_words)
        faithfulness = len(overlap) / len(answer_words)
        
        return faithfulness
    
    def calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Calculate answer relevancy: How relevant the answer is to the question
        Simple implementation using keyword overlap
        """
        if not answer:
            return 0.0
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Key question words (remove question marks)
        question_words = {w.strip('?') for w in question_words}
        
        # Check for key concepts
        relevancy_score = 0.0
        
        # Check if answer addresses key question words
        key_words = {'what', 'how', 'when', 'where', 'who', 'which', 'why'}
        question_keys = question_words.intersection(key_words)
        
        if question_keys:
            # Check if answer provides relevant information
            if 'what' in question_keys and any(w in answer_words for w in ['form', 'pc', 'document', 'need']):
                relevancy_score += 0.3
            if 'how' in question_keys and any(w in answer_words for w in ['process', 'step', 'file', 'submit']):
                relevancy_score += 0.3
            if 'when' in question_keys and any(w in answer_words for w in ['days', 'deadline', 'time', 'before']):
                relevancy_score += 0.3
            if 'where' in question_keys and any(w in answer_words for w in ['court', 'address', 'room', 'genesee']):
                relevancy_score += 0.3
        
        # General keyword overlap
        overlap = question_words.intersection(answer_words)
        if len(question_words) > 0:
            relevancy_score += 0.4 * (len(overlap) / len(question_words))
        
        return min(relevancy_score, 1.0)
    
    def calculate_context_precision(self, question: str, contexts: List[str], k: int = 5) -> float:
        """
        Calculate context precision: Precision of retrieved contexts at k
        Measures if all retrieved contexts are relevant
        """
        if not contexts:
            return 0.0
        
        question_words = set(question.lower().split())
        relevant_contexts = 0
        
        for i, context in enumerate(contexts[:k]):
            context_words = set(context.lower().split())
            overlap = question_words.intersection(context_words)
            
            # Consider context relevant if it has significant overlap
            if len(overlap) >= 2 or any(kw in context.lower() for kw in ['genesee', 'guardianship', 'minor']):
                relevant_contexts += 1
        
        return relevant_contexts / min(k, len(contexts))
    
    def calculate_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Calculate context recall: How much of ground truth is covered by contexts
        """
        if not ground_truth or not contexts:
            return 0.0
        
        truth_words = set(ground_truth.lower().split())
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        truth_words -= stop_words
        
        if not truth_words:
            return 1.0
        
        # Calculate coverage
        covered = truth_words.intersection(context_words)
        recall = len(covered) / len(truth_words)
        
        return recall
    
    def generate_simple_answer(self, question: str, contexts: List[str]) -> str:
        """
        Generate a simple answer based on retrieved contexts
        This simulates what an LLM would do
        """
        if not contexts:
            return "I don't have enough information to answer this question."
        
        # Simple rule-based answer generation
        question_lower = question.lower()
        answer_parts = []
        
        # Look for specific patterns in contexts
        for context in contexts[:3]:  # Use top 3 contexts
            context_lower = context.lower()
            
            # Filing fee questions
            if 'filing fee' in question_lower or 'cost' in question_lower:
                if '$175' in context:
                    answer_parts.append("The filing fee is $175")
                if 'mc 20' in context_lower:
                    answer_parts.append("Fee waiver available with Form MC 20")
            
            # Form questions
            elif 'form' in question_lower or 'pc' in question_lower:
                if 'pc 651' in context_lower:
                    answer_parts.append("Use Form PC 651 for full guardianship")
                if 'pc 650' in context_lower:
                    answer_parts.append("Use Form PC 650 for limited guardianship")
            
            # Location questions
            elif 'where' in question_lower or 'address' in question_lower:
                if '900 s. saginaw' in context_lower:
                    answer_parts.append("Genesee County Probate Court: 900 S. Saginaw St., Room 502, Flint, MI 48502")
            
            # Timing questions
            elif 'when' in question_lower or 'deadline' in question_lower:
                if 'thursday' in context_lower:
                    answer_parts.append("Hearings are held on Thursdays")
                if '7 days' in context_lower:
                    answer_parts.append("Personal service required 7 days before hearing")
                if '14 days' in context_lower:
                    answer_parts.append("Mail service required 14 days before hearing")
        
        # Default to summarizing first context if no specific answer
        if not answer_parts:
            first_context = contexts[0][:200]
            answer_parts.append(f"Based on the information: {first_context}...")
        
        return " ".join(answer_parts[:2])  # Limit to 2 parts
    
    def evaluate_single_query(self, test_item: Dict) -> Dict:
        """Evaluate a single query with all RAGAS metrics"""
        question = test_item['question']
        ground_truth = test_item.get('ground_truth', '')
        
        # Retrieve contexts
        try:
            results, metadata = self.retriever.retrieve_with_latency(question)
            contexts = [r['document'] for r in results]
            
            # Generate answer (simulated)
            answer = self.generate_simple_answer(question, contexts)
            
            # Calculate metrics
            metrics = {
                'faithfulness': self.calculate_faithfulness(answer, contexts),
                'answer_relevancy': self.calculate_answer_relevancy(question, answer),
                'context_precision': self.calculate_context_precision(question, contexts),
                'context_recall': self.calculate_context_recall(ground_truth, contexts) if ground_truth else 0.0,
                'retrieval_latency_ms': metadata['latency']['total_ms'],
                'num_contexts': len(contexts),
                'complexity': metadata['complexity']
            }
            
            return {
                'question': question,
                'answer': answer,
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query '{question}': {e}")
            return {
                'question': question,
                'answer': '',
                'metrics': {
                    'faithfulness': 0.0,
                    'answer_relevancy': 0.0,
                    'context_precision': 0.0,
                    'context_recall': 0.0,
                    'retrieval_latency_ms': 0.0,
                    'num_contexts': 0,
                    'complexity': 'unknown'
                },
                'success': False
            }
    
    def run_evaluation(self, n_samples: int = 50):
        """Run RAGAS evaluation on test samples"""
        log_step("Starting RAGAS evaluation",
                f"Evaluating {n_samples} queries with RAGAS metrics",
                "Phase 2 Step 5: Add RAGAS metrics to evaluation")
        
        # Load test data
        test_data = self.load_test_data(n_samples)
        
        # Evaluate each query
        results = []
        for i, test_item in enumerate(test_data):
            logger.info(f"\nEvaluating query {i+1}/{len(test_data)}")
            result = self.evaluate_single_query(test_item)
            results.append(result)
        
        # Aggregate metrics
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            aggregate_metrics = {
                'faithfulness': {
                    'mean': np.mean([r['metrics']['faithfulness'] for r in successful_results]),
                    'std': np.std([r['metrics']['faithfulness'] for r in successful_results]),
                    'min': np.min([r['metrics']['faithfulness'] for r in successful_results]),
                    'max': np.max([r['metrics']['faithfulness'] for r in successful_results])
                },
                'answer_relevancy': {
                    'mean': np.mean([r['metrics']['answer_relevancy'] for r in successful_results]),
                    'std': np.std([r['metrics']['answer_relevancy'] for r in successful_results]),
                    'min': np.min([r['metrics']['answer_relevancy'] for r in successful_results]),
                    'max': np.max([r['metrics']['answer_relevancy'] for r in successful_results])
                },
                'context_precision': {
                    'mean': np.mean([r['metrics']['context_precision'] for r in successful_results]),
                    'std': np.std([r['metrics']['context_precision'] for r in successful_results]),
                    'min': np.min([r['metrics']['context_precision'] for r in successful_results]),
                    'max': np.max([r['metrics']['context_precision'] for r in successful_results])
                },
                'context_recall': {
                    'mean': np.mean([r['metrics']['context_recall'] for r in successful_results if r['metrics']['context_recall'] > 0]),
                    'std': np.std([r['metrics']['context_recall'] for r in successful_results if r['metrics']['context_recall'] > 0]),
                    'min': np.min([r['metrics']['context_recall'] for r in successful_results if r['metrics']['context_recall'] > 0]),
                    'max': np.max([r['metrics']['context_recall'] for r in successful_results if r['metrics']['context_recall'] > 0])
                } if any(r['metrics']['context_recall'] > 0 for r in successful_results) else None,
                'avg_latency_ms': np.mean([r['metrics']['retrieval_latency_ms'] for r in successful_results]),
                'success_rate': len(successful_results) / len(results)
            }
            
            # By complexity
            complexity_breakdown = {}
            for complexity in ['simple', 'standard', 'complex']:
                complexity_results = [r for r in successful_results if r['metrics']['complexity'] == complexity]
                if complexity_results:
                    complexity_breakdown[complexity] = {
                        'count': len(complexity_results),
                        'faithfulness': np.mean([r['metrics']['faithfulness'] for r in complexity_results]),
                        'answer_relevancy': np.mean([r['metrics']['answer_relevancy'] for r in complexity_results]),
                        'context_precision': np.mean([r['metrics']['context_precision'] for r in complexity_results]),
                        'avg_latency_ms': np.mean([r['metrics']['retrieval_latency_ms'] for r in complexity_results])
                    }
            
            # Save results
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(test_data),
                'successful_queries': len(successful_results),
                'aggregate_metrics': aggregate_metrics,
                'complexity_breakdown': complexity_breakdown,
                'sample_results': results[:10]  # Include first 10 for inspection
            }
            
            report_path = LOG_DIR / "phase2_ragas_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Log summary
            logger.info("\n=== RAGAS Evaluation Summary ===")
            logger.info(f"Total queries: {len(test_data)}")
            logger.info(f"Success rate: {aggregate_metrics['success_rate']:.1%}")
            logger.info(f"\nAggregate Metrics:")
            logger.info(f"  Faithfulness: {aggregate_metrics['faithfulness']['mean']:.3f} (±{aggregate_metrics['faithfulness']['std']:.3f})")
            logger.info(f"  Answer Relevancy: {aggregate_metrics['answer_relevancy']['mean']:.3f} (±{aggregate_metrics['answer_relevancy']['std']:.3f})")
            logger.info(f"  Context Precision: {aggregate_metrics['context_precision']['mean']:.3f} (±{aggregate_metrics['context_precision']['std']:.3f})")
            if aggregate_metrics['context_recall']:
                logger.info(f"  Context Recall: {aggregate_metrics['context_recall']['mean']:.3f} (±{aggregate_metrics['context_recall']['std']:.3f})")
            logger.info(f"  Avg Latency: {aggregate_metrics['avg_latency_ms']:.1f}ms")
            
            logger.info(f"\nFull report saved to: {report_path}")
            
        else:
            logger.error("No successful evaluations completed")
        
        log_step("RAGAS evaluation complete",
                f"Evaluated {len(successful_results)}/{len(test_data)} queries successfully",
                "Phase 2 Step 5 complete")

def main():
    """Main function"""
    evaluator = RAGASEvaluator()
    evaluator.run_evaluation(n_samples=50)

if __name__ == "__main__":
    main()