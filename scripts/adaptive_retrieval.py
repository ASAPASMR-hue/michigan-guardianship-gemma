#!/usr/bin/env python3
"""
adaptive_retrieval.py - Implements adaptive retrieval with latency budgets
Phase 2: Step 3 - Adaptive top-k with latency monitoring
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.retrieval_setup import HybridRetriever
from scripts.log_step import log_step

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "phase2_latency.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AdaptiveHybridRetriever(HybridRetriever):
    """Enhanced retriever with latency monitoring and adaptive fallbacks"""
    
    def __init__(self):
        super().__init__()
        self.latency_logs = []
    
    def retrieve_with_latency(self, query: str) -> Tuple[List[Dict], Dict]:
        """Retrieve with detailed latency tracking and adaptive fallbacks"""
        start_time = time.perf_counter()
        
        # Track individual stages
        stage_times = {}
        
        # 1. Classification stage
        classify_start = time.perf_counter()
        complexity = self.classifier.classify(query)
        confidence = getattr(self.classifier, 'confidence', 0.7)
        params = self.classifier.get_params(complexity)
        stage_times['classification_ms'] = (time.perf_counter() - classify_start) * 1000
        
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Complexity: {complexity} (confidence: {confidence:.2f})")
        logger.info(f"Initial params: top_k={params['top_k']}, rewrites={params['query_rewrites']}")
        logger.info(f"Latency budget: {params['latency_budget_ms']}ms")
        
        # Check if we've already exceeded budget
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > params['latency_budget_ms'] * 0.5:
            # Use fallback params if we're already slow
            if 'fallback_if_slow' in params:
                logger.warning(f"Already at {elapsed_ms:.0f}ms, using fallback params")
                params['top_k'] = params['fallback_if_slow']['top_k']
                params['query_rewrites'] = params['fallback_if_slow']['query_rewrites']
        
        # 2. Query rewriting stage
        rewrite_start = time.perf_counter()
        rewrites = self.generate_query_rewrites(query, params['query_rewrites'])
        stage_times['rewriting_ms'] = (time.perf_counter() - rewrite_start) * 1000
        
        # 3. Embedding stage (for vector search)
        embed_start = time.perf_counter()
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        stage_times['embedding_ms'] = (time.perf_counter() - embed_start) * 1000
        
        # 4. Retrieval stage
        retrieval_start = time.perf_counter()
        all_results = []
        
        for i, rewrite in enumerate(rewrites):
            # Check remaining budget
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            remaining_budget = params['latency_budget_ms'] - elapsed_ms
            
            if remaining_budget < 100:  # Skip if less than 100ms remaining
                logger.warning(f"Skipping rewrite {i+1}/{len(rewrites)} due to latency budget")
                break
            
            results = self.hybrid_search(rewrite, params['top_k'])
            all_results.extend(results)
        
        stage_times['retrieval_ms'] = (time.perf_counter() - retrieval_start) * 1000
        
        # 5. Deduplication
        dedup_start = time.perf_counter()
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        stage_times['deduplication_ms'] = (time.perf_counter() - dedup_start) * 1000
        
        # 6. Reranking stage
        rerank_start = time.perf_counter()
        
        # Check if we have budget for reranking
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > params['latency_budget_ms'] * 0.9:
            logger.warning(f"Skipping reranking due to latency budget ({elapsed_ms:.0f}ms elapsed)")
            final_results = unique_results[:params['rerank_top_k']]
        else:
            final_results = self.rerank(query, unique_results, params['rerank_top_k'])
        
        stage_times['reranking_ms'] = (time.perf_counter() - rerank_start) * 1000
        
        # Total time
        total_ms = (time.perf_counter() - start_time) * 1000
        stage_times['total_ms'] = total_ms
        
        # Log latency breakdown
        logger.info(f"\nLatency breakdown:")
        for stage, ms in stage_times.items():
            if stage != 'total_ms':
                logger.info(f"  {stage}: {ms:.1f}ms")
        logger.info(f"  TOTAL: {total_ms:.1f}ms (budget: {params['latency_budget_ms']}ms)")
        
        # Check if we met budget
        met_budget = total_ms <= params['latency_budget_ms']
        met_p95 = total_ms <= params['latency_p95_ms']
        
        if not met_budget:
            logger.warning(f"⚠️ Exceeded latency budget by {total_ms - params['latency_budget_ms']:.1f}ms")
        else:
            logger.info(f"✓ Met latency budget with {params['latency_budget_ms'] - total_ms:.1f}ms to spare")
        
        # Prepare metadata
        metadata = {
            'complexity': complexity,
            'complexity_confidence': confidence,
            'num_rewrites': len(rewrites),
            'params': params,
            'latency': stage_times,
            'met_budget': met_budget,
            'met_p95': met_p95
        }
        
        # Store for analysis
        self.latency_logs.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            **metadata
        })
        
        return final_results, metadata
    
    def save_latency_report(self, filepath: str):
        """Save latency analysis report"""
        if not self.latency_logs:
            logger.warning("No latency logs to save")
            return
        
        # Analyze by complexity tier
        analysis = {
            'simple': {'queries': [], 'latencies': []},
            'standard': {'queries': [], 'latencies': []},
            'complex': {'queries': [], 'latencies': []}
        }
        
        for log in self.latency_logs:
            tier = log['complexity']
            analysis[tier]['queries'].append(log['query'])
            analysis[tier]['latencies'].append(log['latency']['total_ms'])
        
        # Calculate statistics
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(self.latency_logs),
            'by_complexity': {}
        }
        
        for tier, data in analysis.items():
            if data['latencies']:
                latencies = data['latencies']
                report['by_complexity'][tier] = {
                    'count': len(latencies),
                    'mean_ms': sum(latencies) / len(latencies),
                    'min_ms': min(latencies),
                    'max_ms': max(latencies),
                    'p50_ms': sorted(latencies)[len(latencies)//2],
                    'p95_ms': sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[0],
                    'budget_ms': self.classifier.complexity_tiers[tier]['latency_budget_ms'],
                    'p95_target_ms': self.classifier.complexity_tiers[tier]['latency_p95_ms'],
                    'met_budget_pct': sum(1 for log in self.latency_logs 
                                         if log['complexity'] == tier and log['met_budget']) / len(latencies) * 100,
                    'met_p95_pct': sum(1 for log in self.latency_logs 
                                      if log['complexity'] == tier and log['met_p95']) / len(latencies) * 100
                }
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nLatency report saved to {filepath}")
        logger.info(f"Summary:")
        for tier, stats in report['by_complexity'].items():
            if stats:
                logger.info(f"\n{tier.upper()}:")
                logger.info(f"  Mean: {stats['mean_ms']:.1f}ms (budget: {stats['budget_ms']}ms)")
                logger.info(f"  P95: {stats['p95_ms']:.1f}ms (target: {stats['p95_target_ms']}ms)")
                logger.info(f"  Met budget: {stats['met_budget_pct']:.1f}%")
                logger.info(f"  Met P95: {stats['met_p95_pct']:.1f}%")

def test_adaptive_retrieval():
    """Test adaptive retrieval with various queries"""
    log_step("Testing adaptive retrieval", 
            "Running 100 queries with latency monitoring",
            "Phase 2 Step 3: Implement adaptive top-k with latency budgets")
    
    retriever = AdaptiveHybridRetriever()
    
    # Test queries from synthetic dataset
    test_queries = [
        # Simple queries
        "What's the filing fee?",
        "Where is the court?", 
        "What form do I need?",
        "How many days before hearing?",
        "Court address?",
        "Filing deadline?",
        "PC 651 or PC 650?",
        "Fee waiver form?",
        "Thursday hearings?",
        "How much to file?",
        
        # Standard queries
        "Grandparent wants guardianship of my child",
        "How to terminate guardianship?",
        "Parent doesn't consent to guardianship",
        "Modify existing guardianship",
        "Guardian wants to move out of state",
        "Add co-guardian to arrangement",
        "Requirements for aunt to be guardian",
        "Background check requirements",
        "Guardian relocation process",
        "Parents want guardianship back",
        
        # Complex queries
        "ICWA emergency guardianship tribal member",
        "CPS involved need immediate placement",
        "Multiple family members contesting guardianship",
        "Out of state parent emergency situation",
        "Special needs child guardian hospitalized",
        "Interstate guardianship transfer urgent",
        "Parent arrested child needs care now",
        "Tribal jurisdiction emergency placement",
        "Guardian died need immediate care",
        "ICWA applies parent out of state contested"
    ]
    
    # Add more queries to reach 100
    import random
    all_queries = test_queries * 3  # 90 queries
    all_queries.extend([
        f"Question about {topic} #{i}" 
        for i, topic in enumerate(['filing', 'forms', 'deadlines', 'process', 'requirements', 
                                  'guardianship', 'emergency', 'modification', 'termination', 'ICWA'])
    ])
    
    logger.info(f"\nTesting {len(all_queries)} queries...")
    
    for i, query in enumerate(all_queries[:100]):
        logger.info(f"\n{'='*60}")
        logger.info(f"Query {i+1}/100")
        
        try:
            results, metadata = retriever.retrieve_with_latency(query)
            logger.info(f"Retrieved {len(results)} results")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    
    # Save latency report
    report_path = LOG_DIR / "phase2_latency_report.json"
    retriever.save_latency_report(str(report_path))
    
    log_step("Adaptive retrieval testing complete",
            f"Processed 100 queries, report saved to {report_path}",
            "Phase 2 Step 3 complete")

def main():
    """Main function"""
    test_adaptive_retrieval()

if __name__ == "__main__":
    main()