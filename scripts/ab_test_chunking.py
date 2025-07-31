#!/usr/bin/env python3
"""
ab_test_chunking.py - A/B test pattern-based chunking strategies
Phase 2: Step 4 - Compare chunking approaches
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.log_step import log_step
from scripts.embed_kb import DocumentProcessor, EnhancedDocumentProcessor
from scripts.adaptive_retrieval import AdaptiveHybridRetriever

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "phase2_ab_chunking.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ABChunkingTest:
    """A/B test for chunking strategies"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.kb_dir = self.project_root / "kb_files"
        self.docs_dir = self.project_root / "docs"
        self.test_queries_path = self.project_root / "data" / "synthetic_questions_phase2.csv"
        
    def load_test_queries(self, n=50) -> List[Dict]:
        """Load test queries from synthetic dataset"""
        import pandas as pd
        
        df = pd.read_csv(self.test_queries_path)
        # Filter out out-of-scope queries
        df = df[~df['is_out_of_scope']]
        
        # Sample queries stratified by complexity
        queries = []
        for tier in ['simple', 'standard', 'complex']:
            tier_df = df[df['complexity_tier'] == tier]
            sample_size = min(n // 3, len(tier_df))
            sampled = tier_df.sample(sample_size)
            
            for _, row in sampled.iterrows():
                queries.append({
                    'query': row['question'],
                    'complexity': row['complexity_tier'],
                    'expected_elements': row['expected_elements'].split(';') if pd.notna(row['expected_elements']) else []
                })
        
        logger.info(f"Loaded {len(queries)} test queries")
        return queries[:n]
    
    def run_variant_a(self) -> Dict:
        """Run Variant A: Current semantic chunking"""
        logger.info("\n=== Running Variant A: Standard Semantic Chunking ===")
        
        # Use standard document processor
        processor = DocumentProcessor()
        processor.chunk_config = {
            "size": 1000,
            "overlap": 100,
            "separators": [
                "\n## ",
                "\n### ",
                "\nMCL ",
                "\nPC ",
                "\n§ ",
                "\n- ",
                "\n\n",
            ]
        }
        
        # Process documents
        start_time = time.perf_counter()
        chunks_a = []
        
        # Process each document
        doc_paths = list(self.kb_dir.glob("*.txt")) + list(self.kb_dir.glob("*.pdf"))
        doc_paths.extend(list(self.docs_dir.glob("*.txt")) + list(self.docs_dir.glob("*.md")))
        
        # Fix: Load chunking config from yaml
        import yaml
        with open(self.project_root / "config" / "chunking.yaml", 'r') as f:
            processor.chunk_config = yaml.safe_load(f)
        
        for doc_path in doc_paths[:5]:  # Test with first 5 docs
            try:
                chunks = processor.process_document(str(doc_path))
                chunks_a.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Variant A: Created {len(chunks_a)} chunks in {processing_time:.1f}ms")
        
        # Analyze chunk characteristics
        if chunks_a:
            chunk_lengths = [len(chunk['content'].split()) for chunk in chunks_a]
            stats = {
                'avg_chunk_length': float(np.mean(chunk_lengths)),
                'std_chunk_length': float(np.std(chunk_lengths)),
                'min_chunk_length': int(np.min(chunk_lengths)),
                'max_chunk_length': int(np.max(chunk_lengths))
            }
        else:
            stats = {
                'avg_chunk_length': 0,
                'std_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0
            }
        
        return {
            'variant': 'A',
            'description': 'Standard semantic chunking',
            'chunks': chunks_a,
            'num_chunks': len(chunks_a),
            'processing_time_ms': processing_time,
            **stats
        }
    
    def run_variant_b(self) -> Dict:
        """Run Variant B: Enhanced pattern-based chunking"""
        logger.info("\n=== Running Variant B: Enhanced Pattern-Based Chunking ===")
        
        # Use enhanced processor with preserve_together patterns
        processor = EnhancedDocumentProcessor()
        processor.chunk_config = {
            "size": 1000,
            "overlap": 100,
            "separators": [
                "\n## ",
                "\n### ",
                "\nMCL ",
                "\nPC ",
                "\n§ ",
                "\n- ",
                "\n\n",
            ],
            "preserve_together": [
                r"(Form PC \d+.*?)\n",  # Keep form numbers with descriptions
                r"(MCL \d+\.\d+.*?)\n",  # Keep statutes intact
                r"(\$\d+.*?waiver.*?)\n",  # Keep fees with waiver info
                r"(\d+ days?.*?)\n",  # Keep deadlines together
                r"(Genesee County.*?)\n",  # Keep county-specific info
                r"(ICWA.*?)\n",  # Keep ICWA references together
                r"(Step \d+:.*?)\n",  # Keep process steps
            ]
        }
        
        # Process documents
        start_time = time.perf_counter()
        chunks_b = []
        
        # Process same documents as Variant A
        doc_paths = list(self.kb_dir.glob("*.txt")) + list(self.kb_dir.glob("*.pdf"))
        doc_paths.extend(list(self.docs_dir.glob("*.txt")) + list(self.docs_dir.glob("*.md")))
        
        # Fix: Load base chunking config from yaml
        import yaml
        with open(self.project_root / "config" / "chunking.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
            # Merge with enhanced config
            processor.chunk_config.update(base_config)
        
        for doc_path in doc_paths[:5]:  # Test with first 5 docs
            try:
                chunks = processor.process_document(str(doc_path))
                chunks_b.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Variant B: Created {len(chunks_b)} chunks in {processing_time:.1f}ms")
        
        # Analyze chunk characteristics
        if chunks_b:
            chunk_lengths = [len(chunk['content'].split()) for chunk in chunks_b]
            stats = {
                'avg_chunk_length': float(np.mean(chunk_lengths)),
                'std_chunk_length': float(np.std(chunk_lengths)),
                'min_chunk_length': int(np.min(chunk_lengths)),
                'max_chunk_length': int(np.max(chunk_lengths))
            }
        else:
            stats = {
                'avg_chunk_length': 0,
                'std_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0
            }
        
        return {
            'variant': 'B',
            'description': 'Enhanced pattern-based chunking',
            'chunks': chunks_b,
            'num_chunks': len(chunks_b),
            'processing_time_ms': processing_time,
            **stats
        }
    
    def evaluate_retrieval_quality(self, chunks: List[Dict], queries: List[Dict]) -> Dict:
        """Evaluate retrieval quality for a chunking variant"""
        logger.info(f"\nEvaluating retrieval quality with {len(chunks)} chunks...")
        
        # Create temporary collection for testing
        from chromadb import Client
        from sentence_transformers import SentenceTransformer
        
        client = Client()
        collection_name = f"test_chunking_{int(time.time())}"
        collection = client.create_collection(collection_name)
        
        # Embed chunks
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([c['content'] for c in chunks])
        
        # Add to collection
        collection.add(
            documents=[c['content'] for c in chunks],
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{'source': c.get('source', 'unknown')} for c in chunks]
        )
        
        # Evaluate queries
        metrics = {
            'recall_at_5': [],
            'recall_at_10': [],
            'mrr': [],  # Mean Reciprocal Rank
            'relevance_scores': []
        }
        
        for query_info in queries[:20]:  # Test with 20 queries
            query = query_info['query']
            
            # Retrieve
            query_embedding = model.encode(query)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=10
            )
            
            # Simple relevance check based on expected elements
            relevant_found = 0
            first_relevant_rank = None
            
            for i, doc in enumerate(results['documents'][0]):
                doc_lower = doc.lower()
                
                # Check for relevance based on query keywords
                query_keywords = query.lower().split()
                relevance_score = sum(1 for kw in query_keywords if kw in doc_lower) / len(query_keywords)
                
                if relevance_score > 0.3:  # 30% keyword match threshold
                    relevant_found += 1
                    if first_relevant_rank is None:
                        first_relevant_rank = i + 1
                
                if i < 5:
                    metrics['recall_at_5'].append(1 if relevant_found > 0 else 0)
                
                metrics['relevance_scores'].append(relevance_score)
            
            metrics['recall_at_10'].append(1 if relevant_found > 0 else 0)
            metrics['mrr'].append(1.0 / first_relevant_rank if first_relevant_rank else 0)
        
        # Clean up
        client.delete_collection(collection_name)
        
        # Calculate averages
        return {
            'avg_recall_at_5': np.mean(metrics['recall_at_5']) if metrics['recall_at_5'] else 0,
            'avg_recall_at_10': np.mean(metrics['recall_at_10']) if metrics['recall_at_10'] else 0,
            'avg_mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0,
            'avg_relevance': np.mean(metrics['relevance_scores']) if metrics['relevance_scores'] else 0
        }
    
    def run_ab_test(self):
        """Run complete A/B test"""
        log_step("Starting A/B chunking test",
                "Comparing standard vs pattern-based chunking",
                "Phase 2 Step 4: A/B test pattern-based chunking")
        
        # Load test queries
        queries = self.load_test_queries(50)
        
        # Run both variants
        variant_a = self.run_variant_a()
        variant_b = self.run_variant_b()
        
        # Evaluate retrieval quality
        logger.info("\n=== Evaluating Retrieval Quality ===")
        
        quality_a = self.evaluate_retrieval_quality(variant_a['chunks'], queries)
        quality_b = self.evaluate_retrieval_quality(variant_b['chunks'], queries)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'variant_a': {
                **{k: v for k, v in variant_a.items() if k != 'chunks'},
                'quality_metrics': quality_a
            },
            'variant_b': {
                **{k: v for k, v in variant_b.items() if k != 'chunks'},
                'quality_metrics': quality_b
            },
            'comparison': {
                'chunks_difference': variant_b['num_chunks'] - variant_a['num_chunks'],
                'chunks_difference_pct': ((variant_b['num_chunks'] - variant_a['num_chunks']) / variant_a['num_chunks'] * 100) if variant_a['num_chunks'] > 0 else 0,
                'processing_time_difference': variant_b['processing_time_ms'] - variant_a['processing_time_ms'],
                'recall_at_5_improvement': quality_b['avg_recall_at_5'] - quality_a['avg_recall_at_5'],
                'recall_at_10_improvement': quality_b['avg_recall_at_10'] - quality_a['avg_recall_at_10'],
                'mrr_improvement': quality_b['avg_mrr'] - quality_a['avg_mrr'],
                'relevance_improvement': quality_b['avg_relevance'] - quality_a['avg_relevance']
            }
        }
        
        # Determine winner
        precision_improvement = results['comparison']['recall_at_5_improvement']
        latency_increase_pct = (results['comparison']['processing_time_difference'] / variant_a['processing_time_ms'] * 100) if variant_a['processing_time_ms'] > 0 else 0
        
        if precision_improvement > 0.05 and latency_increase_pct < 20:
            results['winner'] = 'Variant B'
            results['reason'] = f">{5}% precision improvement with <20% latency increase"
        else:
            results['winner'] = 'Variant A'
            results['reason'] = "Insufficient improvement or excessive latency"
        
        # Save results
        results_path = LOG_DIR / "phase2_ab_chunking_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log summary
        logger.info("\n=== A/B Test Results ===")
        logger.info(f"Variant A: {variant_a['num_chunks']} chunks, {quality_a['avg_recall_at_5']:.3f} recall@5")
        logger.info(f"Variant B: {variant_b['num_chunks']} chunks, {quality_b['avg_recall_at_5']:.3f} recall@5")
        logger.info(f"Winner: {results['winner']} ({results['reason']})")
        logger.info(f"Full results saved to: {results_path}")
        
        log_step("A/B chunking test complete",
                f"Winner: {results['winner']} - {results['reason']}",
                "Phase 2 Step 4 complete")
        
        return results

def main():
    """Main function"""
    tester = ABChunkingTest()
    tester.run_ab_test()

if __name__ == "__main__":
    main()