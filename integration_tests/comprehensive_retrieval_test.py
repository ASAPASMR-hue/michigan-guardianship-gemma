#!/usr/bin/env python3
"""
comprehensive_retrieval_test.py - Comprehensive testing of the retrieval system
Tests all aspects of the retrieval component including accuracy, query types, parameters, performance, and edge cases
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.retrieval_setup import HybridRetriever
from scripts.adaptive_retrieval import AdaptiveHybridRetriever

# Test configuration
TEST_DIR = Path(__file__).parent
RESULTS_DIR = TEST_DIR / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

class ComprehensiveRetrievalTester:
    """Comprehensive testing of the retrieval system"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "retrieval_accuracy": {},
            "query_types": {},
            "parameter_testing": {},
            "performance_metrics": {},
            "edge_cases": {},
            "summary": {}
        }
        
        # Comprehensive test queries
        self.test_queries = {
            "simple_factual": [
                {
                    "query": "What is the filing fee for guardianship in Genesee County?",
                    "expected_content": ["$175", "fee", "Genesee"],
                    "relevance_threshold": 0.7
                },
                {
                    "query": "Where is the Genesee County Probate Court?",
                    "expected_content": ["900 S. Saginaw", "Flint", "48502"],
                    "relevance_threshold": 0.8
                },
                {
                    "query": "What day are hearings?",
                    "expected_content": ["Thursday"],
                    "relevance_threshold": 0.6
                },
                {
                    "query": "Court phone number?",
                    "expected_content": ["court", "contact"],
                    "relevance_threshold": 0.5
                }
            ],
            "form_related": [
                {
                    "query": "What forms do I need for minor guardianship?",
                    "expected_content": ["PC 651", "PC 652", "forms"],
                    "relevance_threshold": 0.7
                },
                {
                    "query": "Where can I get form PC 651?",
                    "expected_content": ["PC 651", "form", "petition"],
                    "relevance_threshold": 0.6
                },
                {
                    "query": "Do I need form MC 20?",
                    "expected_content": ["MC 20", "fee waiver"],
                    "relevance_threshold": 0.7
                },
                {
                    "query": "List all required forms",
                    "expected_content": ["forms", "required", "PC"],
                    "relevance_threshold": 0.6
                }
            ],
            "procedural": [
                {
                    "query": "How do I notify parents about the hearing?",
                    "expected_content": ["notify", "parents", "14 days"],
                    "relevance_threshold": 0.7
                },
                {
                    "query": "What are the steps to file for guardianship?",
                    "expected_content": ["steps", "file", "petition"],
                    "relevance_threshold": 0.7
                },
                {
                    "query": "How long before the hearing must I serve notice?",
                    "expected_content": ["14 days", "notice", "hearing"],
                    "relevance_threshold": 0.8
                },
                {
                    "query": "Who must be notified?",
                    "expected_content": ["notify", "parents", "interested parties"],
                    "relevance_threshold": 0.7
                }
            ],
            "complex": [
                {
                    "query": "How does ICWA apply to emergency guardianship when the child is a tribal member?",
                    "expected_content": ["ICWA", "emergency", "tribal", "notification"],
                    "relevance_threshold": 0.8
                },
                {
                    "query": "What are the placement preferences under ICWA and how do they affect guardian selection?",
                    "expected_content": ["ICWA", "placement", "preferences", "Indian"],
                    "relevance_threshold": 0.7
                },
                {
                    "query": "If parents disagree about guardianship and one is tribal member, what special rules apply?",
                    "expected_content": ["tribal", "ICWA", "parent", "consent"],
                    "relevance_threshold": 0.6
                },
                {
                    "query": "Can I get emergency guardianship if CPS is involved and ICWA applies?",
                    "expected_content": ["emergency", "ICWA", "tribal"],
                    "relevance_threshold": 0.7
                }
            ],
            "edge_cases": {
                "out_of_scope": [
                    {
                        "query": "How do I adopt a child?",
                        "should_identify_oos": True,
                        "expected_response": "adoption"
                    },
                    {
                        "query": "I need adult guardianship for my mother",
                        "should_identify_oos": True,
                        "expected_response": "adult guardianship"
                    },
                    {
                        "query": "Information for Wayne County courts",
                        "should_identify_oos": True,
                        "expected_response": "Genesee County"
                    }
                ],
                "ambiguous": [
                    {
                        "query": "guardian",
                        "expected_behavior": "retrieve general guardianship info"
                    },
                    {
                        "query": "help",
                        "expected_behavior": "retrieve overview information"
                    },
                    {
                        "query": "court",
                        "expected_behavior": "retrieve court-related information"
                    }
                ],
                "very_long": [
                    {
                        "query": "I am a grandmother who has been caring for my grandchildren for the past three years after their parents struggled with substance abuse issues and now I need to formalize the arrangement through the court system to enroll them in school and make medical decisions but I'm not sure what forms I need or how much it will cost or whether I need a lawyer and I'm worried about whether the parents will contest it and what happens if they do",
                        "expected_content": ["forms", "PC 651", "fee", "parents"],
                        "relevance_threshold": 0.5
                    }
                ],
                "typos": [
                    {
                        "query": "guardinship filing fe",
                        "expected_content": ["guardianship", "filing", "fee"],
                        "relevance_threshold": 0.4
                    },
                    {
                        "query": "probat court lokation",
                        "expected_content": ["probate", "court", "location"],
                        "relevance_threshold": 0.4
                    }
                ]
            }
        }
        
    def setup_retriever(self, use_adaptive=False):
        """Initialize retriever for testing"""
        if use_adaptive:
            return AdaptiveHybridRetriever()
        else:
            return HybridRetriever()
    
    def test_retrieval_accuracy(self):
        """Test 1: Retrieval Accuracy with Golden Q&A Set"""
        print("\n=== Test 1: Retrieval Accuracy ===")
        
        retriever = self.setup_retriever(use_adaptive=True)
        accuracy_results = []
        
        for category, queries in self.test_queries.items():
            if category == "edge_cases":
                continue  # Handle edge cases separately
                
            print(f"\nTesting {category} queries...")
            category_results = []
            
            for test_case in queries:
                try:
                    start_time = time.time()
                    results, metadata = retriever.retrieve(test_case["query"])
                    retrieval_time = time.time() - start_time
                    
                    # Check if expected content is in top results
                    top_3_texts = " ".join([r['document'].lower() for r in results[:3]])
                    
                    found_content = []
                    missing_content = []
                    for expected in test_case["expected_content"]:
                        if expected.lower() in top_3_texts:
                            found_content.append(expected)
                        else:
                            missing_content.append(expected)
                    
                    relevance_score = len(found_content) / len(test_case["expected_content"])
                    
                    # Get similarity scores
                    scores = [r.get('rerank_score', r.get('score', 0)) for r in results[:3]]
                    avg_score = np.mean(scores) if scores else 0
                    
                    result = {
                        "query": test_case["query"],
                        "chunks_retrieved": len(results),
                        "relevance_score": relevance_score,
                        "avg_similarity_score": avg_score,
                        "found_content": found_content,
                        "missing_content": missing_content,
                        "retrieval_time": retrieval_time,
                        "complexity": metadata.get('complexity', 'unknown'),
                        "passed": relevance_score >= test_case["relevance_threshold"]
                    }
                    
                    category_results.append(result)
                    
                    status = "✓" if result["passed"] else "✗"
                    print(f"  {status} {test_case['query'][:50]}... (relevance: {relevance_score:.2f})")
                    
                except Exception as e:
                    print(f"  ✗ Error testing query: {e}")
                    category_results.append({
                        "query": test_case["query"],
                        "error": str(e),
                        "passed": False
                    })
            
            # Calculate category statistics
            passed = sum(1 for r in category_results if r.get("passed", False))
            total = len(category_results)
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            accuracy_results.append({
                "category": category,
                "total_queries": total,
                "passed": passed,
                "pass_rate": pass_rate,
                "avg_relevance": np.mean([r.get("relevance_score", 0) for r in category_results]),
                "avg_retrieval_time": np.mean([r.get("retrieval_time", 0) for r in category_results if "retrieval_time" in r]),
                "details": category_results
            })
        
        self.test_results["retrieval_accuracy"] = accuracy_results
        
        # Print summary
        print("\nAccuracy Summary:")
        for cat_result in accuracy_results:
            print(f"  {cat_result['category']}: {cat_result['pass_rate']:.1f}% pass rate, "
                  f"avg relevance: {cat_result['avg_relevance']:.2f}")
    
    def test_query_types(self):
        """Test 2: Different Query Types and Formats"""
        print("\n\n=== Test 2: Query Type Testing ===")
        
        retriever = self.setup_retriever(use_adaptive=True)
        
        # Test different query formats
        query_formats = {
            "question": "What is the filing fee for guardianship?",
            "keyword": "filing fee guardianship",
            "statement": "I need to know the filing fee",
            "command": "tell me the filing fee",
            "incomplete": "filing fee",
            "conversational": "hey can you help me figure out how much it costs to file"
        }
        
        format_results = []
        print("\nTesting query formats...")
        
        for format_type, query in query_formats.items():
            try:
                results, metadata = retriever.retrieve(query)
                
                # Check if filing fee info is retrieved
                found_fee = any("$175" in r['document'] or "filing fee" in r['document'].lower() 
                               for r in results[:3])
                
                format_results.append({
                    "format": format_type,
                    "query": query,
                    "chunks_retrieved": len(results),
                    "found_relevant": found_fee,
                    "complexity": metadata.get('complexity', 'unknown'),
                    "top_score": results[0].get('rerank_score', results[0].get('score', 0)) if results else 0
                })
                
                status = "✓" if found_fee else "✗"
                print(f"  {status} {format_type}: chunks={len(results)}, relevant={found_fee}")
                
            except Exception as e:
                print(f"  ✗ {format_type}: Error - {e}")
                format_results.append({
                    "format": format_type,
                    "query": query,
                    "error": str(e)
                })
        
        self.test_results["query_types"]["formats"] = format_results
        
        # Test query lengths
        print("\nTesting query lengths...")
        length_tests = [
            ("single_word", "guardianship"),
            ("short", "filing fee amount"),
            ("medium", "how do I file for guardianship of my grandchild"),
            ("long", "I am a grandmother who has been caring for my grandchildren and need to file for legal guardianship to make medical and educational decisions"),
            ("very_long", "I am a grandmother who has been caring for my grandchildren for the past three years after their parents struggled with substance abuse issues and now I need to formalize the arrangement through the court system to enroll them in school and make medical decisions but I'm not sure what forms I need or how much it will cost")
        ]
        
        length_results = []
        for length_type, query in length_tests:
            try:
                start_time = time.time()
                results, metadata = retriever.retrieve(query)
                retrieval_time = time.time() - start_time
                
                length_results.append({
                    "length_type": length_type,
                    "query_length": len(query),
                    "chunks_retrieved": len(results),
                    "retrieval_time": retrieval_time,
                    "complexity": metadata.get('complexity', 'unknown')
                })
                
                print(f"  {length_type} ({len(query)} chars): {len(results)} chunks in {retrieval_time:.2f}s")
                
            except Exception as e:
                print(f"  ✗ {length_type}: Error - {e}")
                length_results.append({
                    "length_type": length_type,
                    "error": str(e)
                })
        
        self.test_results["query_types"]["lengths"] = length_results
    
    def test_retrieval_parameters(self):
        """Test 3: Retrieval Parameters (top_k, hybrid search, etc.)"""
        print("\n\n=== Test 3: Retrieval Parameter Testing ===")
        
        retriever = self.setup_retriever(use_adaptive=False)
        test_query = "What forms do I need for guardianship?"
        
        # Test different top_k values
        print("\nTesting top_k values...")
        top_k_results = []
        
        for top_k in [1, 3, 5, 10, 15, 20]:
            try:
                # Modify retriever's top_k
                original_top_k = retriever.top_k
                retriever.top_k = top_k
                
                start_time = time.time()
                results, metadata = retriever.retrieve(test_query)
                retrieval_time = time.time() - start_time
                
                # Check result quality
                forms_found = any("PC 651" in r['document'] or "PC 652" in r['document'] 
                                 for r in results)
                
                top_k_results.append({
                    "top_k": top_k,
                    "chunks_retrieved": len(results),
                    "retrieval_time": retrieval_time,
                    "forms_found": forms_found,
                    "avg_score": np.mean([r.get('score', 0) for r in results]) if results else 0
                })
                
                print(f"  top_k={top_k}: {len(results)} chunks, {retrieval_time:.2f}s, forms_found={forms_found}")
                
                retriever.top_k = original_top_k
                
            except Exception as e:
                print(f"  ✗ top_k={top_k}: Error - {e}")
                top_k_results.append({
                    "top_k": top_k,
                    "error": str(e)
                })
        
        self.test_results["parameter_testing"]["top_k"] = top_k_results
        
        # Test vector vs hybrid search
        print("\nTesting search modes...")
        search_mode_results = []
        
        # Test with vector-only search
        try:
            # Temporarily disable BM25
            original_bm25 = retriever.bm25
            retriever.bm25 = None
            
            start_time = time.time()
            results_vector, _ = retriever.retrieve(test_query)
            vector_time = time.time() - start_time
            
            retriever.bm25 = original_bm25
            
            # Test with hybrid search
            start_time = time.time()
            results_hybrid, _ = retriever.retrieve(test_query)
            hybrid_time = time.time() - start_time
            
            # Compare results
            vector_forms = any("PC 651" in r['document'] or "PC 652" in r['document'] 
                              for r in results_vector[:5])
            hybrid_forms = any("PC 651" in r['document'] or "PC 652" in r['document'] 
                              for r in results_hybrid[:5])
            
            search_mode_results = {
                "vector_only": {
                    "chunks": len(results_vector),
                    "time": vector_time,
                    "found_forms": vector_forms,
                    "top_scores": [r.get('score', 0) for r in results_vector[:3]]
                },
                "hybrid": {
                    "chunks": len(results_hybrid),
                    "time": hybrid_time,
                    "found_forms": hybrid_forms,
                    "top_scores": [r.get('score', 0) for r in results_hybrid[:3]]
                }
            }
            
            print(f"  Vector-only: {vector_forms}, time={vector_time:.2f}s")
            print(f"  Hybrid: {hybrid_forms}, time={hybrid_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Search mode comparison failed: {e}")
            search_mode_results = {"error": str(e)}
        
        self.test_results["parameter_testing"]["search_modes"] = search_mode_results
        
        # Test query complexity impact
        print("\nTesting complexity classification impact...")
        complexity_results = []
        
        complexity_queries = [
            ("simple", "filing fee?"),
            ("standard", "how to file for guardianship"),
            ("complex", "ICWA requirements for emergency guardianship when parents disagree")
        ]
        
        adaptive_retriever = self.setup_retriever(use_adaptive=True)
        
        for expected_complexity, query in complexity_queries:
            try:
                results, metadata = adaptive_retriever.retrieve(query)
                
                complexity_results.append({
                    "query": query,
                    "expected_complexity": expected_complexity,
                    "detected_complexity": metadata.get('complexity', 'unknown'),
                    "chunks_retrieved": len(results),
                    "params": metadata.get('params', {})
                })
                
                print(f"  '{query[:30]}...': detected={metadata.get('complexity')}, "
                      f"top_k={metadata.get('params', {}).get('top_k')}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                complexity_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        self.test_results["parameter_testing"]["complexity"] = complexity_results
    
    def test_performance(self):
        """Test 4: Performance and Latency"""
        print("\n\n=== Test 4: Performance Testing ===")
        
        retriever = self.setup_retriever(use_adaptive=True)
        
        # Latency budgets by complexity
        latency_budgets = {
            "simple": 300,    # 300ms
            "standard": 500,  # 500ms
            "complex": 800    # 800ms
        }
        
        performance_results = defaultdict(list)
        
        # Run multiple iterations for each query type
        iterations = 5
        
        for category, queries in self.test_queries.items():
            if category == "edge_cases":
                continue
                
            print(f"\nTesting {category} performance ({iterations} iterations)...")
            
            for test_case in queries[:2]:  # Test first 2 queries from each category
                latencies = []
                
                for i in range(iterations):
                    try:
                        start_time = time.time()
                        results, metadata = retriever.retrieve_with_latency(test_case["query"])
                        latency = (time.time() - start_time) * 1000  # Convert to ms
                        
                        latencies.append(latency)
                        
                    except Exception as e:
                        print(f"  ✗ Error in iteration {i+1}: {e}")
                
                if latencies:
                    complexity = metadata.get('complexity', 'standard')
                    budget = latency_budgets.get(complexity, 500)
                    
                    perf_result = {
                        "query": test_case["query"][:50] + "...",
                        "complexity": complexity,
                        "latencies": latencies,
                        "avg_latency": np.mean(latencies),
                        "min_latency": np.min(latencies),
                        "max_latency": np.max(latencies),
                        "std_latency": np.std(latencies),
                        "budget_ms": budget,
                        "met_budget": np.mean(latencies) <= budget,
                        "budget_compliance": sum(1 for l in latencies if l <= budget) / len(latencies) * 100
                    }
                    
                    performance_results[category].append(perf_result)
                    
                    status = "✓" if perf_result["met_budget"] else "✗"
                    print(f"  {status} {perf_result['query']}: "
                          f"avg={perf_result['avg_latency']:.0f}ms "
                          f"(budget={budget}ms, compliance={perf_result['budget_compliance']:.0f}%)")
        
        self.test_results["performance_metrics"] = dict(performance_results)
        
        # Overall performance summary
        all_results = []
        for category_results in performance_results.values():
            all_results.extend(category_results)
        
        if all_results:
            overall_summary = {
                "total_queries_tested": len(all_results),
                "avg_latency_ms": np.mean([r["avg_latency"] for r in all_results]),
                "budget_compliance_rate": np.mean([r["budget_compliance"] for r in all_results]),
                "queries_meeting_budget": sum(1 for r in all_results if r["met_budget"]),
                "by_complexity": {}
            }
            
            # Group by complexity
            for complexity in ["simple", "standard", "complex"]:
                complexity_results = [r for r in all_results if r["complexity"] == complexity]
                if complexity_results:
                    overall_summary["by_complexity"][complexity] = {
                        "count": len(complexity_results),
                        "avg_latency": np.mean([r["avg_latency"] for r in complexity_results]),
                        "compliance_rate": np.mean([r["budget_compliance"] for r in complexity_results])
                    }
            
            self.test_results["performance_metrics"]["summary"] = overall_summary
            
            print("\nPerformance Summary:")
            print(f"  Overall avg latency: {overall_summary['avg_latency_ms']:.0f}ms")
            print(f"  Budget compliance: {overall_summary['budget_compliance_rate']:.1f}%")
            for complexity, stats in overall_summary["by_complexity"].items():
                print(f"  {complexity}: {stats['avg_latency']:.0f}ms avg, "
                      f"{stats['compliance_rate']:.0f}% compliance")
    
    def test_edge_cases(self):
        """Test 5: Edge Cases"""
        print("\n\n=== Test 5: Edge Case Testing ===")
        
        retriever = self.setup_retriever(use_adaptive=True)
        edge_results = {}
        
        # Test out-of-scope queries
        print("\nTesting out-of-scope detection...")
        oos_results = []
        
        for test_case in self.test_queries["edge_cases"]["out_of_scope"]:
            try:
                results, metadata = retriever.retrieve(test_case["query"])
                
                # Check if any results mention the out-of-scope topic
                top_text = " ".join([r['document'].lower() for r in results[:3]])
                contains_oos_topic = test_case["expected_response"].lower() in top_text
                
                oos_results.append({
                    "query": test_case["query"],
                    "should_identify_oos": test_case["should_identify_oos"],
                    "contains_oos_topic": contains_oos_topic,
                    "chunks_retrieved": len(results),
                    "correct": contains_oos_topic == test_case["should_identify_oos"]
                })
                
                status = "✓" if oos_results[-1]["correct"] else "✗"
                print(f"  {status} {test_case['query']}: "
                      f"found_oos_topic={contains_oos_topic}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                oos_results.append({
                    "query": test_case["query"],
                    "error": str(e)
                })
        
        edge_results["out_of_scope"] = oos_results
        
        # Test ambiguous queries
        print("\nTesting ambiguous queries...")
        ambiguous_results = []
        
        for test_case in self.test_queries["edge_cases"]["ambiguous"]:
            try:
                results, metadata = retriever.retrieve(test_case["query"])
                
                ambiguous_results.append({
                    "query": test_case["query"],
                    "expected_behavior": test_case["expected_behavior"],
                    "chunks_retrieved": len(results),
                    "complexity": metadata.get('complexity', 'unknown'),
                    "top_sources": [r['metadata'].get('source', 'unknown') for r in results[:3]]
                })
                
                print(f"  {test_case['query']}: {len(results)} chunks, "
                      f"complexity={metadata.get('complexity')}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                ambiguous_results.append({
                    "query": test_case["query"],
                    "error": str(e)
                })
        
        edge_results["ambiguous"] = ambiguous_results
        
        # Test very long queries
        print("\nTesting very long queries...")
        long_results = []
        
        for test_case in self.test_queries["edge_cases"]["very_long"]:
            try:
                start_time = time.time()
                results, metadata = retriever.retrieve(test_case["query"])
                retrieval_time = time.time() - start_time
                
                # Check if relevant content found despite length
                top_text = " ".join([r['document'].lower() for r in results[:5]])
                found_content = [exp for exp in test_case["expected_content"] 
                               if exp.lower() in top_text]
                
                long_results.append({
                    "query_preview": test_case["query"][:100] + "...",
                    "query_length": len(test_case["query"]),
                    "chunks_retrieved": len(results),
                    "retrieval_time": retrieval_time,
                    "found_content": found_content,
                    "relevance_score": len(found_content) / len(test_case["expected_content"])
                })
                
                print(f"  Long query ({len(test_case['query'])} chars): "
                      f"{len(results)} chunks in {retrieval_time:.2f}s, "
                      f"relevance={long_results[-1]['relevance_score']:.2f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                long_results.append({
                    "query_length": len(test_case["query"]),
                    "error": str(e)
                })
        
        edge_results["very_long"] = long_results
        
        # Test queries with typos
        print("\nTesting queries with typos...")
        typo_results = []
        
        for test_case in self.test_queries["edge_cases"]["typos"]:
            try:
                results, metadata = retriever.retrieve(test_case["query"])
                
                # Check if correct content found despite typos
                top_text = " ".join([r['document'].lower() for r in results[:3]])
                found_content = [exp for exp in test_case["expected_content"] 
                               if exp.lower() in top_text]
                
                typo_results.append({
                    "query": test_case["query"],
                    "chunks_retrieved": len(results),
                    "found_content": found_content,
                    "relevance_score": len(found_content) / len(test_case["expected_content"]),
                    "passed": len(found_content) / len(test_case["expected_content"]) >= test_case["relevance_threshold"]
                })
                
                status = "✓" if typo_results[-1]["passed"] else "✗"
                print(f"  {status} '{test_case['query']}': "
                      f"relevance={typo_results[-1]['relevance_score']:.2f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                typo_results.append({
                    "query": test_case["query"],
                    "error": str(e)
                })
        
        edge_results["typos"] = typo_results
        
        self.test_results["edge_cases"] = edge_results
    
    def generate_comprehensive_report(self):
        """Generate detailed test report"""
        print("\n\n=== Generating Comprehensive Test Report ===")
        
        # Calculate overall statistics
        summary = {
            "total_tests_run": 0,
            "total_passed": 0,
            "overall_pass_rate": 0,
            "category_summaries": {}
        }
        
        # Retrieval accuracy summary
        if "retrieval_accuracy" in self.test_results:
            total_accuracy_tests = 0
            total_accuracy_passed = 0
            
            for category in self.test_results["retrieval_accuracy"]:
                total_accuracy_tests += category["total_queries"]
                total_accuracy_passed += category["passed"]
            
            summary["category_summaries"]["retrieval_accuracy"] = {
                "total_tests": total_accuracy_tests,
                "passed": total_accuracy_passed,
                "pass_rate": (total_accuracy_passed / total_accuracy_tests * 100) if total_accuracy_tests > 0 else 0
            }
            
            summary["total_tests_run"] += total_accuracy_tests
            summary["total_passed"] += total_accuracy_passed
        
        # Performance summary
        if "performance_metrics" in self.test_results and "summary" in self.test_results["performance_metrics"]:
            perf_summary = self.test_results["performance_metrics"]["summary"]
            
            summary["category_summaries"]["performance"] = {
                "avg_latency_ms": perf_summary.get("avg_latency_ms", 0),
                "budget_compliance_rate": perf_summary.get("budget_compliance_rate", 0),
                "by_complexity": perf_summary.get("by_complexity", {})
            }
        
        # Edge cases summary
        if "edge_cases" in self.test_results:
            edge_summary = {
                "out_of_scope_correct": 0,
                "typos_handled": 0,
                "long_queries_handled": 0
            }
            
            if "out_of_scope" in self.test_results["edge_cases"]:
                oos_correct = sum(1 for r in self.test_results["edge_cases"]["out_of_scope"]
                                 if r.get("correct", False))
                edge_summary["out_of_scope_correct"] = oos_correct
            
            if "typos" in self.test_results["edge_cases"]:
                typos_passed = sum(1 for r in self.test_results["edge_cases"]["typos"]
                                  if r.get("passed", False))
                edge_summary["typos_handled"] = typos_passed
            
            summary["category_summaries"]["edge_cases"] = edge_summary
        
        # Calculate overall pass rate
        if summary["total_tests_run"] > 0:
            summary["overall_pass_rate"] = (summary["total_passed"] / summary["total_tests_run"]) * 100
        
        self.test_results["summary"] = summary
        
        # Save detailed report
        report_path = RESULTS_DIR / f"comprehensive_retrieval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("COMPREHENSIVE RETRIEVAL TEST SUMMARY")
        print("="*70)
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {summary['total_tests_run']}")
        print(f"  Tests Passed: {summary['total_passed']}")
        print(f"  Overall Pass Rate: {summary['overall_pass_rate']:.1f}%")
        
        if "retrieval_accuracy" in summary["category_summaries"]:
            acc_summary = summary["category_summaries"]["retrieval_accuracy"]
            print(f"\nRetrieval Accuracy:")
            print(f"  Pass Rate: {acc_summary['pass_rate']:.1f}%")
            
            # Show category breakdown
            for category in self.test_results["retrieval_accuracy"]:
                print(f"  - {category['category']}: {category['pass_rate']:.1f}% "
                      f"(avg relevance: {category['avg_relevance']:.2f})")
        
        if "performance" in summary["category_summaries"]:
            perf = summary["category_summaries"]["performance"]
            print(f"\nPerformance:")
            print(f"  Average Latency: {perf['avg_latency_ms']:.0f}ms")
            print(f"  Budget Compliance: {perf['budget_compliance_rate']:.1f}%")
            
            for complexity, stats in perf.get("by_complexity", {}).items():
                print(f"  - {complexity}: {stats['avg_latency']:.0f}ms avg, "
                      f"{stats['compliance_rate']:.0f}% compliance")
        
        if "edge_cases" in summary["category_summaries"]:
            edge = summary["category_summaries"]["edge_cases"]
            print(f"\nEdge Cases:")
            print(f"  Out-of-scope detection: {edge.get('out_of_scope_correct', 0)} correct")
            print(f"  Typo handling: {edge.get('typos_handled', 0)} handled")
        
        print("\n" + "="*70)
        
        return report_path

def main():
    """Run comprehensive retrieval tests"""
    print("Starting Comprehensive Retrieval System Tests")
    print("="*70)
    
    tester = ComprehensiveRetrievalTester()
    
    # Run all test suites
    tester.test_retrieval_accuracy()
    tester.test_query_types()
    tester.test_retrieval_parameters()
    tester.test_performance()
    tester.test_edge_cases()
    
    # Generate report
    report_path = tester.generate_comprehensive_report()
    
    print(f"\n✓ All tests completed!")
    print(f"Full results saved to: {report_path}")

if __name__ == "__main__":
    main()