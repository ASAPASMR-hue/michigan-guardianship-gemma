#!/usr/bin/env python3
"""
full_pipeline_test.py - Comprehensive Integration Tests for Michigan Guardianship AI
Tests the complete pipeline from document processing to response generation
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import yaml
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.embed_kb import DocumentProcessor, chunk_text, extract_pdf_text
from scripts.retrieval_setup import HybridRetriever
from scripts.validator_setup import ResponseValidator
from scripts.adaptive_retrieval import AdaptiveHybridRetriever
from scripts.log_step import log_step

# Test configuration
TEST_DIR = Path(__file__).parent
TEST_DOCS_DIR = TEST_DIR / "test_documents"
TEST_CHROMA_DIR = TEST_DIR / "test_chroma_db"
RESULTS_DIR = TEST_DIR / "test_results"

# Create results directory
RESULTS_DIR.mkdir(exist_ok=True)

class IntegrationTester:
    """Runs comprehensive integration tests on the guardianship AI system"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_tests": {},
            "golden_qa_tests": [],
            "summary": {}
        }
        
        # Golden Q&A pairs for testing
        self.golden_qa_pairs = [
            # Simple factual queries
            {
                "question": "What is the filing fee for guardianship in Genesee County?",
                "expected_facts": ["$175", "fee waiver", "MC 20"],
                "expected_type": "simple",
                "must_not_contain": ["$150", "$200", "Monday", "Tuesday"]
            },
            {
                "question": "Where is the Genesee County Probate Court located?",
                "expected_facts": ["900 S. Saginaw", "Flint", "48502"],
                "expected_type": "simple",
                "must_not_contain": ["Detroit", "Lansing", "wrong address"]
            },
            {
                "question": "What day are guardianship hearings held?",
                "expected_facts": ["Thursday"],
                "expected_type": "simple",
                "must_not_contain": ["Monday", "Tuesday", "Wednesday", "Friday"]
            },
            
            # Procedural questions
            {
                "question": "What forms do I need to file for minor guardianship?",
                "expected_facts": ["PC 651", "PC 652"],
                "expected_type": "standard",
                "must_not_contain": ["PC 600", "wrong form"]
            },
            {
                "question": "How do I request a fee waiver?",
                "expected_facts": ["MC 20", "financial", "cannot afford"],
                "expected_type": "standard",
                "must_not_contain": ["automatic", "no form needed"]
            },
            {
                "question": "Who needs to be notified about the guardianship hearing?",
                "expected_facts": ["parents", "14 days", "interested parties"],
                "expected_type": "standard",
                "must_not_contain": ["no notice", "same day"]
            },
            
            # Complex scenarios
            {
                "question": "How does ICWA apply to emergency guardianship proceedings?",
                "expected_facts": ["tribal notification", "ICWA", "emergency", "MCL 712B"],
                "expected_type": "complex",
                "must_not_contain": ["no notification", "skip ICWA"]
            },
            {
                "question": "My grandson is a tribal member and I need emergency guardianship. What special requirements apply?",
                "expected_facts": ["ICWA", "notify tribe", "active efforts", "placement preferences"],
                "expected_type": "complex",
                "must_not_contain": ["no special requirements", "same as regular"]
            },
            
            # Out-of-scope queries
            {
                "question": "How do I get guardianship of my elderly mother with dementia?",
                "expected_facts": ["adult guardianship", "elder law attorney", "minor guardianship only"],
                "expected_type": "out_of_scope",
                "must_not_contain": ["PC 651", "minor forms"]
            },
            {
                "question": "I need guardianship information for Oakland County",
                "expected_facts": ["Genesee County", "contact.*probate court"],
                "expected_type": "out_of_scope",
                "must_not_contain": ["same process", "PC 651"]
            }
        ]
        
        # Initialize semantic similarity model for validation
        self.similarity_model = None
    
    def check_semantic_similarity(self, expected_text: str, actual_text: str, threshold: float = 0.85) -> bool:
        """Check if two texts are semantically similar using embeddings"""
        if self.similarity_model is None:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Handle None or empty inputs
        if not expected_text or not actual_text:
            return False
        
        # Generate embeddings
        embeddings = self.similarity_model.encode([expected_text, actual_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
        
        return similarity >= threshold
    
    def check_content_match(self, expected_facts: List[str], response_text: str, use_semantic: bool = True) -> Tuple[List[str], bool]:
        """Check if expected facts are present in response, with semantic similarity fallback"""
        missing_facts = []
        semantic_matches = []
        
        for fact in expected_facts:
            fact_lower = fact.lower()
            response_lower = response_text.lower()
            
            # First try exact match
            if fact_lower in response_lower:
                continue
            
            # For critical values, require exact match
            if fact in ["$175", "Thursday", "MCL 712B", "PC 651", "PC 652", "MC 20"]:
                missing_facts.append(fact)
                continue
            
            # Try semantic similarity for other facts
            if use_semantic:
                # Extract relevant sentence/chunk from response
                sentences = response_text.split('. ')
                found_similar = False
                best_similarity = 0
                best_sentence = ""
                
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Skip very short sentences
                        # Initialize model if needed
                        if self.similarity_model is None:
                            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                        similarity = cosine_similarity(
                            self.similarity_model.encode([fact]).reshape(1, -1),
                            self.similarity_model.encode([sentence]).reshape(1, -1)
                        )[0][0]
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_sentence = sentence
                        
                        if similarity >= 0.8:
                            found_similar = True
                            semantic_matches.append(f"{fact} → {sentence[:50]}... (similarity: {similarity:.2f})")
                            break
                
                if not found_similar:
                    missing_facts.append(f"{fact} (best match: {best_similarity:.2f})")
            else:
                missing_facts.append(fact)
        
        if semantic_matches:
            print(f"    Semantic matches found: {len(semantic_matches)}")
            for match in semantic_matches[:3]:  # Show first 3 matches
                print(f"      - {match}")
        
        return missing_facts, len(missing_facts) == 0
    
    def setup_test_environment(self):
        """Set up test ChromaDB and configurations"""
        print("\n=== Setting Up Test Environment ===")
        
        # Clean up any existing test database
        if TEST_CHROMA_DIR.exists():
            shutil.rmtree(TEST_CHROMA_DIR)
        TEST_CHROMA_DIR.mkdir(exist_ok=True)
        
        # Use small model for faster testing
        os.environ['USE_SMALL_MODEL'] = 'true'
        
        log_step("Test environment setup", "Initialized test directories and configurations", "Integration testing")
    
    def test_document_pipeline(self):
        """Test 1: Document processing and embedding pipeline"""
        print("\n=== Test 1: Document-to-Database Pipeline ===")
        start_time = time.time()
        
        try:
            # Load test configurations
            with open(Path(__file__).parent.parent / "config" / "chunking.yaml", "r") as f:
                chunking_config = yaml.safe_load(f)
            
            # Initialize embedding model
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize test ChromaDB
            client = chromadb.PersistentClient(
                path=str(TEST_CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collection
            collection = client.create_collection(
                name="test_guardianship",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Process test documents
            doc_processor = DocumentProcessor()
            all_chunks = []
            
            for doc_path in TEST_DOCS_DIR.glob("*.txt"):
                print(f"\nProcessing: {doc_path.name}")
                
                # Read document
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Chunk document with better strategy for test
                # Split by sections for better retrieval
                chunks = []
                sections = content.split('\n\n')
                chunk_id = 0
                
                for section in sections:
                    if section.strip():
                        # Keep sections under 500 chars, split if needed
                        if len(section) <= 500:
                            chunks.append({
                                'text': section.strip(),
                                'chunk_id': f"{chunk_id:04d}"
                            })
                            chunk_id += 1
                        else:
                            # Split long sections by sentences
                            sentences = section.split('. ')
                            current_chunk = ""
                            for sentence in sentences:
                                if sentence.strip():
                                    sentence = sentence.strip() + '.'
                                    if len(current_chunk) + len(sentence) < 500:
                                        current_chunk += " " + sentence if current_chunk else sentence
                                    else:
                                        if current_chunk:
                                            chunks.append({
                                                'text': current_chunk,
                                                'chunk_id': f"{chunk_id:04d}"
                                            })
                                            chunk_id += 1
                                        current_chunk = sentence
                            if current_chunk:
                                chunks.append({
                                    'text': current_chunk,
                                    'chunk_id': f"{chunk_id:04d}"
                                })
                                chunk_id += 1
                
                print(f"  Created {len(chunks)} chunks")
                
                # Add metadata
                for chunk in chunks:
                    chunk_with_metadata = {
                        'id': f"{doc_path.stem}_{chunk['chunk_id']}",
                        'text': chunk['text'],
                        'metadata': {
                            'source': doc_path.name,
                            'doc_type': 'test',
                            'jurisdiction': 'Genesee County',
                            'chunk_id': chunk['chunk_id']
                        }
                    }
                    all_chunks.append(chunk_with_metadata)
            
            print(f"\nTotal chunks created: {len(all_chunks)}")
            
            # Embed chunks
            print("Embedding chunks...")
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = embed_model.encode(texts, normalize_embeddings=True)
            
            # Store in ChromaDB
            print("Storing in ChromaDB...")
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[chunk['metadata'] for chunk in all_chunks],
                ids=[chunk['id'] for chunk in all_chunks]
            )
            
            # Verify storage
            count = collection.count()
            print(f"Chunks stored in database: {count}")
            
            # Test retrieval
            test_query = "filing fee"
            query_embedding = embed_model.encode(test_query, normalize_embeddings=True)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3
            )
            
            print(f"\nTest query '{test_query}' retrieved {len(results['documents'][0])} results")
            
            pipeline_time = time.time() - start_time
            
            self.test_results["pipeline_tests"]["document_embedding"] = {
                "status": "PASSED",
                "chunks_created": len(all_chunks),
                "chunks_stored": count,
                "test_retrieval_count": len(results['documents'][0]),
                "time_seconds": pipeline_time
            }
            
            print(f"\n✓ Document pipeline test PASSED in {pipeline_time:.2f}s")
            
        except Exception as e:
            self.test_results["pipeline_tests"]["document_embedding"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"\n✗ Document pipeline test FAILED: {e}")
    
    def test_query_pipeline(self):
        """Test 2: Query-to-Response Pipeline with Golden Q&A"""
        print("\n\n=== Test 2: Query-to-Response Pipeline ===")
        
        # Initialize components with test database
        os.environ['CHROMA_PATH'] = str(TEST_CHROMA_DIR)
        
        # Mock retriever that uses test database
        class TestRetriever(HybridRetriever):
            def init_chromadb(self):
                """Override to use test database"""
                self.chroma_client = chromadb.PersistentClient(
                    path=str(TEST_CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False)
                )
                try:
                    self.collection = self.chroma_client.get_collection("test_guardianship")
                    results = self.collection.get()
                    self.all_docs = results['documents']
                    self.all_ids = results['ids']
                    self.all_metadata = results['metadatas']
                    print(f"Connected to test database with {len(self.all_docs)} documents")
                except:
                    print("Warning: Test collection not found, creating empty one")
                    self.collection = self.chroma_client.create_collection("test_guardianship")
                    self.all_docs = []
                    self.all_ids = []
                    self.all_metadata = []
        
        try:
            retriever = TestRetriever()
            validator = ResponseValidator()
            
            for i, qa_pair in enumerate(self.golden_qa_pairs):
                print(f"\n--- Golden Q&A Test {i+1}/{len(self.golden_qa_pairs)} ---")
                print(f"Question: {qa_pair['question']}")
                
                test_result = {
                    "question": qa_pair["question"],
                    "expected_type": qa_pair["expected_type"],
                    "tests": {}
                }
                
                try:
                    # Step 1: Retrieve chunks
                    start_time = time.time()
                    results, metadata = retriever.retrieve(qa_pair["question"])
                    retrieval_time = time.time() - start_time
                    
                    print(f"Retrieved {len(results)} chunks in {retrieval_time:.2f}s")
                    print(f"Complexity: {metadata['complexity']}")
                    
                    # Debug: Show what was retrieved
                    if qa_pair["question"] in ["Where is the Genesee County Probate Court located?", 
                                              "How do I request a fee waiver?",
                                              "How does ICWA apply to emergency guardianship proceedings?"]:
                        print(f"\n  DEBUG: Retrieved chunks for query:")
                        for i, result in enumerate(results[:3]):
                            print(f"    {i+1}. Score: {result.get('rerank_score', result.get('score', 0)):.3f}")
                            print(f"       Source: {result['metadata'].get('source', 'Unknown')}")
                            print(f"       Text: {result['document'][:150]}...")
                        
                        # Check if court info is in any document
                        print("\n  DEBUG: Checking all documents for court info:")
                        for i, doc in enumerate(retriever.all_docs):
                            if "900 S. Saginaw" in doc or "court location" in doc.lower():
                                print(f"    Found in doc {i}: {retriever.all_metadata[i].get('source')}")
                                print(f"    Text snippet: {doc[:200]}...")
                    
                    test_result["tests"]["retrieval"] = {
                        "status": "PASSED",
                        "chunks_retrieved": len(results),
                        "complexity": metadata['complexity'],
                        "time_seconds": retrieval_time
                    }
                    
                    # Check if we got relevant chunks
                    chunk_texts = [r['document'] for r in results[:3]]
                    relevant_chunks = any(
                        any(fact.lower() in chunk.lower() for fact in qa_pair["expected_facts"])
                        for chunk in chunk_texts
                    )
                    
                    if not relevant_chunks and qa_pair["expected_type"] != "out_of_scope":
                        print("  ⚠️  Warning: No relevant chunks retrieved")
                    
                    # Step 2: Generate response (mock for now)
                    # In a real system, this would call the LLM
                    mock_response = self.generate_mock_response(qa_pair, chunk_texts)
                    
                    # Step 3: Validate response
                    validation_result = validator.validate(
                        mock_response,
                        chunk_texts,
                        metadata['complexity'],
                        qa_pair["question"]
                    )
                    
                    print(f"Validation: {'PASSED' if validation_result['pass'] else 'FAILED'}")
                    
                    test_result["tests"]["validation"] = {
                        "status": "PASSED" if validation_result['pass'] else "FAILED",
                        "out_of_scope": validation_result.get('out_of_scope', False)
                    }
                    
                    # Step 4: Verify expected content
                    if qa_pair["expected_type"] == "out_of_scope":
                        if validation_result.get('out_of_scope'):
                            test_result["tests"]["content_check"] = {"status": "PASSED"}
                            print("  ✓ Correctly identified as out-of-scope")
                        else:
                            test_result["tests"]["content_check"] = {
                                "status": "FAILED",
                                "reason": "Should have been identified as out-of-scope"
                            }
                    else:
                        # Check for expected facts using semantic similarity
                        response_to_check = validation_result.get('final_response', mock_response)
                        missing_facts, all_facts_found = self.check_content_match(
                            qa_pair["expected_facts"], 
                            response_to_check,
                            use_semantic=True
                        )
                        
                        # Check for forbidden content (still use exact match for these)
                        forbidden_found = []
                        for forbidden in qa_pair.get("must_not_contain", []):
                            if forbidden.lower() in response_to_check.lower():
                                forbidden_found.append(forbidden)
                        
                        if all_facts_found and not forbidden_found:
                            test_result["tests"]["content_check"] = {"status": "PASSED"}
                            print("  ✓ All expected facts found (including semantic matches), no forbidden content")
                        else:
                            test_result["tests"]["content_check"] = {
                                "status": "FAILED",
                                "missing_facts": missing_facts,
                                "forbidden_found": forbidden_found
                            }
                            if missing_facts:
                                print(f"  ✗ Missing facts: {missing_facts}")
                            if forbidden_found:
                                print(f"  ✗ Forbidden content found: {forbidden_found}")
                    
                    # Overall status
                    all_passed = all(
                        test.get("status") == "PASSED" 
                        for test in test_result["tests"].values()
                    )
                    test_result["overall_status"] = "PASSED" if all_passed else "FAILED"
                    
                except Exception as e:
                    print(f"  ✗ Test failed with error: {e}")
                    test_result["overall_status"] = "FAILED"
                    test_result["error"] = str(e)
                
                self.test_results["golden_qa_tests"].append(test_result)
                
        except Exception as e:
            print(f"\n✗ Query pipeline setup failed: {e}")
            self.test_results["pipeline_tests"]["query_response"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    def generate_mock_response(self, qa_pair: Dict, chunks: List[str]) -> str:
        """Generate a mock response based on retrieved chunks for testing"""
        # This simulates what an LLM would generate from the actual retrieved chunks
        if qa_pair["expected_type"] == "out_of_scope":
            return "I can help with adult guardianship matters."
        
        # Actually use the retrieved chunks to build response
        response_parts = []
        question_lower = qa_pair["question"].lower()
        
        # Extract relevant information from chunks
        for chunk in chunks:
            chunk_lower = chunk.lower()
            
            # Filing fee information
            if "filing fee" in question_lower and ("$175" in chunk or "filing fee" in chunk_lower):
                # Extract the filing fee info from chunk
                if "$175" in chunk:
                    response_parts.append("The filing fee for guardianship in Genesee County is $175.00.")
                if "mc 20" in chunk_lower:
                    response_parts.append("If you cannot afford this fee, you can request a fee waiver using Form MC 20.")
            
            # Court location
            if ("court" in question_lower and "located" in question_lower) or "address" in question_lower:
                if "900 s. saginaw" in chunk_lower or "court location" in chunk_lower:
                    # Extract the full address from the chunk
                    if "900 s. saginaw" in chunk_lower and "flint" in chunk_lower:
                        response_parts.append("The Genesee County Probate Court is located at 900 S. Saginaw Street, Flint, MI 48502.")
            
            # Hearing days
            if "hearing" in question_lower and ("day" in question_lower or "when" in question_lower):
                if "thursday" in chunk_lower and "hearing" in chunk_lower:
                    response_parts.append("Guardianship hearings in Genesee County are held on Thursdays.")
                    if "9:00 am" in chunk_lower:
                        response_parts.append("Hearings typically start at 9:00 AM.")
            
            # Forms needed
            if "forms" in question_lower and "need" in question_lower:
                if "pc 651" in chunk_lower or "pc 652" in chunk_lower:
                    forms = []
                    if "pc 651" in chunk_lower:
                        forms.append("Form PC 651 (Petition for Appointment of Guardian of a Minor)")
                    if "pc 652" in chunk_lower:
                        forms.append("Form PC 652 (Notice of Hearing)")
                    if forms:
                        response_parts.append(f"You will need: {', '.join(forms)}.")
            
            # Fee waiver
            if "fee waiver" in question_lower:
                if "mc 20" in chunk_lower:
                    response_parts.append("To request a fee waiver, you need to file Form MC 20.")
                    if "financial" in chunk_lower or "cannot afford" in chunk_lower:
                        response_parts.append("The court will review your financial situation to determine if you qualify for a waiver.")
            
            # Notification requirements
            if "notifi" in question_lower:
                if "parent" in chunk_lower and "14 day" in chunk_lower:
                    response_parts.append("Parents must be notified at least 14 days before the hearing.")
                if "interested part" in chunk_lower:
                    response_parts.append("All interested parties must be properly notified.")
            
            # ICWA requirements
            if "icwa" in question_lower or "tribal" in question_lower:
                # Look for ICWA mentions
                if "icwa" in chunk_lower:
                    if "emergency" in chunk_lower and "tribal notification" in chunk_lower:
                        response_parts.append("ICWA requires tribal notification even in emergency situations.")
                    if "mcl 712b" in chunk_lower or "712b.15" in chunk_lower:
                        response_parts.append("You must notify the child's tribe immediately as required by MCL 712B.15.")
                
                if "notify" in chunk_lower and ("tribe" in chunk_lower or "tribal" in chunk_lower):
                    response_parts.append("You must notify the child's tribe about the guardianship proceedings.")
                if "active efforts" in chunk_lower:
                    response_parts.append("Active efforts must be made to prevent the breakup of the Indian family.")
                if "placement preference" in chunk_lower:
                    response_parts.append("ICWA placement preferences must be followed.")
        
        # If no relevant info found, return a generic message
        if not response_parts:
            return "I found some information in the knowledge base but couldn't extract specific details for your question."
        
        return " ".join(response_parts)
    
    def test_latency_compliance(self):
        """Test 3: Latency compliance testing"""
        print("\n\n=== Test 3: Latency Compliance Testing ===")
        
        try:
            # Use adaptive retriever for latency testing
            os.environ['CHROMA_PATH'] = str(TEST_CHROMA_DIR)
            
            class TestAdaptiveRetriever(AdaptiveHybridRetriever):
                def init_chromadb(self):
                    """Override to use test database"""
                    self.chroma_client = chromadb.PersistentClient(
                        path=str(TEST_CHROMA_DIR),
                        settings=Settings(anonymized_telemetry=False)
                    )
                    try:
                        self.collection = self.chroma_client.get_collection("test_guardianship")
                        results = self.collection.get()
                        self.all_docs = results['documents']
                        self.all_ids = results['ids']
                        self.all_metadata = results['metadatas']
                    except:
                        self.collection = self.chroma_client.create_collection("test_guardianship")
                        self.all_docs = []
                        self.all_ids = []
                        self.all_metadata = []
            
            retriever = TestAdaptiveRetriever()
            
            latency_results = {
                "simple": [],
                "standard": [],
                "complex": []
            }
            
            # Test queries by complexity
            test_queries = {
                "simple": ["filing fee?", "court address?", "what form?"],
                "standard": ["how to file guardianship", "parent consent needed"],
                "complex": ["ICWA emergency placement", "multi-state guardianship"]
            }
            
            for complexity, queries in test_queries.items():
                print(f"\nTesting {complexity} queries...")
                for query in queries:
                    try:
                        results, metadata = retriever.retrieve_with_latency(query)
                        latency_ms = metadata['latency']['total_ms']
                        budget_ms = metadata['params']['latency_budget_ms']
                        met_budget = metadata['met_budget']
                        
                        latency_results[complexity].append({
                            "query": query,
                            "latency_ms": latency_ms,
                            "budget_ms": budget_ms,
                            "met_budget": met_budget
                        })
                        
                        status = "✓" if met_budget else "✗"
                        print(f"  {status} '{query}': {latency_ms:.0f}ms (budget: {budget_ms}ms)")
                        
                    except Exception as e:
                        print(f"  ✗ '{query}': Failed - {e}")
            
            # Calculate compliance rates
            compliance_summary = {}
            for complexity, results in latency_results.items():
                if results:
                    met_count = sum(1 for r in results if r['met_budget'])
                    compliance_rate = (met_count / len(results)) * 100
                    avg_latency = sum(r['latency_ms'] for r in results) / len(results)
                    
                    compliance_summary[complexity] = {
                        "compliance_rate": compliance_rate,
                        "average_latency_ms": avg_latency,
                        "queries_tested": len(results)
                    }
            
            self.test_results["pipeline_tests"]["latency_compliance"] = {
                "status": "PASSED" if all(s.get('compliance_rate', 0) >= 80 for s in compliance_summary.values()) else "PARTIAL",
                "summary": compliance_summary,
                "details": latency_results
            }
            
            print("\nLatency Compliance Summary:")
            for complexity, summary in compliance_summary.items():
                print(f"  {complexity}: {summary['compliance_rate']:.0f}% compliance, avg {summary['average_latency_ms']:.0f}ms")
            
        except Exception as e:
            print(f"\n✗ Latency testing failed: {e}")
            self.test_results["pipeline_tests"]["latency_compliance"] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n\n=== Generating Test Report ===")
        
        # Calculate summary statistics
        total_tests = len(self.test_results["golden_qa_tests"])
        passed_tests = sum(1 for test in self.test_results["golden_qa_tests"] 
                          if test.get("overall_status") == "PASSED")
        
        pipeline_tests_total = len(self.test_results["pipeline_tests"])
        pipeline_tests_passed = sum(1 for test in self.test_results["pipeline_tests"].values()
                                   if test.get("status") == "PASSED")
        
        self.test_results["summary"] = {
            "golden_qa_tests": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "pipeline_tests": {
                "total": pipeline_tests_total,
                "passed": pipeline_tests_passed,
                "failed": pipeline_tests_total - pipeline_tests_passed
            }
        }
        
        # Save detailed report
        report_path = RESULTS_DIR / f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        print("\nPipeline Tests:")
        for test_name, result in self.test_results["pipeline_tests"].items():
            status = result.get("status", "UNKNOWN")
            print(f"  {test_name}: {status}")
        
        print(f"\nGolden Q&A Tests:")
        print(f"  Total: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Pass Rate: {self.test_results['summary']['golden_qa_tests']['pass_rate']:.1f}%")
        
        # Show failed tests
        failed_tests = [test for test in self.test_results["golden_qa_tests"] 
                       if test.get("overall_status") != "PASSED"]
        if failed_tests:
            print("\nFailed Tests:")
            for test in failed_tests:
                print(f"  - {test['question']}")
                if 'error' in test:
                    print(f"    Error: {test['error']}")
                else:
                    for test_name, result in test.get('tests', {}).items():
                        if result.get('status') != 'PASSED':
                            print(f"    {test_name}: {result}")
        
        print("\n" + "="*60)
        
        return report_path
    
    def cleanup(self):
        """Clean up test environment"""
        print("\nCleaning up test environment...")
        if TEST_CHROMA_DIR.exists():
            shutil.rmtree(TEST_CHROMA_DIR)
        os.environ.pop('USE_SMALL_MODEL', None)
        os.environ.pop('CHROMA_PATH', None)
        print("✓ Cleanup complete")

def create_test_scripts():
    """Create automated test scripts"""
    
    # Create golden question runner
    golden_qa_script = '''#!/usr/bin/env python3
"""
run_golden_qa.py - Run golden Q&A tests against the system
"""

import json
from pathlib import Path
from datetime import datetime
from integration_tests.full_pipeline_test import IntegrationTester

def main():
    print("Running Golden Q&A Test Suite...")
    print("="*60)
    
    tester = IntegrationTester()
    
    # Only run golden Q&A tests
    tester.setup_test_environment()
    tester.test_document_pipeline()  # Need documents for retrieval
    tester.test_query_pipeline()
    
    # Generate report
    report_path = tester.generate_test_report()
    
    # Quick summary
    with open(report_path, 'r') as f:
        results = json.load(f)
    
    print(f"\\nQuick Results:")
    print(f"Pass Rate: {results['summary']['golden_qa_tests']['pass_rate']:.1f}%")
    
    tester.cleanup()

if __name__ == "__main__":
    main()
'''
    
    script_path = TEST_DIR / "run_golden_qa.py"
    with open(script_path, 'w') as f:
        f.write(golden_qa_script)
    os.chmod(script_path, 0o755)
    
    # Create quick test runner
    quick_test_script = '''#!/usr/bin/env python3
"""
quick_test.py - Quick integration test for development
"""

from integration_tests.full_pipeline_test import IntegrationTester

def main():
    print("Running Quick Integration Test...")
    
    tester = IntegrationTester()
    tester.setup_test_environment()
    
    # Just test basic pipeline
    tester.test_document_pipeline()
    
    # Test a few queries
    tester.golden_qa_pairs = tester.golden_qa_pairs[:3]  # Only first 3
    tester.test_query_pipeline()
    
    tester.generate_test_report()
    tester.cleanup()
    
    print("\\n✓ Quick test complete!")

if __name__ == "__main__":
    main()
'''
    
    script_path = TEST_DIR / "quick_test.py"
    with open(script_path, 'w') as f:
        f.write(quick_test_script)
    os.chmod(script_path, 0o755)
    
    print(f"Created test scripts in {TEST_DIR}")

def main():
    """Run full integration tests"""
    log_step("Starting Integration Tests", 
             "Running comprehensive system integration tests",
             "Quality Assurance")
    
    tester = IntegrationTester()
    
    try:
        # Run all tests
        tester.setup_test_environment()
        tester.test_document_pipeline()
        tester.test_query_pipeline()
        tester.test_latency_compliance()
        
        # Generate report
        report_path = tester.generate_test_report()
        
        log_step("Integration Tests Complete",
                f"All tests completed. Report saved to {report_path}",
                "Testing")
        
    finally:
        tester.cleanup()
    
    # Create helper scripts
    create_test_scripts()

if __name__ == "__main__":
    main()