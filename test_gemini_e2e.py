#!/usr/bin/env python3
"""
test_gemini_e2e.py - End-to-End Test with Google Gemini API
Tests the complete system with real LLM responses
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from scripts.retrieval_setup import HybridRetriever
from scripts.validator_setup import ResponseValidator
from scripts.log_step import log_step

# Google Gemini setup
import google.generativeai as genai

class GeminiEndToEndTester:
    """Test the complete pipeline with Google Gemini"""
    
    def __init__(self, api_key: str):
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Set up environment
        # Ensure HuggingFace token is set from environment
        if not os.getenv('HUGGING_FACE_HUB_TOKEN'):
            print("Warning: HUGGING_FACE_HUB_TOKEN not set. Some models may require authentication.")
        os.environ['USE_SMALL_MODEL'] = 'true'  # Use small models for faster testing
        
        # Initialize components
        print("Initializing retrieval system...")
        self.retriever = HybridRetriever()
        self.validator = ResponseValidator()
        
        # System prompt
        self.system_prompt = """You are a helpful assistant specializing in Michigan minor guardianship law, specifically for Genesee County. 

Your role is to provide accurate, accessible information about minor guardianship procedures, requirements, and forms.

Guidelines:
1. Only provide information about MINOR (under 18) guardianship in Genesee County, Michigan
2. Always cite specific forms (e.g., PC 651) and statutes (e.g., MCL 700.5204)
3. Include these Genesee County specifics when relevant:
   - Filing fee: $175 (fee waiver available via Form MC 20)
   - Court location: 900 S. Saginaw Street, Flint, MI 48502
   - Hearings: Thursdays at 9:00 AM
4. Base your answers ONLY on the provided context chunks
5. If information is not in the context, say so clearly"""
        
        # Test cases focusing on the problematic queries
        self.test_cases = [
            {
                "id": "fee_test",
                "question": "What is the filing fee for guardianship in Genesee County?",
                "expected_facts": ["$175", "fee waiver", "MC 20"],
                "critical_values": ["$175"]
            },
            {
                "id": "court_location_test", 
                "question": "Where is the Genesee County Probate Court located?",
                "expected_facts": ["900 S. Saginaw", "Flint", "48502"],
                "critical_values": ["900 S. Saginaw"]
            },
            {
                "id": "hearing_day_test",
                "question": "What day are guardianship hearings held?",
                "expected_facts": ["Thursday"],
                "critical_values": ["Thursday"]
            },
            {
                "id": "icwa_test",
                "question": "How does ICWA apply to emergency guardianship proceedings?",
                "expected_facts": ["tribal notification", "ICWA", "emergency", "MCL 712B"],
                "critical_values": ["ICWA", "tribal notification"]
            },
            {
                "id": "fee_waiver_test",
                "question": "How do I request a fee waiver?",
                "expected_facts": ["MC 20", "cannot afford", "financial"],
                "critical_values": ["MC 20"]
            }
        ]
    
    def generate_llm_response(self, query: str, context_chunks: List[str]) -> str:
        """Generate response using Google Gemini"""
        # Format context
        context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Build prompt
        prompt = f"""{self.system_prompt}

Context Information:
{context}

User Question: {query}

Please provide a helpful, accurate response based only on the context provided."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single end-to-end test"""
        print(f"\n{'='*60}")
        print(f"Test: {test_case['id']}")
        print(f"Question: {test_case['question']}")
        print(f"{'='*60}")
        
        result = {
            "test_id": test_case["id"],
            "question": test_case["question"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Retrieve relevant chunks
        print("\n1. RETRIEVAL PHASE")
        start_time = time.time()
        
        chunks, metadata = self.retriever.retrieve(test_case["question"])
        retrieval_time = time.time() - start_time
        
        print(f"   Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        
        # Show retrieval details
        result["retrieval"] = {
            "num_chunks": len(chunks),
            "time_seconds": retrieval_time,
            "complexity": metadata.get("complexity", "unknown"),
            "chunks_preview": []
        }
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            print(f"\n   Chunk {i+1} (score: {chunk.get('score', 0):.3f}):")
            print(f"   Source: {chunk.get('metadata', {}).get('source', 'unknown')}")
            preview = chunk['document'][:150] + "..." if len(chunk['document']) > 150 else chunk['document']
            print(f"   Text: {preview}")
            
            # Check for keyword boosts
            if 'boosted_keywords' in chunk:
                print(f"   Boosted keywords: {chunk['boosted_keywords']}")
            
            result["retrieval"]["chunks_preview"].append({
                "score": chunk.get('score', 0),
                "source": chunk.get('metadata', {}).get('source', 'unknown'),
                "boosted": chunk.get('boosted_keywords', [])
            })
        
        # Step 2: Generate LLM response
        print("\n2. GENERATION PHASE")
        start_time = time.time()
        
        context_chunks = [chunk['document'] for chunk in chunks]
        llm_response = self.generate_llm_response(test_case["question"], context_chunks)
        generation_time = time.time() - start_time
        
        print(f"   Generated response in {generation_time:.2f}s")
        print(f"\n   Response preview:")
        print(f"   {llm_response[:200]}...")
        
        result["generation"] = {
            "response": llm_response,
            "time_seconds": generation_time
        }
        
        # Step 3: Validate response
        print("\n3. VALIDATION PHASE")
        validation_result = self.validator.validate(
            test_case["question"],
            chunks,
            llm_response,
            test_case["question"]  # Pass original_query as required
        )
        
        print(f"   Hallucination check: {'PASSED' if not validation_result.get('hallucinations') else 'FAILED'}")
        print(f"   Citation check: {'PASSED' if not validation_result.get('uncited_claims') else 'FAILED'}")
        print(f"   Out of scope: {'Yes' if validation_result.get('out_of_scope') else 'No'}")
        
        result["validation"] = validation_result
        
        # Step 4: Check expected content
        print("\n4. CONTENT VERIFICATION")
        missing_facts = []
        found_facts = []
        
        response_lower = llm_response.lower()
        for fact in test_case["expected_facts"]:
            if fact.lower() in response_lower:
                found_facts.append(fact)
            else:
                missing_facts.append(fact)
        
        # Check critical values specifically
        critical_missing = []
        for critical in test_case.get("critical_values", []):
            if critical.lower() not in response_lower:
                critical_missing.append(critical)
        
        print(f"   Found facts: {found_facts}")
        print(f"   Missing facts: {missing_facts}")
        if critical_missing:
            print(f"   CRITICAL VALUES MISSING: {critical_missing}")
        
        result["content_check"] = {
            "found_facts": found_facts,
            "missing_facts": missing_facts,
            "critical_missing": critical_missing,
            "passed": len(missing_facts) == 0 and len(critical_missing) == 0
        }
        
        # Overall status
        result["overall_passed"] = (
            result["content_check"]["passed"] and
            not validation_result.get('hallucinations') and
            not validation_result.get('uncited_claims')
        )
        
        print(f"\n   OVERALL: {'✅ PASSED' if result['overall_passed'] else '❌ FAILED'}")
        
        return result
    
    def run_all_tests(self):
        """Run all test cases"""
        print(f"\nStarting End-to-End Testing with Google Gemini")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Number of tests: {len(self.test_cases)}")
        
        results = {
            "test_run": datetime.now().isoformat(),
            "model": "gemini-1.5-flash",
            "tests": [],
            "summary": {
                "total": len(self.test_cases),
                "passed": 0,
                "failed": 0
            }
        }
        
        for test_case in self.test_cases:
            try:
                result = self.run_single_test(test_case)
                results["tests"].append(result)
                
                if result["overall_passed"]:
                    results["summary"]["passed"] += 1
                else:
                    results["summary"]["failed"] += 1
                    
            except Exception as e:
                print(f"\n❌ Test {test_case['id']} failed with error: {e}")
                results["tests"].append({
                    "test_id": test_case["id"],
                    "error": str(e),
                    "overall_passed": False
                })
                results["summary"]["failed"] += 1
        
        # Save results
        output_file = f"gemini_e2e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']} ({results['summary']['passed']/results['summary']['total']*100:.1f}%)")
        print(f"Failed: {results['summary']['failed']}")
        print(f"\nResults saved to: {output_file}")
        
        return results


if __name__ == "__main__":
    # Use the provided API key
    API_KEY = os.getenv('GOOGLE_AI_API_KEY', '')
    
    print("Initializing Gemini E2E Tester...")
    tester = GeminiEndToEndTester(API_KEY)
    
    print("\nRunning end-to-end tests...")
    results = tester.run_all_tests()