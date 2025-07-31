#!/usr/bin/env python3
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
    
    print("\n✓ Quick test complete!")

if __name__ == "__main__":
    main()
