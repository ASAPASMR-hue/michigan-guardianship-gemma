#!/usr/bin/env python3
"""
run_golden_qa.py - Run golden Q&A tests against the system
"""

import json
from pathlib import Path
from datetime import datetime
from full_pipeline_test import IntegrationTester

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
    
    print(f"\nQuick Results:")
    print(f"Pass Rate: {results['summary']['golden_qa_tests']['pass_rate']:.1f}%")
    
    tester.cleanup()

if __name__ == "__main__":
    main()
