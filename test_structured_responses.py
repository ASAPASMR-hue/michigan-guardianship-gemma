#!/usr/bin/env python3
"""
Test script for structured response generation
Demonstrates how the backend returns structured JSON
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.production_pipeline import GuardianshipRAG
from server.schemas import AnswerPayload, extract_forms_from_text, extract_fees_from_text, extract_risk_flags


def test_extraction_functions():
    """Test the extraction helper functions"""
    print("="*60)
    print("Testing extraction functions")
    print("="*60)
    
    # Test form extraction
    test_text = "You need to file Form PC 651 and PC 654. If you can't afford the fee, use MC 20."
    forms = extract_forms_from_text(test_text)
    print(f"\nForm extraction test:")
    print(f"Text: {test_text}")
    print(f"Extracted forms: {forms}")
    assert "PC 651" in forms
    assert "PC 654" in forms
    assert "MC 20" in forms
    
    # Test fee extraction
    test_text = "The filing fee is $175. You may request a fee waiver if you qualify."
    fees = extract_fees_from_text(test_text)
    print(f"\nFee extraction test:")
    print(f"Text: {test_text}")
    print(f"Extracted fees: {fees}")
    assert any("$175" in fee for fee in fees)
    
    # Test risk flag extraction
    test_text = "This is an emergency situation requiring immediate action."
    question = "My child is in immediate danger and needs protection"
    risk_flags = extract_risk_flags(test_text, question)
    print(f"\nRisk flag extraction test:")
    print(f"Text: {test_text}")
    print(f"Question: {question}")
    print(f"Extracted risk flags: {risk_flags}")
    assert "legal_risk" in risk_flags
    
    print("\n✅ All extraction tests passed!")


def test_structured_response():
    """Test the complete structured response generation"""
    print("\n" + "="*60)
    print("Testing structured response generation")
    print("="*60)
    
    # Initialize the RAG pipeline
    print("\nInitializing GuardianshipRAG pipeline...")
    rag = GuardianshipRAG(model_name="google/gemma-3-4b-it")
    
    # Test questions
    test_questions = [
        {
            "question": "What is the filing fee for guardianship in Genesee County?",
            "expected": {
                "fees": ["$175"],
                "forms": [],
                "risk_flags": []
            }
        },
        {
            "question": "How do I file for emergency guardianship? What forms do I need?",
            "expected": {
                "fees": ["$175"],
                "forms": ["PC 651", "PC 670"],
                "risk_flags": ["legal_risk"]
            }
        },
        {
            "question": "Can I get help with adult guardianship for my elderly parent?",
            "expected": {
                "fees": [],
                "forms": [],
                "risk_flags": ["out_of_scope"]
            }
        }
    ]
    
    for i, test_case in enumerate(test_questions):
        print(f"\n{'='*40}")
        print(f"Test Case {i+1}: {test_case['question']}")
        print("="*40)
        
        try:
            # Get structured response
            result = rag.get_answer(test_case['question'])
            
            # Extract the structured data
            if 'data' in result:
                data = result['data']
                print(f"\nStructured Response:")
                print(f"- Forms: {data.get('forms', [])}")
                print(f"- Fees: {data.get('fees', [])}")
                print(f"- Risk Flags: {data.get('risk_flags', [])}")
                print(f"- Citations: {len(data.get('citations', []))} found")
                print(f"- Steps: {len(data.get('steps', []))} found")
                
                # Show answer preview
                answer = data.get('answer_markdown', '')
                print(f"\nAnswer Preview (first 200 chars):")
                print(answer[:200] + "..." if len(answer) > 200 else answer)
                
                # Validate against expected
                print(f"\nValidation:")
                expected = test_case['expected']
                
                # Check fees
                if expected['fees']:
                    fees_found = any(exp_fee in str(data.get('fees', [])) for exp_fee in expected['fees'])
                    print(f"- Fees: {'✅ Found' if fees_found else '❌ Missing'}")
                
                # Check forms
                if expected['forms']:
                    forms_found = all(form in data.get('forms', []) for form in expected['forms'])
                    print(f"- Forms: {'✅ Found' if forms_found else '❌ Missing'}")
                
                # Check risk flags
                if expected['risk_flags']:
                    flags_found = any(flag in data.get('risk_flags', []) for flag in expected['risk_flags'])
                    print(f"- Risk Flags: {'✅ Found' if flags_found else '❌ Missing'}")
                
                # Save example response
                if i == 0:  # Save first response as example
                    with open('example_structured_response.json', 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"\n💾 Saved example response to example_structured_response.json")
                
            else:
                print("❌ No structured data in response")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def test_api_endpoints():
    """Test the API endpoints (requires server to be running)"""
    print("\n" + "="*60)
    print("API Endpoint Test Instructions")
    print("="*60)
    
    print("\n1. Start the server:")
    print("   python app.py")
    
    print("\n2. Test the standard endpoint (backward compatible):")
    print("   curl -X POST http://localhost:5000/api/ask \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"question\": \"What is the filing fee?\"}'")
    
    print("\n3. Test the structured endpoint:")
    print("   curl -X POST http://localhost:5000/api/ask/structured \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"question\": \"What forms do I need for guardianship?\"}'")
    
    print("\n4. Check the example response in example_structured_response.json")


if __name__ == "__main__":
    print("Michigan Guardianship AI - Structured Response Testing")
    print("="*60)
    
    # Run tests
    test_extraction_functions()
    test_structured_response()
    test_api_endpoints()
    
    print("\n✅ Testing complete!")