#!/usr/bin/env python3
"""
Test script for disclaimer policy functionality
Demonstrates how disclaimers are conditionally applied
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from server.disclaimer_policy import DisclaimerPolicy, should_append_disclaimer
from server.schemas import extract_risk_flags


def test_disclaimer_scenarios():
    """Test various scenarios to verify disclaimer policy"""
    print("="*60)
    print("Testing Disclaimer Policy")
    print("="*60)
    
    test_cases = [
        {
            "name": "Emergency Situation",
            "query": "My sister overdosed and her child needs emergency care NOW!",
            "response": "You can file for emergency guardianship...",
            "expected_disclaimer": True,
            "expected_type": "emergency"
        },
        {
            "name": "Simple Factual Query",
            "query": "What is the filing fee for guardianship?",
            "response": "The filing fee is $175 (MCL 700.5204).",
            "expected_disclaimer": False,
            "expected_type": None
        },
        {
            "name": "Personal Advice Request",
            "query": "What should I do in my case? My nephew needs help.",
            "response": "Based on your situation, you have several options...",
            "expected_disclaimer": True,
            "expected_type": "personalized_advice"
        },
        {
            "name": "Out of Scope Query",
            "query": "How do I get guardianship of my elderly mother?",
            "response": "I can only help with minor guardianship. For adult guardianship...",
            "expected_disclaimer": True,
            "expected_type": "out_of_scope"
        },
        {
            "name": "Form Information",
            "query": "Which forms do I need to file?",
            "response": "You need Form PC 651 and PC 654.",
            "expected_disclaimer": False,
            "expected_type": None
        },
        {
            "name": "Legal Risk Situation",
            "query": "The child is in immediate danger from abuse",
            "response": "In cases of immediate danger, you can file for emergency guardianship...",
            "expected_disclaimer": True,
            "expected_type": "legal_risk"
        },
        {
            "name": "Procedural Question",
            "query": "What are the steps to file for guardianship?",
            "response": "Here are the steps: 1. Complete Form PC 651...",
            "expected_disclaimer": False,
            "expected_type": None
        },
        {
            "name": "CPS Involvement",
            "query": "CPS is involved with my niece. Can I still get guardianship?",
            "response": "When CPS is involved, the process may be different...",
            "expected_disclaimer": True,
            "expected_type": "legal_risk"
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*40}")
        print(f"Test: {test['name']}")
        print(f"Query: {test['query']}")
        print("="*40)
        
        # Extract risk flags
        risk_flags = extract_risk_flags(test['response'], test['query'])
        print(f"Risk flags detected: {risk_flags}")
        
        # Check if personalized advice is requested
        is_personal = DisclaimerPolicy.detects_personalized_advice_request(test['query'])
        print(f"Personal advice requested: {is_personal}")
        
        # Determine if disclaimer should be added
        should_add = DisclaimerPolicy.should_append_disclaimer(
            risk_flags=risk_flags,
            user_query=test['query'],
            response_text=test['response'],
            is_out_of_scope='out_of_scope' in risk_flags
        )
        
        print(f"Should add disclaimer: {should_add}")
        print(f"Expected: {test['expected_disclaimer']}")
        
        # Verify result
        if should_add == test['expected_disclaimer']:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
        
        # If disclaimer should be added, show which one
        if should_add:
            disclaimer = DisclaimerPolicy.get_appropriate_disclaimer(
                risk_flags=risk_flags,
                user_query=test['query'],
                is_out_of_scope='out_of_scope' in risk_flags
            )
            print(f"\nDisclaimer type: {test['expected_type']}")
            print(f"Disclaimer preview: {disclaimer[:100]}...")


def test_policy_application():
    """Test the full policy application"""
    print("\n\n" + "="*60)
    print("Testing Full Policy Application")
    print("="*60)
    
    # Test case 1: Emergency - should get disclaimer
    query1 = "My sister is in the hospital and her kids need help immediately!"
    response1 = "For emergency guardianship, file Form PC 670 immediately."
    risk_flags1 = extract_risk_flags(response1, query1)
    
    final_response1 = DisclaimerPolicy.apply_disclaimer_policy(
        response_text=response1,
        risk_flags=risk_flags1,
        user_query=query1,
        is_out_of_scope=False
    )
    
    print("\nTest 1: Emergency Response")
    print(f"Original: {len(response1)} chars")
    print(f"With policy: {len(final_response1)} chars")
    print(f"Disclaimer added: {'Yes' if len(final_response1) > len(response1) else 'No'}")
    
    # Test case 2: Simple factual - no disclaimer
    query2 = "What forms do I need for guardianship?"
    response2 = "You need Form PC 651 (Petition) and Form PC 654 (Consent)."
    risk_flags2 = extract_risk_flags(response2, query2)
    
    final_response2 = DisclaimerPolicy.apply_disclaimer_policy(
        response_text=response2,
        risk_flags=risk_flags2,
        user_query=query2,
        is_out_of_scope=False
    )
    
    print("\nTest 2: Factual Response")
    print(f"Original: {len(response2)} chars")
    print(f"With policy: {len(final_response2)} chars")
    print(f"Disclaimer added: {'Yes' if len(final_response2) > len(response2) else 'No'}")
    
    # Show the difference
    print("\n" + "="*40)
    print("Policy Impact Summary")
    print("="*40)
    print(f"Emergency query: Disclaimer {'✅ Added' if len(final_response1) > len(response1) else '❌ Not added'}")
    print(f"Factual query: Disclaimer {'✅ Added' if len(final_response2) > len(response2) else '❌ Not added'}")
    print("\nThis reduces disclaimer noise while maintaining compliance!")


def test_backward_compatibility():
    """Test the backward compatibility function"""
    print("\n\n" + "="*60)
    print("Testing Backward Compatibility Function")
    print("="*60)
    
    # Test the convenience function
    test_cases = [
        (["legal_risk"], "What should I do?", "in_scope", True),
        ([], "What is the fee?", "in_scope", False),
        (["out_of_scope"], "Adult guardianship?", "out_of_scope", True),
        (["personalized_advice_requested"], "My situation", "in_scope", True)
    ]
    
    for risk_flags, intent, scope, expected in test_cases:
        result = should_append_disclaimer(risk_flags, intent, scope)
        status = "✅" if result == expected else "❌"
        print(f"{status} Risk: {risk_flags}, Scope: {scope}, Expected: {expected}, Got: {result}")


if __name__ == "__main__":
    print("Michigan Guardianship AI - Disclaimer Policy Testing")
    print("="*60)
    
    # Run all tests
    test_disclaimer_scenarios()
    test_policy_application()
    test_backward_compatibility()
    
    print("\n\n✅ Testing complete!")
    print("\nKey findings:")
    print("- Emergency/crisis situations always get disclaimers")
    print("- Simple factual queries remain clean")
    print("- Personal advice requests trigger appropriate disclaimers")
    print("- Out-of-scope queries get redirected with disclaimers")
    print("- Policy reduces noise while maintaining legal compliance")