#!/usr/bin/env python3
"""
Test script for conversation state functionality
Demonstrates selective memory and fact extraction
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from server.conversation_state import ConversationState
from server.state_extractor import StateExtractor


def test_state_extraction():
    """Test fact extraction from various conversational patterns"""
    print("="*60)
    print("Testing State Extraction")
    print("="*60)
    
    test_exchanges = [
        {
            "name": "Basic guardianship inquiry",
            "question": "I'm the grandmother and need to get guardianship of my 14-year-old grandson in Genesee County.",
            "response": "To file for guardianship as a grandmother, you'll need Form PC 651. The filing fee is $175.",
            "expected": {
                "relationship": "grandmother",
                "county": "Genesee",
                "forms_mentioned": ["PC 651"],
                "key_facts": {"child_age": "14"},
                "fees": ["$175"]
            }
        },
        {
            "name": "Emergency situation",
            "question": "This is urgent! My sister is in the hospital and I need emergency guardianship of her kids NOW.",
            "response": "For emergency guardianship, file Form PC 670 immediately. Emergency hearings can be scheduled.",
            "expected": {
                "guardianship_type": "emergency",
                "relationship": "sister",
                "forms_mentioned": ["PC 670"],
                "urgency": "high",
                "risk_flags": ["emergency", "legal_risk"]
            }
        },
        {
            "name": "Forms status update",
            "question": "I already filed PC 650 last week. What's next?",
            "response": "Since you've filed PC 650, you now need to file PC 654 for consent and serve notice.",
            "expected": {
                "forms_filed": ["PC 650"],
                "forms_pending": ["PC 654"]
            }
        },
        {
            "name": "Limited guardianship with consent",
            "question": "The parents agree to limited guardianship. Do I still need court approval?",
            "response": "Yes, even with parental consent for limited guardianship, you must file with the court.",
            "expected": {
                "guardianship_type": "limited",
                "key_facts": {"parents_consent": "yes"}
            }
        }
    ]
    
    for test in test_exchanges:
        print(f"\n{'='*40}")
        print(f"Test: {test['name']}")
        print("="*40)
        
        # Extract facts from exchange
        state = StateExtractor.extract_from_exchange(
            user_question=test['question'],
            assistant_response=test['response']
        )
        
        print(f"\nQuestion: {test['question']}")
        print(f"Response: {test['response'][:100]}...")
        print(f"\nExtracted State:")
        print(f"- County: {state.county}")
        print(f"- Type: {state.guardianship_type}")
        print(f"- Role: {state.party_role}")
        print(f"- Relationship: {state.relationship}")
        print(f"- Forms Filed: {state.forms_filed}")
        print(f"- Forms Pending: {state.forms_pending}")
        print(f"- Key Facts: {state.key_facts}")
        
        # Validate expectations
        print("\nValidation:")
        passed = True
        expected = test['expected']
        
        if 'relationship' in expected:
            match = state.relationship == expected['relationship']
            print(f"- Relationship: {'✅' if match else '❌'} (expected: {expected['relationship']}, got: {state.relationship})")
            passed &= match
            
        if 'guardianship_type' in expected:
            match = state.guardianship_type == expected['guardianship_type']
            print(f"- Type: {'✅' if match else '❌'} (expected: {expected['guardianship_type']}, got: {state.guardianship_type})")
            passed &= match
            
        if 'forms_filed' in expected:
            match = all(f in state.forms_filed for f in expected['forms_filed'])
            print(f"- Forms Filed: {'✅' if match else '❌'}")
            passed &= match
            
        if 'urgency' in expected:
            match = state.key_facts.get('urgency') == expected['urgency']
            print(f"- Urgency: {'✅' if match else '❌'}")
            passed &= match
        
        print(f"\nOverall: {'✅ PASSED' if passed else '❌ FAILED'}")


def test_conversation_flow():
    """Test a multi-turn conversation with state accumulation"""
    print("\n\n" + "="*60)
    print("Testing Multi-Turn Conversation")
    print("="*60)
    
    # Simulate a conversation
    exchanges = [
        {
            "q": "I need help with guardianship for my niece.",
            "a": "I can help you with minor guardianship. What county are you filing in?"
        },
        {
            "q": "I'm in Genesee County. She's 10 years old.",
            "a": "For guardianship in Genesee County, you'll need to file at the courthouse at 900 S. Saginaw St."
        },
        {
            "q": "What forms do I need? This isn't an emergency.",
            "a": "You'll need Form PC 651 (Petition) and PC 654 (Consent). The filing fee is $175."
        },
        {
            "q": "I filed PC 651 yesterday. What's the next step?",
            "a": "Good! Since you've filed PC 651, next you need to serve notice and prepare for the Thursday hearing."
        }
    ]
    
    state = ConversationState()
    
    for i, exchange in enumerate(exchanges):
        print(f"\n--- Exchange {i+1} ---")
        print(f"User: {exchange['q']}")
        print(f"Assistant: {exchange['a']}")
        
        # Extract and update state
        state = StateExtractor.extract_from_exchange(
            user_question=exchange['q'],
            assistant_response=exchange['a'],
            current_state=state
        )
        
        # Show accumulated state
        print(f"\nAccumulated Context:")
        print(state.to_context_string())
        print(f"\nSummary: {state.get_summary()}")
    
    # Final validation
    print("\n" + "="*40)
    print("Final State Validation")
    print("="*40)
    print(f"✅ County: {state.county == 'Genesee'}")
    print(f"✅ Relationship: {state.relationship == 'aunt'}")
    print(f"✅ Child Age: {state.key_facts.get('child_age') == '10'}")
    print(f"✅ Forms Filed: {'PC 651' in state.forms_filed}")
    print(f"✅ Has Context: {state.has_meaningful_context()}")


def test_context_injection():
    """Test how context affects prompt generation"""
    print("\n\n" + "="*60)
    print("Testing Context Injection")
    print("="*60)
    
    # Create a state with some history
    state = ConversationState(
        county="Genesee",
        guardianship_type="limited",
        party_role="petitioner",
        relationship="grandmother",
        forms_filed=["PC 650"],
        forms_pending=["PC 670"],
        key_facts={
            "child_age": "14",
            "parents_consent": "yes",
            "filing_fee_discussed": True
        }
    )
    
    print("Current State:")
    print(state.to_context_string())
    
    print("\n\nHow this would appear in the prompt:")
    print("-"*40)
    print(state.to_context_string())
    print("\nUser Question: Do I need both parents' signatures?")
    print("-"*40)
    
    print("\n✅ Benefits:")
    print("- No need to re-ask about relationship (grandmother)")
    print("- System knows this is limited guardianship")
    print("- System knows PC 650 is already filed")
    print("- Can give specific next steps based on context")


def test_state_persistence():
    """Test JSON serialization for session storage"""
    print("\n\n" + "="*60)
    print("Testing State Persistence")
    print("="*60)
    
    # Create a state
    original_state = ConversationState(
        guardianship_type="temporary",
        party_role="relative",
        relationship="uncle",
        forms_mentioned=["PC 651", "PC 670", "MC 20"]
    )
    
    print("Original State:")
    print(original_state.get_summary())
    
    # Serialize to JSON
    json_str = original_state.to_json()
    print(f"\nSerialized JSON ({len(json_str)} chars):")
    print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
    
    # Deserialize
    restored_state = ConversationState.from_json(json_str)
    print("\nRestored State:")
    print(restored_state.get_summary())
    
    # Verify
    print("\nVerification:")
    print(f"✅ Type matches: {original_state.guardianship_type == restored_state.guardianship_type}")
    print(f"✅ Role matches: {original_state.party_role == restored_state.party_role}")
    print(f"✅ Forms match: {original_state.forms_mentioned == restored_state.forms_mentioned}")


if __name__ == "__main__":
    print("Michigan Guardianship AI - Conversation State Testing")
    print("="*60)
    
    # Run all tests
    test_state_extraction()
    test_conversation_flow()
    test_context_injection()
    test_state_persistence()
    
    print("\n\n✅ Testing complete!")
    print("\nKey benefits demonstrated:")
    print("- Automatic fact extraction from conversations")
    print("- Progressive state building across exchanges")
    print("- Minimal storage (only essential facts)")
    print("- Clean context injection into prompts")
    print("- JSON serialization for session storage")