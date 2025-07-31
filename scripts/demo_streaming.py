#!/usr/bin/env python3
"""
Demo script showing streaming responses for better UX
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.llm_handler import LLMHandler
from scripts.log_step import log_step


def demo_streaming():
    """Demonstrate streaming responses"""
    load_dotenv()
    
    # Initialize handler
    handler = LLMHandler()
    
    # Test question
    test_question = """
    I'm a grandparent and my daughter is entering rehab for 4 months. 
    She wants me to take care of her 8-year-old son during this time. 
    What type of guardianship should we file for in Genesee County?
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a Michigan minor guardianship assistant serving Genesee County residents."
        },
        {
            "role": "user",
            "content": test_question
        }
    ]
    
    # Test with a fast, free model
    model = {
        "id": "mistralai/mistral-nemo:free",
        "api": "openrouter",
        "name": "Mistral Nemo (Free)"
    }
    
    print(f"\nTesting streaming with {model['name']}...")
    print("="*60)
    print("Question:", test_question.strip())
    print("="*60)
    print("\nResponse (streaming):\n")
    
    # Call with streaming
    start_time = time.time()
    
    try:
        # For now, we'll simulate streaming by showing progress
        # Full streaming implementation would require modifying the LLM handler
        print("[Generating response", end="", flush=True)
        
        result = handler.call_llm(
            model_id=model['id'],
            messages=messages,
            model_api=model['api'],
            temperature=0.1,
            max_tokens=500
        )
        
        # Simulate streaming dots
        for _ in range(5):
            time.sleep(0.2)
            print(".", end="", flush=True)
        
        print("]")
        print("\n" + result['response'])
        
        print(f"\n{'='*60}")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"Cost: ${result['cost_usd']:.6f}")
        if result['error']:
            print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    demo_streaming()