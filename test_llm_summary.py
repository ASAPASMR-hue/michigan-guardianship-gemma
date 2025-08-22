#!/usr/bin/env python3
"""
Test if LLM summarization is working correctly
"""

import os
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Test the fixed function
from scripts.llm_handler import LLMHandler

print("=" * 60)
print("🧪 Testing LLM Summarization")
print("=" * 60)

# Check API key
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ No Google API key found")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:20]}...")

# Test the LLM handler directly
llm_handler = LLMHandler(timeout=30)

test_text = """
This document describes the process for filing a guardianship petition in Genesee County Probate Court. 
The petitioner must file form PC651 within 14 days of the initial hearing. A filing fee of $175 is required. 
The court will schedule a hearing on Thursday mornings at 9:00 AM. All parties must be notified at least 
7 days before the hearing date. ICWA compliance is required for Native American children.
"""

prompt = f"""Summarize this legal document chunk in 1-2 sentences, focusing on key procedures, requirements, or deadlines:

{test_text}

Summary:"""

print("\n📝 Testing LLM call...")
print(f"Model: google/gemini-flash-1.5-8b")
print(f"API: google_ai")

try:
    result = llm_handler.call_llm(
        messages=[{"role": "user", "content": prompt}],
        model_id="google/gemini-flash-1.5-8b",
        model_api="google_ai",
        max_tokens=100,
        temperature=0.3
    )
    
    if result.get('error'):
        print(f"❌ LLM Error: {result['error']}")
    elif result.get('response'):
        print(f"✅ Summary generated successfully:")
        print(f"   {result['response']}")
        print(f"   Latency: {result.get('latency', 0):.2f}s")
        print(f"   Cost: ${result.get('cost_usd', 0):.6f}")
    else:
        print("❌ No response received")
        
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "=" * 60)
print("Test complete!")
