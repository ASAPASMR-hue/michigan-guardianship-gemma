#!/usr/bin/env python3
"""
Test Phase 3 Setup
Quick verification that APIs and components are working
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.llm_handler import LLMHandler
from scripts.log_step import log_step


def test_setup():
    """Test Phase 3 setup"""
    log_step("Testing Phase 3 Setup")
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    log_step("\n1. Checking API Keys...")
    
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    google_key = os.environ.get("GOOGLE_AI_API_KEY", "")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    
    if not openrouter_key or openrouter_key.startswith("sk-or-v1-your"):
        log_step("❌ OpenRouter API key not configured", level="error")
        log_step("   Set OPENROUTER_API_KEY in your .env file")
    else:
        log_step("✅ OpenRouter API key found")
    
    if not google_key or google_key.startswith("AIzaSy-your"):
        log_step("❌ Google AI API key not configured", level="error")
        log_step("   Set GOOGLE_AI_API_KEY in your .env file")
    else:
        log_step("✅ Google AI API key found")
    
    if not hf_token or hf_token.startswith("hf_your"):
        log_step("⚠️  HuggingFace token not configured", level="warning")
        log_step("   Set HUGGINGFACE_TOKEN in your .env file")
    else:
        log_step("✅ HuggingFace token found")
    
    # Test LLM connectivity
    log_step("\n2. Testing Model Connectivity...")
    
    handler = LLMHandler()
    
    # Test one model from each API
    test_models = [
        {
            "id": "google/gemini-2.5-flash-lite",
            "api": "google_ai",
            "name": "Google AI Test"
        },
        {
            "id": "mistralai/mistral-nemo:free",
            "api": "openrouter", 
            "name": "OpenRouter Test"
        }
    ]
    
    for model in test_models:
        log_step(f"\nTesting {model['name']} ({model['id']})...")
        result = handler.test_connectivity(model['id'], model['api'])
        
        if result['success']:
            log_step(f"✅ {model['name']} connectivity successful")
        else:
            log_step(f"❌ {model['name']} failed: {result['error']}", level="error")
    
    # Check for test questions
    log_step("\n3. Checking Test Questions...")
    
    csv_paths = [
        Path("data/Synthetic Test Questions.xlsx"),
        Path("/Users/claytoncanady/Library/michigan-guardianship-ai/Synthetic Test Questions - Sheet1.csv")
    ]
    
    questions_found = False
    for path in csv_paths:
        if path.exists():
            log_step(f"✅ Found test questions at: {path}")
            questions_found = True
            break
    
    if not questions_found:
        log_step("❌ Test questions not found", level="error")
        log_step("   Expected locations:")
        for path in csv_paths:
            log_step(f"   - {path}")
    
    # Check ChromaDB
    log_step("\n4. Checking ChromaDB...")
    
    chroma_path = Path("chroma_db")
    if chroma_path.exists() and any(chroma_path.iterdir()):
        log_step("✅ ChromaDB found")
    else:
        log_step("❌ ChromaDB not found", level="error")
        log_step("   Run: python scripts/embed_kb.py")
    
    log_step("\n" + "="*60)
    log_step("Setup test complete!")
    log_step("To run full evaluation: python scripts/run_full_evaluation.py")
    log_step("="*60)


if __name__ == "__main__":
    test_setup()