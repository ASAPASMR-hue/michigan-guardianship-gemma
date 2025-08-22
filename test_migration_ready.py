#!/usr/bin/env python3
"""
Quick test to verify if the migration setup will work
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("🧪 Testing Phase 1 Migration Setup")
print("=" * 60)

# Test 1: Check environment variables
print("\n1️⃣ Checking Environment Variables...")
from dotenv import load_dotenv
load_dotenv()

api_keys = {
    'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
}

for key, value in api_keys.items():
    if value:
        print(f"   ✅ {key}: Found ({value[:20]}...)")
    else:
        print(f"   ❌ {key}: Missing")

# Test 2: Check if kb_files exist
print("\n2️⃣ Checking Knowledge Base Files...")
kb_dir = Path(__file__).parent / "kb_files"
if kb_dir.exists():
    pdf_files = list(kb_dir.glob("**/*.pdf"))
    txt_files = list(kb_dir.glob("**/*.txt"))
    print(f"   ✅ Found {len(pdf_files)} PDFs and {len(txt_files)} text files")
else:
    print(f"   ❌ kb_files directory not found")

# Test 3: Try importing required modules
print("\n3️⃣ Testing Module Imports...")
imports_to_test = [
    ("pinecone", "Pinecone API"),
    ("google.generativeai", "Google AI"),
    ("pdfplumber", "PDF processing"),
    ("chromadb", "ChromaDB (for comparison)"),
    ("tqdm", "Progress bars"),
    ("colorlog", "Colored logging")
]

for module_name, description in imports_to_test:
    try:
        __import__(module_name)
        print(f"   ✅ {description}: Available")
    except ImportError as e:
        print(f"   ❌ {description}: Not installed ({module_name})")

# Test 4: Check if migration script can import embed_kb_cloud functions
print("\n4️⃣ Testing Migration Script Imports...")
try:
    sys.path.append(str(Path(__file__).parent))
    from scripts.embed_kb_cloud import (
        extract_metadata_from_text,
        chunk_text,
        load_documents,
        embed_with_google_ai
    )
    print("   ✅ All functions from embed_kb_cloud.py can be imported")
except ImportError as e:
    print(f"   ❌ Import error: {e}")

# Test 5: Check Pinecone configuration
print("\n5️⃣ Checking Pinecone Configuration...")
import yaml
config_path = Path(__file__).parent / "config" / "pinecone.yaml"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    index_name = config['index_config']['name']
    dimension = config['index_config']['dimension']
    print(f"   ✅ Config loaded: {index_name} with {dimension} dimensions")
else:
    print(f"   ❌ pinecone.yaml not found")

# Test 6: Quick Pinecone connection test
print("\n6️⃣ Testing Pinecone Connection...")
if api_keys['PINECONE_API_KEY']:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_keys['PINECONE_API_KEY'])
        indexes = pc.list_indexes()
        print(f"   ✅ Connected to Pinecone (found {len(indexes.indexes)} indexes)")
        for idx in indexes.indexes:
            print(f"      - {idx.name}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
else:
    print("   ⏭️ Skipped (no API key)")

# Summary
print("\n" + "=" * 60)
print("📊 Summary")
print("=" * 60)

all_good = all([
    api_keys['PINECONE_API_KEY'],
    api_keys['GOOGLE_API_KEY'],
    kb_dir.exists(),
    config_path.exists()
])

if all_good:
    print("✅ Your setup appears to be ready!")
    print("\nNext steps:")
    print("1. Install any missing Python packages")
    print("2. Run: python scripts/setup_pinecone.py")
    print("3. Run: python scripts/migrate_to_pinecone.py --dry-run")
else:
    print("⚠️ Some issues need to be resolved first")
    print("\nPlease fix the issues marked with ❌ above")
