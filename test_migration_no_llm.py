#!/usr/bin/env python3
"""
Quick test to run migration with LLM summarization disabled
"""

import subprocess
import sys

print("🚀 Running migration with LLM summarization disabled...")
print("=" * 60)

# Set environment variable to disable summarization
import os
os.environ['DISABLE_LLM_SUMMARY'] = 'true'

# Run the migration script
result = subprocess.run([
    sys.executable,
    'scripts/migrate_to_pinecone.py',
    '--dry-run',
    '--verbose'
], capture_output=False, text=True)

sys.exit(result.returncode)
