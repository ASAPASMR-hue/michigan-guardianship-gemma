#!/usr/bin/env python3
"""
Quick migration test without LLM summarization
Run this if LLM summarization is causing issues
"""

import os
import sys
from pathlib import Path

# Add path
sys.path.append(str(Path(__file__).parent))

# Temporarily patch the config to disable summarization
import yaml

config_path = Path(__file__).parent / "config" / "pinecone.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Disable summarization
config['migration_settings']['chunk_summarization']['enabled'] = False

# Save temporarily
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("✅ Temporarily disabled LLM summarization")
print("Running migration in dry-run mode...")

# Run migration
import subprocess
result = subprocess.run([
    sys.executable,
    "scripts/migrate_to_pinecone.py",
    "--dry-run",
    "--verbose"
])

# Re-enable summarization
config['migration_settings']['chunk_summarization']['enabled'] = True
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("✅ Re-enabled LLM summarization for future runs")

sys.exit(result.returncode)
