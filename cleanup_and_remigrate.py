#!/usr/bin/env python3
"""
Clean up Pinecone index and re-run migration with fixed IDs
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment
load_dotenv()

print("=" * 60)
print("🧹 Cleaning up Pinecone index and re-running migration")
print("=" * 60)

# Initialize Pinecone
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    print("❌ PINECONE_API_KEY not found")
    sys.exit(1)

pc = Pinecone(api_key=api_key)
index_name = "michigan-guardianship-v2"

try:
    # Connect to index
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    print(f"\n📊 Current index stats:")
    print(f"  Vectors: {stats.get('total_vector_count', 0)}")
    print(f"  Namespaces: {list(stats.get('namespaces', {}).keys())}")
    
    # Delete all vectors in the namespace
    if stats.get('total_vector_count', 0) > 0:
        print("\n🗑️  Deleting existing vectors...")
        index.delete(delete_all=True, namespace="genesee-county")
        print("✅ Vectors deleted")
    
    print("\n🚀 Running migration with fixed ASCII IDs...")
    
    # Set the API key for Google
    os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyBbKoIlva0P6TTzOOK3L830IAJrQX7tqko'
    
    # Run the migration
    import subprocess
    result = subprocess.run([
        sys.executable,
        "scripts/migrate_to_pinecone.py",
        "--test"
    ])
    
    if result.returncode == 0:
        print("\n✅ Migration completed successfully!")
        
        # Check final stats
        final_stats = index.describe_index_stats()
        print(f"\n📈 Final stats:")
        print(f"  Total vectors: {final_stats.get('total_vector_count', 0)}")
        
        if final_stats.get('total_vector_count', 0) >= 100:
            print("\n🎉 Phase 1 COMPLETE! All vectors uploaded successfully!")
        else:
            print(f"\n⚠️  Only {final_stats.get('total_vector_count', 0)} vectors uploaded. Check for errors.")
    else:
        print("\n❌ Migration failed. Check the error messages above.")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
