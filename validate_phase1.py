#!/usr/bin/env python3
"""
Final Phase 1 validation and metrics collection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
import json
from datetime import datetime

load_dotenv()

# Set up APIs
os.environ['GOOGLE_AI_API_KEY'] = 'AIzaSyBbKoIlva0P6TTzOOK3L830IAJrQX7tqko'
genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("michigan-guardianship-v2")

print("=" * 60)
print("📊 Phase 1 Final Validation")
print("=" * 60)

# Get index statistics
stats = index.describe_index_stats()
print(f"\n✅ Pinecone Index Stats:")
print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
print(f"  Namespace: genesee-county")
print(f"  Dimensions: 768")

# Test comprehensive queries
test_queries = [
    "What is the filing fee for guardianship in Genesee County?",
    "How do ICWA requirements apply to Native American children?",
    "What forms are needed for emergency guardianship?",
    "Thursday morning hearing schedule",
    "Guardian duties and responsibilities",
    "Terminating a guardianship",
    "PC651 form requirements",
    "Minor's assets management"
]

print(f"\n🔍 Testing {len(test_queries)} queries:")
print("-" * 40)

results_summary = []

for query in test_queries:
    # Generate embedding
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    
    # Query Pinecone
    response = index.query(
        vector=result['embedding'],
        top_k=3,
        namespace="genesee-county",
        include_metadata=True
    )
    
    if response.matches:
        top_score = response.matches[0].score
        top_source = response.matches[0].metadata.get('source', 'unknown')
        summary = response.matches[0].metadata.get('chunk_summary', 'No summary')
        
        print(f"\n📝 Query: {query[:50]}...")
        print(f"   Top Score: {top_score:.3f}")
        print(f"   Source: {top_source}")
        print(f"   Summary: {summary[:100]}...")
        
        results_summary.append({
            "query": query,
            "score": top_score,
            "source": top_source
        })

# Calculate metrics
avg_score = sum(r['score'] for r in results_summary) / len(results_summary)
high_quality = sum(1 for r in results_summary if r['score'] > 0.75)

print("\n" + "=" * 60)
print("📈 Performance Summary")
print("=" * 60)
print(f"✅ Average Relevance Score: {avg_score:.3f}")
print(f"✅ High Quality Results (>0.75): {high_quality}/{len(test_queries)}")
print(f"✅ Success Rate: {(high_quality/len(test_queries))*100:.1f}%")

# Save metrics
metrics = {
    "phase": "1",
    "date": datetime.now().isoformat(),
    "total_vectors": stats.get('total_vector_count', 0),
    "test_queries": len(test_queries),
    "avg_score": avg_score,
    "high_quality_results": high_quality,
    "success_rate": (high_quality/len(test_queries))*100
}

with open("phase1_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n💾 Metrics saved to phase1_metrics.json")

if avg_score > 0.7 and high_quality >= 6:
    print("\n🎉 PHASE 1 VALIDATION: PASSED")
    print("Ready for Phase 2 implementation!")
else:
    print("\n⚠️ Consider tuning before Phase 2")
