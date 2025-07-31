#!/usr/bin/env python3
"""
Debug test retrieval issues - examine what's happening in the test environment
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

TEST_CHROMA_DIR = Path(__file__).parent / "integration_tests" / ".test_chroma_db"

def debug_test_retrieval():
    """Debug test retrieval process"""
    print("=== Debugging Test Retrieval ===\n")
    
    # Connect to test database
    client = chromadb.PersistentClient(
        path=str(TEST_CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        collection = client.get_collection("test_guardianship")
        print(f"Connected to test collection")
    except:
        print("Test collection not found!")
        return
    
    # Get all documents
    results = collection.get()
    docs = results['documents']
    ids = results['ids']
    metadatas = results['metadatas']
    
    print(f"Total documents in test database: {len(docs)}")
    print("\nAll documents:")
    for i, (doc_id, doc, metadata) in enumerate(zip(ids, docs, metadatas)):
        print(f"\n{i+1}. ID: {doc_id}")
        print(f"   Source: {metadata.get('source', 'Unknown')}")
        print(f"   Text: {doc[:200]}...")
        if "900 S. Saginaw" in doc or "court" in doc.lower():
            print(f"   *** CONTAINS COURT INFO ***")
    
    # Test specific queries with embeddings
    print("\n" + "="*60)
    print("Testing embeddings and similarity:")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    test_queries = [
        "Where is the court located?",
        "court address",
        "900 S. Saginaw",
        "Court Location",
        "Genesee County Probate Court address"
    ]
    
    # Get embeddings for all documents
    doc_embeddings = model.encode(docs, normalize_embeddings=True)
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        query_embedding = model.encode(query, normalize_embeddings=True)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            sim = query_embedding @ doc_embedding  # Cosine similarity
            similarities.append((i, sim, docs[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 most similar chunks:")
        for i, (idx, sim, doc) in enumerate(similarities[:3]):
            print(f"  {i+1}. Similarity: {sim:.4f}")
            print(f"     Text: {doc[:100]}...")
            if "court" in doc.lower() or "900" in doc:
                print(f"     *** CONTAINS RELEVANT INFO ***")
    
    # Check content directly
    print("\n" + "="*60)
    print("Direct content search:")
    
    for pattern in ["court location", "Court Location", "900 S. Saginaw", "Flint, MI"]:
        print(f"\nSearching for '{pattern}':")
        found = False
        for i, doc in enumerate(docs):
            if pattern in doc:
                print(f"  Found in document {i}:")
                start = max(0, doc.find(pattern) - 50)
                end = min(len(doc), doc.find(pattern) + 100)
                print(f"  Context: ...{doc[start:end]}...")
                found = True
        if not found:
            print("  Not found!")

if __name__ == "__main__":
    debug_test_retrieval()