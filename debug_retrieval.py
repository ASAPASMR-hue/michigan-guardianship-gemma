#!/usr/bin/env python3
"""
Debug retrieval issues - examine what's happening during search
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.retrieval_setup import HybridRetriever
import chromadb
from chromadb.config import Settings

def debug_retrieval():
    """Debug retrieval process"""
    print("=== Debugging Retrieval ===\n")
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Check what's in the database
    print(f"Total documents in database: {len(retriever.all_docs)}")
    print("\nDocument samples:")
    for i, (doc_id, doc, metadata) in enumerate(zip(retriever.all_ids[:5], retriever.all_docs[:5], retriever.all_metadata[:5])):
        print(f"\n{i+1}. ID: {doc_id}")
        print(f"   Source: {metadata.get('source', 'Unknown')}")
        print(f"   Text: {doc[:200]}...")
    
    # Test specific queries
    test_queries = [
        "Where is the court located?",
        "court address",
        "Genesee County Probate Court",
        "900 S. Saginaw"
    ]
    
    print("\n" + "="*60)
    print("Testing queries with detailed logging:")
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Get embeddings
        query_embedding = retriever.embed_model.encode(query, normalize_embeddings=True)
        
        # Direct vector search
        vector_results = retriever.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10,
            where={"jurisdiction": "Genesee County"}
        )
        
        print(f"\nVector search results ({len(vector_results['ids'][0])} found):")
        for i, (doc_id, distance) in enumerate(zip(vector_results['ids'][0][:3], vector_results['distances'][0][:3])):
            idx = retriever.all_ids.index(doc_id)
            print(f"  {i+1}. Distance: {distance:.4f}, Text: {retriever.all_docs[idx][:100]}...")
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = retriever.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        
        print(f"\nBM25 search results (top 3):")
        for i, idx in enumerate(top_bm25_indices[:3]):
            if retriever.all_metadata[idx].get('jurisdiction') == 'Genesee County':
                print(f"  {i+1}. Score: {bm25_scores[idx]:.4f}, Text: {retriever.all_docs[idx][:100]}...")
        
        # Hybrid search
        results = retriever.hybrid_search(query, top_k=5)
        print(f"\nHybrid search results ({len(results)} found):")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Combined score: {result['score']:.4f}, Text: {result['document'][:100]}...")
    
    # Check for specific content
    print("\n" + "="*60)
    print("Searching for specific content patterns:")
    
    patterns = ["900 S. Saginaw", "Flint", "court location", "probate court"]
    for pattern in patterns:
        print(f"\nPattern: '{pattern}'")
        found = False
        for i, doc in enumerate(retriever.all_docs):
            if pattern.lower() in doc.lower():
                print(f"  Found in document {i}: {retriever.all_metadata[i].get('source', 'Unknown')}")
                print(f"  Context: ...{doc[max(0, doc.lower().find(pattern.lower())-50):doc.lower().find(pattern.lower())+100]}...")
                found = True
                break
        if not found:
            print("  Not found in any document!")

if __name__ == "__main__":
    debug_retrieval()