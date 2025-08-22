#!/usr/bin/env python3
"""
Migrate Michigan Guardianship KB from ChromaDB to Pinecone
Reuses existing embedding logic from embed_kb_cloud.py
Enhanced with chunk summarization and rich metadata
"""

import os
import sys
import argparse
import yaml
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import logging
from colorlog import ColoredFormatter
import google.generativeai as genai
import pdfplumber
from pinecone import Pinecone
import chromadb
from chromadb.config import Settings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.log_step import log_step
from scripts.llm_handler import LLMHandler

# Load environment variables
load_dotenv()

# Configuration paths (reuse from embed_kb_cloud.py)
CONFIG_DIR = Path(__file__).parent.parent / "config"
KB_DIR = Path(__file__).parent.parent / "kb_files"
DOCS_DIR = Path(__file__).parent.parent / "docs"
CONSTANTS_DIR = Path(__file__).parent.parent / "constants"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Import functions from embed_kb_cloud.py
sys.path.append(str(Path(__file__).parent))
from embed_kb_cloud import (
    extract_metadata_from_text,
    chunk_text,
    load_documents,
    embed_with_google_ai
)

def setup_logging(verbose: bool = False):
    """Setup colored logging"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    console_handler = logging.StreamHandler()
    console_format = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    
    return logger

def load_pinecone_config():
    """Load Pinecone configuration"""
    config_path = CONFIG_DIR / "pinecone.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def enhance_metadata(metadata: Dict, text: str) -> Dict:
    """Enhance metadata with additional fields for Pinecone"""
    # Add complexity level based on content analysis
    if len(text) > 2000 or 'whereas' in text.lower() or 'pursuant to' in text.lower():
        metadata['complexity_level'] = 'complex'
    elif len(text) > 500:
        metadata['complexity_level'] = 'standard'
    else:
        metadata['complexity_level'] = 'simple'
    
    # Determine target audience
    if 'attorney' in text.lower() or 'counsel' in text.lower():
        metadata['target_audience'] = 'attorney'
    elif 'court clerk' in text.lower() or 'filing' in text.lower():
        metadata['target_audience'] = 'court_staff'
    else:
        metadata['target_audience'] = 'petitioner'
    
    # Check for ICWA content
    metadata['has_icwa_content'] = bool(
        'icwa' in text.lower() or 
        'indian child' in text.lower() or 
        'tribal' in text.lower()
    )
    
    # Check for emergency procedures
    metadata['is_emergency'] = bool(
        'emergency' in text.lower() or 
        'immediate' in text.lower() or 
        'urgent' in text.lower()
    )
    
    # Add content length
    metadata['content_length'] = len(text)
    
    # Add timestamp
    metadata['last_updated'] = datetime.now().isoformat()
    
    return metadata

def generate_chunk_summary(text: str, max_retries: int = 2) -> str:
    """Generate a 1-2 sentence summary of the chunk using local LLM"""
    prompt = f"""Summarize this legal document chunk in 1-2 sentences, focusing on key procedures, requirements, or deadlines:

{text[:1000]}  # Limit context to avoid token limits

Summary:"""
    
    # Initialize LLM handler
    llm_handler = LLMHandler(timeout=30)
    
    for attempt in range(max_retries):
        try:
            # Use the LLMHandler's call_llm method with proper message format
            result = llm_handler.call_llm(
                messages=[{"role": "user", "content": prompt}],  # Correct format
                model_id="gemini-1.5-flash",  # Fast model for summaries
                model_api="google_ai",  # Note: should be 'google_ai' not 'google'
                max_tokens=100,
                temperature=0.3
            )
            if result and result.get('response') and len(result['response']) > 10:
                return result['response'].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                logging.warning(f"LLM summarization failed: {e}")
    
    # Fallback to truncation
    return text[:128] + "..." if len(text) > 128 else text

def compare_with_chromadb(query_embeddings: List[List[float]], n_results: int = 5) -> Dict:
    """Query ChromaDB for comparison"""
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection("michigan_guardianship_v2")
        
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where={"jurisdiction": "Genesee County"}
        )
        
        return results
    except Exception as e:
        logging.warning(f"ChromaDB comparison failed: {e}")
        return None

def migrate_to_pinecone(args):
    """Main migration function"""
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("🏛️  Michigan Guardianship RAG - Pinecone Migration")
    logger.info("=" * 60)
    
    # Load configurations
    pinecone_config = load_pinecone_config()
    index_config = pinecone_config['index_config']
    migration_settings = pinecone_config['migration_settings']
    
    # Load existing chunking and embedding configs
    with open(CONFIG_DIR / "chunking.yaml", "r") as f:
        chunking_config = yaml.safe_load(f)
    
    # Check API keys
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    
    if not google_api_key:
        logger.error("❌ GOOGLE_API_KEY not found in environment")
        return 1
    
    if not pinecone_api_key:
        logger.error("❌ PINECONE_API_KEY not found in environment")
        return 1
    
    # Configure Google AI
    genai.configure(api_key=google_api_key)
    
    # Initialize Pinecone
    logger.info("🔧 Initializing Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    
    try:
        index = pc.Index(index_config['name'])
        stats = index.describe_index_stats()
        logger.info(f"✅ Connected to index: {index_config['name']}")
        logger.info(f"  Current vectors: {stats.get('total_vector_count', 0)}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Pinecone: {e}")
        logger.info("💡 Run 'python scripts/setup_pinecone.py' first")
        return 1
    
    # Load documents (reuse existing function)
    logger.info("\n📄 Loading documents...")
    documents = load_documents()
    
    if not documents:
        logger.error("❌ No documents found!")
        return 1
    
    logger.info(f"✅ Loaded {len(documents)} documents")
    
    # Process documents
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    logger.info("\n📝 Processing documents...")
    
    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = chunk_text(doc['content'], chunking_config)
        
        for chunk in chunks:
            # Enhance metadata
            chunk_metadata = doc['metadata'].copy()
            chunk_metadata = enhance_metadata(chunk_metadata, chunk['text'])
            chunk_metadata['chunk_id'] = str(chunk['chunk_id'])
            chunk_metadata['chunk_index'] = chunk['chunk_id']
            
            # Generate chunk summary if enabled
            if migration_settings['chunk_summarization']['enabled']:
                summary = generate_chunk_summary(chunk['text'])
                chunk_metadata['chunk_summary'] = summary
            
            # Clean metadata - ensure all values are strings or basic types
            for key, value in chunk_metadata.items():
                if isinstance(value, bool):
                    chunk_metadata[key] = str(value).lower()
                elif isinstance(value, (int, float)):
                    chunk_metadata[key] = str(value)
                elif value is None:
                    chunk_metadata[key] = ""
            
            all_chunks.append(chunk['text'])
            all_metadata.append(chunk_metadata)
            
            # Create unique ID - ensure it's ASCII-safe for Pinecone
            # Replace non-ASCII characters and sanitize the source name
            safe_source = doc['metadata']['source'].encode('ascii', 'ignore').decode('ascii')
            safe_source = safe_source.replace(' ', '_').replace('–', '-').replace('—', '-')
            safe_source = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in safe_source)
            doc_id = f"{safe_source}_{chunk['chunk_id']}_{hashlib.md5(chunk['text'].encode()).hexdigest()[:8]}"
            all_ids.append(doc_id)
    
    logger.info(f"✅ Created {len(all_chunks)} chunks with enhanced metadata")
    
    if args.dry_run:
        logger.info("\n🧪 DRY RUN MODE - Not uploading to Pinecone")
        logger.info("\nSample chunks:")
        for i in range(min(3, len(all_chunks))):
            logger.info(f"\nChunk {i+1}:")
            logger.info(f"  ID: {all_ids[i]}")
            logger.info(f"  Metadata: {all_metadata[i]}")
            logger.info(f"  Text preview: {all_chunks[i][:200]}...")
        return 0
    
    # Generate embeddings (reuse existing function)
    logger.info("\n🤖 Generating embeddings with Google AI...")
    all_embeddings = embed_with_google_ai(all_chunks, google_api_key)
    
    if not all_embeddings or len(all_embeddings) != len(all_chunks):
        logger.error("❌ Embedding generation failed!")
        return 1
    
    logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
    
    # Upload to Pinecone
    logger.info("\n📤 Uploading to Pinecone...")
    namespace = index_config['namespace']
    batch_size = migration_settings['batch_processing']['chunks_per_batch']
    
    vectors_uploaded = 0
    failed_uploads = []
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Uploading batches"):
        end_idx = min(i + batch_size, len(all_chunks))
        
        # Prepare vectors for upload
        vectors = []
        for j in range(i, end_idx):
            vectors.append({
                'id': all_ids[j],
                'values': all_embeddings[j],
                'metadata': all_metadata[j]
            })
        
        try:
            # Upload to Pinecone
            index.upsert(vectors=vectors, namespace=namespace)
            vectors_uploaded += len(vectors)
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Failed to upload batch {i//batch_size}: {e}")
            failed_uploads.extend(all_ids[i:end_idx])
            continue
    
    # Final statistics
    logger.info("\n" + "=" * 60)
    logger.info("📊 Migration Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Documents processed: {len(documents)}")
    logger.info(f"✅ Chunks created: {len(all_chunks)}")
    logger.info(f"✅ Vectors uploaded: {vectors_uploaded}")
    
    if failed_uploads:
        logger.warning(f"⚠️  Failed uploads: {len(failed_uploads)}")
    
    # Test retrieval
    if args.test:
        logger.info("\n🧪 Testing retrieval...")
        test_queries = [
            "filing fee genesee county",
            "native american guardianship ICWA",
            "emergency guardianship procedure"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            
            # Generate query embedding
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']
            
            # Query Pinecone
            pinecone_results = index.query(
                vector=query_embedding,
                top_k=3,
                namespace=namespace,
                include_metadata=True
            )
            
            # Display results
            for match in pinecone_results.matches:
                logger.info(f"  Score: {match.score:.3f}")
                logger.info(f"  Source: {match.metadata.get('source', 'unknown')}")
                logger.info(f"  Type: {match.metadata.get('doc_type', 'unknown')}")
                if 'chunk_summary' in match.metadata:
                    logger.info(f"  Summary: {match.metadata['chunk_summary']}")
            
            # Compare with ChromaDB if available
            if args.compare:
                chromadb_results = compare_with_chromadb([query_embedding], n_results=3)
                if chromadb_results:
                    logger.info("  ChromaDB comparison:")
                    for doc, distance in zip(chromadb_results['documents'][0], 
                                            chromadb_results['distances'][0]):
                        logger.info(f"    Distance: {distance:.3f}")
    
    logger.info("\n✅ Migration completed successfully!")
    
    # Update index stats
    final_stats = index.describe_index_stats()
    logger.info(f"\n📈 Final Index Stats:")
    logger.info(f"  Total vectors: {final_stats.get('total_vector_count', 0)}")
    logger.info(f"  Namespaces: {list(final_stats.get('namespaces', {}).keys())}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description='Migrate Michigan Guardianship KB to Pinecone'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test run without uploading data'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test retrieval after migration'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare results with ChromaDB'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    sys.exit(migrate_to_pinecone(args))

if __name__ == "__main__":
    main()
