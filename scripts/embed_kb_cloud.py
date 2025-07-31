#!/usr/bin/env python3
"""
embed_kb_cloud.py - Cloud-based Document Embedding Pipeline for Michigan Guardianship AI
Uses Google AI Studio's text-embedding-004 API to avoid local memory issues
"""

import os
import sys
import yaml
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Tuple
import pdfplumber
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.log_step import log_step

# Configuration paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
KB_DIR = Path(__file__).parent.parent / "kb_files"
DOCS_DIR = Path(__file__).parent.parent / "docs"
CONSTANTS_DIR = Path(__file__).parent.parent / "constants"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# PDFs to embed (from additional notes)
PDFS_TO_EMBED = [
    "pc651.pdf", "pc654.pdf", "pc562.pdf",  # existing
    "pc650.pdf", "pc652.pdf", "pc670.pdf", "pc675.pdf", "pc564.pdf",  # additional (may not exist)
    "GM-ARI.pdf", "GM-PSH.pdf", "GM-NRS-MINOR.pdf"  # Genesee specific
]

# PDFs that are expected to exist
EXPECTED_PDFS = ["pc651.pdf", "pc654.pdf", "pc562.pdf", "GM-ARI.pdf", "GM-PSH.pdf", "GM-NRS-MINOR.pdf"]

def load_configs():
    """Load configuration files"""
    with open(CONFIG_DIR / "chunking.yaml", "r") as f:
        chunking_config = yaml.safe_load(f)
    
    with open(CONFIG_DIR / "embedding.yaml", "r") as f:
        embedding_config = yaml.safe_load(f)
    
    return chunking_config, embedding_config

def extract_metadata_from_text(text: str) -> Dict:
    """Extract structured metadata from document text"""
    metadata = {
        "doc_type": "guidance",  # default
        "jurisdiction": "Genesee County"  # always include
    }
    
    # Detect document type
    if re.search(r'STATE OF MICHIGAN.*PROBATE COURT', text[:1000], re.I):
        metadata["doc_type"] = "form"
    elif re.search(r'(MCL|MCR|EPIC)\s*\d+', text):
        metadata["doc_type"] = "statute"
    elif re.search(r'(procedure|step.by.step|how.to|checklist)', text[:500], re.I):
        metadata["doc_type"] = "procedure"
    
    # Extract form numbers
    form_matches = re.findall(r'PC\s*\d{3}[A-Z]?', text)
    if form_matches:
        # Join as comma-separated string for ChromaDB compatibility
        metadata["form_numbers"] = ', '.join(list(set(form_matches)))
    
    # Extract statutory citations
    statute_matches = re.findall(r'(MCL|MCR|EPIC)\s*\d+\.\d+[a-z]?(?:\(\d+\))?', text)
    if statute_matches:
        # Join as comma-separated string
        metadata["statutes"] = ', '.join(list(set(statute_matches)))
    
    # Extract deadlines and timing
    deadline_matches = re.findall(r'\d+\s*(?:days?|weeks?|months?|years?)', text)
    if deadline_matches:
        # Join as comma-separated string
        metadata["deadlines"] = ', '.join(list(set(deadline_matches)))
    
    # Extract fees
    fee_matches = re.findall(r'\$\d+(?:\.\d{2})?', text)
    if fee_matches:
        # Join as comma-separated string
        metadata["fees"] = ', '.join(list(set(fee_matches)))
    
    return metadata

def chunk_text(text: str, config: Dict) -> List[Dict]:
    """Chunk text using semantic boundaries"""
    chunks = []
    
    # Split by legal section markers
    section_pattern = r'(?:^|\n)(?:SECTION|Section|§|Article|ARTICLE)\s+[\dIVX]+[:\.]'
    sections = re.split(section_pattern, text)
    
    chunk_id = 0
    for section in sections:
        if not section.strip():
            continue
            
        # Further split long sections by paragraphs
        max_chunk_size = config.get('chunk_config', {}).get('size', 1000)
        if len(section) > max_chunk_size:
            paragraphs = section.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size:
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'chunk_id': chunk_id
                        })
                        chunk_id += 1
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para
            
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id
                })
                chunk_id += 1
        else:
            chunks.append({
                'text': section.strip(),
                'chunk_id': chunk_id
            })
            chunk_id += 1
    
    # Apply minimum chunk size
    min_chunk_size = config.get('chunk_config', {}).get('min_size', 100)
    filtered_chunks = []
    for chunk in chunks:
        if len(chunk['text']) >= min_chunk_size:
            filtered_chunks.append(chunk)
    
    return filtered_chunks

def load_documents() -> List[Dict]:
    """Load all documents from kb_files directory"""
    documents = []
    
    # Load text and markdown files
    for pattern in ["*.txt", "*.md"]:
        for file_path in KB_DIR.glob(f"**/{pattern}"):
            try:
                content = file_path.read_text(encoding='utf-8')
                metadata = extract_metadata_from_text(content)
                metadata['source'] = file_path.name
                
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load PDFs
    court_forms_dir = KB_DIR / "Court Forms"
    genesee_forms_dir = court_forms_dir / "Genesee County Specific"
    
    for pdf_name in PDFS_TO_EMBED:
        pdf_path = None
        
        # Check both directories
        if (court_forms_dir / pdf_name).exists():
            pdf_path = court_forms_dir / pdf_name
        elif (genesee_forms_dir / pdf_name).exists():
            pdf_path = genesee_forms_dir / pdf_name
        
        if pdf_path:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    
                    metadata = extract_metadata_from_text(text)
                    metadata['source'] = pdf_name
                    metadata['doc_type'] = 'form'
                    
                    documents.append({
                        'content': text,
                        'metadata': metadata
                    })
                    print(f"Loaded PDF: {pdf_name}")
            except Exception as e:
                print(f"Error loading PDF {pdf_name}: {e}")
        elif pdf_name in EXPECTED_PDFS:
            print(f"Warning: Expected PDF not found: {pdf_name}")
    
    # Load system instruction files
    instruction_files = [
        DOCS_DIR / "Dynamic Mode Examples copy.txt",
        DOCS_DIR / "Genesee County Specifics.txt",
        DOCS_DIR / "Master Prompt Template copy.txt",
        DOCS_DIR / "Michigan Minor Guardianship Knowledge Base Index copy.txt",
        DOCS_DIR / "Out-of-Scope Guidelines copy.txt"
    ]
    
    for file_path in instruction_files:
        if file_path.exists():
            try:
                content = file_path.read_text(encoding='utf-8')
                metadata = {
                    'source': file_path.name,
                    'doc_type': 'system_instruction',
                    'jurisdiction': 'Genesee County'
                }
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
                print(f"Loaded instruction: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load constants
    for const_file in CONSTANTS_DIR.glob("*.json"):
        try:
            content = const_file.read_text(encoding='utf-8')
            metadata = {
                'source': const_file.name,
                'doc_type': 'constants',
                'jurisdiction': 'Genesee County'
            }
            documents.append({
                'content': content,
                'metadata': metadata
            })
            print(f"Loaded constants: {const_file.name}")
        except Exception as e:
            print(f"Error loading {const_file}: {e}")
    
    return documents

def embed_with_google_ai(texts: List[str], api_key: str) -> List[List[float]]:
    """Generate embeddings using Google AI Studio's text-embedding-004"""
    # Configure the API
    genai.configure(api_key=api_key)
    
    embeddings = []
    batch_size = 100  # Google AI allows up to 100 texts per batch
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Generate embeddings for the batch
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="retrieval_document"  # Optimize for retrieval
            )
            
            # Extract embeddings
            for embedding in result['embedding']:
                embeddings.append(embedding)
            
            # Rate limiting - be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {e}")
            # Add zero embeddings for failed batch
            for _ in batch:
                embeddings.append([0.0] * 768)  # text-embedding-004 uses 768 dimensions
    
    return embeddings

def main():
    """Main embedding pipeline using Google AI Studio"""
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        print("Please set it with: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Load configurations
    chunking_config, embedding_config = load_configs()
    
    # Load documents
    documents = load_documents()
    if not documents:
        print("No documents found to embed!")
        return
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create or get collection with correct dimensions
    try:
        # Delete existing collection if it exists (to ensure clean state)
        try:
            client.delete_collection("michigan_guardianship_v2")
            print("Deleted existing collection")
        except:
            pass
            
        collection = client.create_collection(
            name="michigan_guardianship_v2",
            metadata={"hnsw:space": "cosine"}
        )
        print("Created new collection: michigan_guardianship_v2")
    except Exception as e:
        print(f"Error with collection: {e}")
        return
    
    # Process documents
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    print(f"Processing {len(documents)} documents...")
    
    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = chunk_text(doc['content'], chunking_config)
        
        for chunk in chunks:
            all_chunks.append(chunk['text'])
            
            # Combine document metadata with chunk metadata
            chunk_metadata = doc['metadata'].copy()
            chunk_metadata['chunk_id'] = chunk['chunk_id']
            all_metadata.append(chunk_metadata)
            
            # Create unique ID
            doc_id = f"{doc['metadata']['source']}_{chunk['chunk_id']}"
            all_ids.append(doc_id)
    
    print(f"Created {len(all_chunks)} chunks")
    print("Generating embeddings with Google AI Studio...")
    
    # Generate embeddings using Google AI
    all_embeddings = []
    batch_size = 50  # Process in smaller batches for progress tracking
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch_texts = all_chunks[i:i+batch_size]
        batch_embeddings = embed_with_google_ai(batch_texts, api_key)
        all_embeddings.extend(batch_embeddings)
    
    # Add to ChromaDB in batches
    print("Adding to ChromaDB...")
    storage_batch_size = 100
    for i in tqdm(range(0, len(all_chunks), storage_batch_size), desc="Storing in ChromaDB"):
        end_idx = min(i + storage_batch_size, len(all_chunks))
        collection.add(
            embeddings=all_embeddings[i:end_idx],
            documents=all_chunks[i:end_idx],
            metadatas=all_metadata[i:end_idx],
            ids=all_ids[i:end_idx]
        )
    
    print(f"Successfully embedded {len(all_chunks)} chunks into ChromaDB")
    
    # Test retrieval
    test_retrieval(collection, api_key)

def test_retrieval(collection, api_key):
    """Test retrieval with sample queries"""
    test_queries = [
        "filing fee genesee",
        "native american guardianship",
        "thursday hearing"
    ]
    
    print("\n=== Testing Retrieval ===")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Embed query using Google AI
        query_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"  # Optimize for query
        )
        query_embedding = query_result['embedding']
        
        # Search with Genesee filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"jurisdiction": "Genesee County"}
        )
        
        print(f"Top {len(results['documents'][0])} results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n{i+1}. Source: {metadata['source']} (distance: {distance:.3f})")
            print(f"   Type: {metadata['doc_type']}")
            if 'form_numbers' in metadata:
                print(f"   Forms: {metadata['form_numbers']}")
            print(f"   Text: {doc[:200]}...")
            
            # Check for expected content
            if query == "filing fee" and "$175" in doc:
                print("   ✓ Contains expected fee information")
            elif query == "native american" and ("ICWA" in doc or "Indian" in doc):
                print("   ✓ Contains ICWA-related content")
            elif query == "thursday hearing" and ("Thursday" in doc or "hearing" in doc):
                print("   ✓ Contains hearing schedule information")
    
    print("\n✓ Cloud-based document embedding pipeline completed successfully!")

if __name__ == "__main__":
    main()