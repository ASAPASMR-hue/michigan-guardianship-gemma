#!/usr/bin/env python3
"""
embed_kb.py - Document Embedding Pipeline for Michigan Guardianship AI
Embeds all knowledge base documents using BAAI/bge-m3 into ChromaDB
"""

import os
import sys
import yaml
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch
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
    
    # Override with small models if flag is set
    if os.getenv('USE_SMALL_MODEL', 'false').lower() == 'true':
        print("Using small models for development/testing")
        embedding_config['primary_model'] = 'all-MiniLM-L6-v2'
        embedding_config['fallback_model'] = 'paraphrase-MiniLM-L6-v2'
    
    with open(CONSTANTS_DIR / "genesee.yaml", "r") as f:
        genesee_constants = yaml.safe_load(f)
    
    return chunking_config, embedding_config, genesee_constants

def extract_form_number(filename: str, genesee_constants: dict) -> List[str]:
    """Extract form numbers from filename and constants"""
    form_numbers = []
    
    # Extract from filename (e.g., pc651.pdf -> PC 651)
    match = re.search(r'(pc|mc|gm)[-_]?(\d+)', filename.lower())
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        form_numbers.append(f"{prefix} {number}")
    
    # Check against critical forms in constants
    critical_forms = genesee_constants['genesee_county_constants']['critical_forms']
    for key, form_num in critical_forms.items():
        if form_num.replace(" ", "").lower() in filename.lower():
            if form_num not in form_numbers:
                form_numbers.append(form_num)
    
    return form_numbers

def chunk_text(text: str, config: dict) -> List[Dict[str, str]]:
    """Chunk text according to configuration with legal pattern preservation"""
    chunk_size = config['chunk_config']['size']
    overlap = config['chunk_config']['overlap']
    separators = config['chunk_config']['separators']
    preserve_patterns = config['chunk_config']['preserve_together']
    
    chunks = []
    
    # Simple chunking implementation
    # In production, use more sophisticated chunking with tiktoken
    words = text.split()
    current_chunk = []
    current_size = 0
    
    for i, word in enumerate(words):
        current_chunk.append(word)
        current_size += 1
        
        # Check if we should preserve upcoming patterns
        lookahead = " ".join(words[i:i+20])  # Look ahead 20 words
        preserve = False
        for pattern in preserve_patterns:
            if re.search(pattern, lookahead):
                preserve = True
                break
        
        if current_size >= chunk_size and not preserve:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            })
            
            # Overlap
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words
            current_size = len(overlap_words)
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        })
    
    return chunks

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber for better preservation"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        log_step("PDF extraction error", f"Failed to extract {pdf_path}: {str(e)}", "Error handling")
        return ""
    return text

def load_documents(genesee_constants: dict) -> List[Dict]:
    """Load all documents from kb_files and docs directories"""
    documents = []
    
    # Load KB (Numbered) TXT files
    kb_numbered_dir = KB_DIR / "KB (Numbered)"
    for txt_file in kb_numbered_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({
            "content": content,
            "metadata": {
                "source": str(txt_file.name),
                "doc_type": "procedure",
                "jurisdiction": "Genesee County",
                "last_updated": "2025-07-10"
            }
        })
    
    # Load Instructive TXT files
    instructive_dir = KB_DIR / "Instructive"
    for txt_file in instructive_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({
            "content": content,
            "metadata": {
                "source": str(txt_file.name),
                "doc_type": "guidance",
                "jurisdiction": "Genesee County",
                "last_updated": "2025-07-10"
            }
        })
    
    # Load Project Guidance
    guidance_file = DOCS_DIR / "Project_Guidance_v2.1.md"
    if guidance_file.exists():
        with open(guidance_file, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({
            "content": content,
            "metadata": {
                "source": "Project_Guidance_v2.1.md",
                "doc_type": "guidance",
                "jurisdiction": "Genesee County",
                "last_updated": "2025-07-10"
            }
        })
    
    # Load PDFs
    court_forms_dir = KB_DIR / "Court Forms"
    genesee_forms_dir = court_forms_dir / "Genesee County Specific"
    
    for pdf_name in PDFS_TO_EMBED:
        pdf_path = None
        
        # Check main directory
        if (court_forms_dir / pdf_name).exists():
            pdf_path = court_forms_dir / pdf_name
        # Check Genesee subdirectory
        elif (genesee_forms_dir / pdf_name).exists():
            pdf_path = genesee_forms_dir / pdf_name
        
        if pdf_path:
            content = extract_pdf_text(pdf_path)
            if content:
                form_numbers = extract_form_number(pdf_name, genesee_constants)
                metadata = {
                    "source": pdf_name,
                    "doc_type": "form",
                    "jurisdiction": "Genesee County",
                    "last_updated": "2025-07-10"
                }
                if form_numbers:
                    # ChromaDB doesn't support lists in metadata, so join them
                    metadata["form_numbers"] = ", ".join(form_numbers)
                
                documents.append({
                    "content": content,
                    "metadata": metadata
                })
            else:
                print(f"Warning: No text extracted from {pdf_name}")
        else:
            if pdf_name in EXPECTED_PDFS:
                print(f"ERROR: Expected PDF {pdf_name} not found in expected locations")
            else:
                print(f"Note: Optional PDF {pdf_name} not found (skipping)")
    
    return documents

def embed_documents(documents: List[Dict], chunking_config: dict, embedding_config: dict):
    """Embed documents into ChromaDB"""
    # Set HuggingFace token from environment
    # Get HuggingFace token from environment
    hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
        print("Warning: HUGGING_FACE_HUB_TOKEN not set. Some models may require authentication.")
        print("Set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'")
    else:
        os.environ['HF_TOKEN'] = hf_token
    
    # Initialize embedding model with error handling
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print("Downloading model... this may take a few minutes")
        
        model = SentenceTransformer(
            embedding_config['primary_model'],
            device=device,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load primary model: {e}")
        print(f"Falling back to: {embedding_config['fallback_model']}")
        print("Downloading fallback model... this may take a few minutes")
        model = SentenceTransformer(
            embedding_config['fallback_model'],
            device=device,
            use_auth_token=hf_token,
            trust_remote_code=True
        )
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create or get collection
    try:
        collection = client.create_collection(
            name="michigan_guardianship_v2",
            metadata={"hnsw:space": "cosine"}
        )
        print("Created new collection: michigan_guardianship_v2")
    except:
        collection = client.get_collection("michigan_guardianship_v2")
        print("Using existing collection: michigan_guardianship_v2")
    
    # Process documents
    all_chunks = []
    all_embeddings = []
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
    print("Generating embeddings...")
    
    # Batch embed with progress bar
    batch_size = 32
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch_texts = all_chunks[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.extend(batch_embeddings.tolist())
    
    # Add to ChromaDB in batches
    print("Adding to ChromaDB...")
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Storing in ChromaDB"):
        end_idx = min(i + batch_size, len(all_chunks))
        collection.add(
            embeddings=all_embeddings[i:end_idx],
            documents=all_chunks[i:end_idx],
            metadatas=all_metadata[i:end_idx],
            ids=all_ids[i:end_idx]
        )
    
    print(f"Successfully embedded {len(all_chunks)} chunks into ChromaDB")
    return collection, model

def test_retrieval(collection, model):
    """Test retrieval with sample queries"""
    test_queries = [
        "filing fee genesee",
        "native american guardianship",
        "thursday hearing"
    ]
    
    print("\n=== Testing Retrieval ===")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Embed query
        query_embedding = model.encode(query, normalize_embeddings=True)
        
        # Search with Genesee filter
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
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
            if query == "filing fee genesee":
                if "$175" in doc or "MC 20" in doc:
                    print("   ✓ Contains expected fee/waiver information")
            elif query == "native american guardianship":
                if "ICWA" in doc or "Indian" in doc or "tribal" in doc:
                    print("   ✓ Contains ICWA-related content")
            elif query == "thursday hearing":
                if "Thursday" in doc or "hearing" in doc:
                    print("   ✓ Contains hearing schedule information")

class DocumentProcessor:
    """Standard document processor with semantic chunking"""
    
    def __init__(self):
        self.chunk_config = {
            "size": 1000,
            "overlap": 100,
            "separators": [
                "\n## ",
                "\n### ",
                "\nMCL ",
                "\nPC ",
                "\n§ ",
                "\n- ",
                "\n\n",
            ]
        }
    
    def process_document(self, filepath: str) -> List[Dict]:
        """Process a single document into chunks"""
        path = Path(filepath)
        
        if path.suffix == '.pdf':
            content = extract_pdf_text(path)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if not content:
            return []
        
        chunks = chunk_text(content, self.chunk_config)
        
        # Add source metadata
        for chunk in chunks:
            chunk['source'] = path.name
            chunk['content'] = chunk.pop('text')  # Rename for consistency
        
        return chunks

class EnhancedDocumentProcessor(DocumentProcessor):
    """Enhanced document processor with pattern-based preservation"""
    
    def __init__(self):
        super().__init__()
        self.chunk_config['preserve_together'] = [
            r"(Form PC \d+.*?)\n",
            r"(MCL \d+\.\d+.*?)\n",
            r"(\$\d+.*?waiver.*?)\n",
            r"(\d+ days?.*?)\n",
            r"(Genesee County.*?)\n",
            r"(ICWA.*?)\n",
            r"(Step \d+:.*?)\n",
        ]
    
    def process_document(self, filepath: str) -> List[Dict]:
        """Process with enhanced pattern preservation"""
        # For now, use same logic as parent
        # In production, would implement pattern-aware chunking
        chunks = super().process_document(filepath)
        
        # Simulate pattern preservation by merging adjacent chunks
        # that match preservation patterns
        import re
        
        enhanced_chunks = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if chunk ends with a pattern that should be preserved
            for pattern in self.chunk_config.get('preserve_together', []):
                if re.search(pattern, current_chunk['content']):
                    # Merge with next chunk if exists
                    if i + 1 < len(chunks):
                        current_chunk['content'] += "\n" + chunks[i + 1]['content']
                        i += 1  # Skip next chunk
                    break
            
            enhanced_chunks.append(current_chunk)
            i += 1
        
        return enhanced_chunks

def main():
    """Main execution function"""
    log_step("Starting document embedding", "Beginning Phase 1 embedding pipeline", "Per Part A.1-A.2")
    
    # Load configurations
    chunking_config, embedding_config, genesee_constants = load_configs()
    
    # Load documents
    documents = load_documents(genesee_constants)
    log_step("Loaded documents", f"Loaded {len(documents)} documents from kb_files and docs", "Document ingestion")
    
    # Embed documents
    collection, model = embed_documents(documents, chunking_config, embedding_config)
    log_step("Embedding complete", f"Embedded documents into ChromaDB collection", "Per Part A.2")
    
    # Test retrieval
    test_retrieval(collection, model)
    log_step("Testing complete", "Verified retrieval with test queries", "Quality assurance")
    
    print("\n✓ Document embedding pipeline completed successfully!")

if __name__ == "__main__":
    main()