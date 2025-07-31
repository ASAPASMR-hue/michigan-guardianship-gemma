#!/usr/bin/env python3
"""
incremental_embeddings.py - Incremental embedding updates
Phase 2: Step 6 - Set up incremental embedding updates
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.log_step import log_step
from scripts.embed_kb import load_documents, chunk_text, extract_pdf_text

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "phase2_embeddings.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class IncrementalEmbeddingManager:
    """Manages incremental updates to document embeddings"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.chroma_dir = self.project_root / "chroma_db"
        self.metadata_file = self.project_root / "data" / "embedding_metadata.json"
        self.config_dir = self.project_root / "config"
        
        # Load configurations
        self.load_configs()
        
        # Initialize models and DB
        self.init_models()
        self.init_chromadb()
        
        # Load existing metadata
        self.load_metadata()
    
    def load_configs(self):
        """Load configuration files"""
        import yaml
        
        with open(self.config_dir / "chunking.yaml", 'r') as f:
            self.chunking_config = yaml.safe_load(f)
        
        with open(self.config_dir / "embedding.yaml", 'r') as f:
            self.embedding_config = yaml.safe_load(f)
    
    def init_models(self):
        """Initialize embedding model"""
        # Use small model for testing
        if os.getenv('USE_SMALL_MODEL', 'false').lower() == 'true':
            model_name = 'all-MiniLM-L6-v2'
        else:
            model_name = self.embedding_config['primary_model']
        
        logger.info(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)
    
    def init_chromadb(self):
        """Initialize ChromaDB connection"""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.chroma_client.get_collection("michigan_guardianship_v2")
            logger.info("Connected to existing collection")
        except:
            logger.error("Collection not found. Run embed_kb.py first.")
            raise
    
    def load_metadata(self):
        """Load existing embedding metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'documents': {},
                'last_updated': None,
                'embedding_model': str(self.embed_model),
                'chunk_strategy': 'semantic',
                'version': '1.0'
            }
    
    def save_metadata(self):
        """Save embedding metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata_file.parent.mkdir(exist_ok=True)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {self.metadata_file}")
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def compute_chunk_id(self, doc_path: str, chunk_index: int) -> str:
        """Generate consistent chunk ID"""
        return f"{Path(doc_path).stem}_{chunk_index:04d}"
    
    def detect_changed_documents(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Detect which documents have changed
        Returns: (new_docs, modified_docs, deleted_docs)
        """
        # Load genesee constants
        import yaml
        genesee_constants_path = self.project_root / "constants" / "genesee.yaml"
        with open(genesee_constants_path, 'r') as f:
            genesee_constants = yaml.safe_load(f)
        
        # Load current documents
        current_docs = load_documents(genesee_constants)
        current_paths = {Path(doc['metadata']['source']) for doc in current_docs}
        
        # Get stored document paths
        stored_paths = {Path(path) for path in self.metadata['documents'].keys()}
        
        # Categorize changes
        new_docs = []
        modified_docs = []
        deleted_docs = list(stored_paths - current_paths)
        
        for doc in current_docs:
            doc_path = Path(doc['metadata']['source'])
            doc_hash = self.compute_content_hash(doc['content'])
            
            if str(doc_path) not in self.metadata['documents']:
                new_docs.append(doc_path)
            elif self.metadata['documents'][str(doc_path)]['content_hash'] != doc_hash:
                modified_docs.append(doc_path)
        
        return new_docs, modified_docs, deleted_docs
    
    def process_document_changes(self, doc_path: Path, content: str, is_new: bool):
        """Process a new or modified document"""
        logger.info(f"Processing {'new' if is_new else 'modified'} document: {doc_path}")
        
        # Chunk the document
        chunks = chunk_text(content, self.chunking_config)
        
        # Track chunk IDs
        chunk_ids = []
        chunk_embeddings = []
        chunk_metadatas = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = self.compute_chunk_id(str(doc_path), i)
            chunk_ids.append(chunk_id)
            
            # Prepare metadata
            metadata = {
                'source': str(doc_path),
                'chunk_index': i,
                'doc_type': doc_path.suffix.strip('.'),
                'jurisdiction': 'Genesee County',
                'last_updated': datetime.now().isoformat()
            }
            chunk_metadatas.append(metadata)
        
        # Generate embeddings in batch
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_model.encode(chunk_texts, normalize_embeddings=True)
        
        # If modifying, delete old chunks first
        if not is_new and str(doc_path) in self.metadata['documents']:
            old_chunk_ids = self.metadata['documents'][str(doc_path)]['chunk_ids']
            if old_chunk_ids:
                try:
                    self.collection.delete(ids=old_chunk_ids)
                    logger.info(f"Deleted {len(old_chunk_ids)} old chunks")
                except Exception as e:
                    logger.warning(f"Error deleting old chunks: {e}")
        
        # Upsert new chunks
        if chunk_ids:
            self.collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            logger.info(f"Upserted {len(chunk_ids)} chunks")
        
        # Update metadata
        self.metadata['documents'][str(doc_path)] = {
            'content_hash': self.compute_content_hash(content),
            'chunk_ids': chunk_ids,
            'num_chunks': len(chunks),
            'last_modified': datetime.now().isoformat()
        }
    
    def delete_document_chunks(self, doc_path: Path):
        """Delete chunks for a removed document"""
        logger.info(f"Deleting chunks for removed document: {doc_path}")
        
        if str(doc_path) in self.metadata['documents']:
            chunk_ids = self.metadata['documents'][str(doc_path)]['chunk_ids']
            if chunk_ids:
                try:
                    self.collection.delete(ids=chunk_ids)
                    logger.info(f"Deleted {len(chunk_ids)} chunks")
                except Exception as e:
                    logger.warning(f"Error deleting chunks: {e}")
            
            # Remove from metadata
            del self.metadata['documents'][str(doc_path)]
    
    def simulate_document_change(self):
        """Simulate a document change for testing"""
        test_file = self.project_root / "kb_files" / "test_incremental.txt"
        
        # Create or modify test file
        if test_file.exists():
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Modify content
            new_content = content + f"\n\nUpdated at {datetime.now()}"
            logger.info(f"Modifying test file: {test_file}")
        else:
            new_content = f"""Test Document for Incremental Updates

This is a test document created at {datetime.now()} to verify incremental embedding updates work correctly.

Genesee County Probate Court Information:
- Filing fee: $175
- Court address: 900 S. Saginaw St., Room 502, Flint, MI 48502
- Hearings: Thursdays only

This document will be modified to test the update mechanism.
"""
            logger.info(f"Creating test file: {test_file}")
        
        with open(test_file, 'w') as f:
            f.write(new_content)
        
        return test_file
    
    def run_incremental_update(self):
        """Run incremental embedding update"""
        log_step("Starting incremental embedding update",
                "Detecting and processing document changes",
                "Phase 2 Step 6: Set up incremental embedding updates")
        
        start_time = datetime.now()
        
        # Detect changes
        logger.info("Detecting document changes...")
        new_docs, modified_docs, deleted_docs = self.detect_changed_documents()
        
        logger.info(f"Changes detected:")
        logger.info(f"  New documents: {len(new_docs)}")
        logger.info(f"  Modified documents: {len(modified_docs)}")
        logger.info(f"  Deleted documents: {len(deleted_docs)}")
        
        total_changes = len(new_docs) + len(modified_docs) + len(deleted_docs)
        
        if total_changes == 0:
            logger.info("No changes detected")
            return
        
        # Process new documents
        for doc_path in new_docs:
            # Handle full paths
            if not doc_path.exists():
                # Try relative to project root
                doc_path = self.project_root / doc_path
            
            if not doc_path.exists():
                logger.warning(f"Document not found: {doc_path}")
                continue
                
            if doc_path.suffix == '.pdf':
                content = extract_pdf_text(doc_path)
            else:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if content:
                self.process_document_changes(doc_path, content, is_new=True)
        
        # Process modified documents
        for doc_path in modified_docs:
            # Handle full paths
            if not doc_path.exists():
                # Try relative to project root
                doc_path = self.project_root / doc_path
            
            if not doc_path.exists():
                logger.warning(f"Document not found: {doc_path}")
                continue
                
            if doc_path.suffix == '.pdf':
                content = extract_pdf_text(doc_path)
            else:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if content:
                self.process_document_changes(doc_path, content, is_new=False)
        
        # Process deleted documents
        for doc_path in deleted_docs:
            self.delete_document_chunks(doc_path)
        
        # Save updated metadata
        self.save_metadata()
        
        # Log summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nIncremental update complete in {elapsed:.1f} seconds")
        logger.info(f"  Processed {len(new_docs)} new documents")
        logger.info(f"  Updated {len(modified_docs)} modified documents")
        logger.info(f"  Removed {len(deleted_docs)} deleted documents")
        
        # Verify collection state
        collection_count = self.collection.count()
        logger.info(f"  Total chunks in collection: {collection_count}")
        
        log_step("Incremental embedding update complete",
                f"Processed {total_changes} document changes in {elapsed:.1f}s",
                "Phase 2 Step 6 complete")

def main():
    """Main function"""
    manager = IncrementalEmbeddingManager()
    
    # Simulate a document change for testing
    test_file = manager.simulate_document_change()
    
    # For testing, only process the test file
    logger.info("Testing incremental update with single file...")
    
    # Check if test file is detected as new or modified
    if str(test_file) not in manager.metadata['documents']:
        logger.info(f"Processing new test document: {test_file}")
        with open(test_file, 'r') as f:
            content = f.read()
        manager.process_document_changes(test_file, content, is_new=True)
    else:
        logger.info(f"Test document already exists, modifying...")
        # Modify and reprocess
        with open(test_file, 'a') as f:
            f.write(f"\nModified again at {datetime.now()}")
        with open(test_file, 'r') as f:
            content = f.read()
        manager.process_document_changes(test_file, content, is_new=False)
    
    # Save metadata
    manager.save_metadata()
    
    # Verify
    collection_count = manager.collection.count()
    logger.info(f"Total chunks in collection: {collection_count}")
    
    log_step("Incremental embedding update test complete",
            f"Successfully processed test document",
            "Phase 2 Step 6 complete")
    
    # Clean up test file
    if test_file.exists():
        test_file.unlink()
        logger.info(f"Cleaned up test file: {test_file}")

if __name__ == "__main__":
    main()