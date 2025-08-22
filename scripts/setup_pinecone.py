#!/usr/bin/env python3
"""
Enhanced Pinecone setup for Michigan Guardianship RAG
Supports both text-embedding-004 (768d) and gemini-embedding-001 (3072d)
"""

import os
import sys
import yaml
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
from colorama import init, Fore, Style

# Initialize colorama for Windows compatibility
init()

# Load environment variables
load_dotenv()

def print_header():
    """Print setup header"""
    print(Fore.CYAN + "=" * 60)
    print("🏛️  Michigan Guardianship RAG - Enhanced Pinecone Setup")
    print("=" * 60 + Style.RESET_ALL)

def load_config():
    """Load configuration from pinecone.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'pinecone.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_embedding_config(config):
    """Validate embedding model and dimension compatibility"""
    embedding_config = config['embedding_config']
    index_config = config['index_config']
    
    model = embedding_config.get('model', '')
    dimension = index_config.get('dimension', 0)
    
    print(f"\n📊 Validating Configuration:")
    print(f"  Model: {Fore.YELLOW}{model}{Style.RESET_ALL}")
    print(f"  Dimension: {Fore.YELLOW}{dimension}{Style.RESET_ALL}")
    
    # Model-dimension compatibility check
    compatibility = {
        'models/text-embedding-004': 768,
        'models/gemini-embedding-001': 3072,
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072
    }
    
    if model in compatibility:
        expected_dim = compatibility[model]
        if dimension != expected_dim:
            print(f"\n{Fore.RED}⚠️  WARNING: Dimension mismatch!{Style.RESET_ALL}")
            print(f"  Model {model} expects {expected_dim} dimensions")
            print(f"  Configuration has {dimension} dimensions")
            
            response = input(f"\nUpdate dimension to {expected_dim}? (y/n): ")
            if response.lower() == 'y':
                index_config['dimension'] = expected_dim
                embedding_config['dimension'] = expected_dim
                print(f"{Fore.GREEN}✅ Dimension updated to {expected_dim}{Style.RESET_ALL}")
                return expected_dim
            else:
                print(f"{Fore.YELLOW}⚠️  Proceeding with potential mismatch{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Unknown model - ensure dimension is correct{Style.RESET_ALL}")
    
    return dimension

def choose_embedding_model(config):
    """Allow user to choose embedding model"""
    print(f"\n{Fore.CYAN}🤖 Select Embedding Model:{Style.RESET_ALL}")
    print("1. text-embedding-004 (768d) - Stable, lower dimension, good baseline")
    print("2. gemini-embedding-001 (3072d) - State-of-the-art, +10-15% recall")
    print("3. Keep current configuration")
    
    choice = input("\nChoice (1-3): ")
    
    if choice == '1':
        config['embedding_config']['model'] = 'models/text-embedding-004'
        config['embedding_config']['dimension'] = 768
        config['index_config']['dimension'] = 768
        print(f"{Fore.GREEN}✅ Selected text-embedding-004 (768d){Style.RESET_ALL}")
        return 768
    elif choice == '2':
        config['embedding_config']['model'] = 'models/gemini-embedding-001'
        config['embedding_config']['dimension'] = 3072
        config['index_config']['dimension'] = 3072
        print(f"{Fore.GREEN}✅ Selected gemini-embedding-001 (3072d){Style.RESET_ALL}")
        return 3072
    else:
        print(f"{Fore.YELLOW}Keeping current configuration{Style.RESET_ALL}")
        return config['index_config']['dimension']

def setup_pinecone():
    """Initialize Pinecone index with proper configuration"""
    print_header()
    
    # Load configuration
    config = load_config()
    
    # Allow model selection
    dimension = choose_embedding_model(config)
    
    # Validate configuration
    dimension = validate_embedding_config(config)
    
    index_config = config['index_config']
    index_name = index_config['name']
    metric = index_config['metric']
    namespace = index_config['namespace']
    
    # Get API key
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print(f"{Fore.RED}❌ Error: PINECONE_API_KEY not found in .env{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"\n{Fore.CYAN}🔧 Initializing Pinecone client...{Style.RESET_ALL}")
    pc = Pinecone(api_key=api_key)
    
    # Display configuration
    print(f"\n📊 Index Configuration:")
    print(f"  Name: {Fore.YELLOW}{index_name}{Style.RESET_ALL}")
    print(f"  Dimension: {Fore.YELLOW}{dimension}{Style.RESET_ALL}")
    print(f"  Metric: {Fore.YELLOW}{metric}{Style.RESET_ALL}")
    print(f"  Namespace: {Fore.YELLOW}{namespace}{Style.RESET_ALL}")
    print(f"  Cloud: AWS (us-east-1)")
    
    # Check for existing index
    print(f"\n🔍 Checking for existing index...")
    existing_indexes = pc.list_indexes()
    
    index_exists = any(idx.name == index_name for idx in existing_indexes.indexes)
    
    if index_exists:
        print(f"{Fore.YELLOW}⚠️  Index '{index_name}' already exists{Style.RESET_ALL}")
        index = pc.Index(index_name)
        
        # Get index info
        index_info = pc.describe_index(index_name)
        stats = index.describe_index_stats()
        
        print(f"\n📈 Existing Index Stats:")
        print(f"  Dimension: {index_info.dimension}")
        print(f"  Metric: {index_info.metric}")
        print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"  Namespaces: {list(stats.get('namespaces', {}).keys())}")
        
        # Check dimension compatibility
        if index_info.dimension != dimension:
            print(f"\n{Fore.RED}⚠️  CRITICAL: Index dimension ({index_info.dimension}) != Config dimension ({dimension}){Style.RESET_ALL}")
            print("You must delete and recreate the index to change dimensions")
            
            response = input("\nDelete and recreate index? (y/n): ")
            if response.lower() != 'y':
                print(f"{Fore.RED}❌ Cannot proceed with dimension mismatch{Style.RESET_ALL}")
                sys.exit(1)
            
            print(f"\n{Fore.YELLOW}🗑️  Deleting index...{Style.RESET_ALL}")
            pc.delete_index(index_name)
            time.sleep(5)
            index_exists = False
        else:
            response = input("\n🔄 Delete and recreate index? (y/n): ")
            if response.lower() == 'y':
                print(f"\n{Fore.YELLOW}🗑️  Deleting index...{Style.RESET_ALL}")
                pc.delete_index(index_name)
                time.sleep(5)
                index_exists = False
    
    if not index_exists:
        print(f"\n{Fore.GREEN}🚀 Creating new index...{Style.RESET_ALL}")
        print(f"  Name: {index_name}")
        print(f"  Dimension: {dimension}")
        print(f"  Metric: {metric}")
        
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            print("⏳ Waiting for index to be ready", end="")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
                print(".", end="", flush=True)
            
            print(f"\n{Fore.GREEN}✅ Index created successfully!{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error creating index: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    # Connect to index
    index = pc.Index(index_name)
    
    # Test with sample vector
    print(f"\n{Fore.CYAN}🧪 Testing index...{Style.RESET_ALL}")
    test_vector = [0.1] * dimension
    test_id = "test_setup_vector"
    
    try:
        index.upsert(
            vectors=[(test_id, test_vector, {"test": "true"})],
            namespace=namespace
        )
        print(f"{Fore.GREEN}✅ Test upload successful{Style.RESET_ALL}")
        
        # Clean up test vector
        index.delete(ids=[test_id], namespace=namespace)
        print(f"{Fore.GREEN}✅ Test cleanup successful{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️  Test failed: {e}{Style.RESET_ALL}")
    
    # Save configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'pinecone.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n{Fore.GREEN}✅ Configuration saved{Style.RESET_ALL}")
    
    # Print summary
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("📊 Setup Summary")
    print('=' * 60 + Style.RESET_ALL)
    print(f"✅ Index: {Fore.GREEN}{index_name}{Style.RESET_ALL}")
    print(f"✅ Dimension: {Fore.GREEN}{dimension}{Style.RESET_ALL}")
    print(f"✅ Model: {Fore.GREEN}{config['embedding_config']['model']}{Style.RESET_ALL}")
    print(f"✅ Status: {Fore.GREEN}Ready for migration{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}📝 Next Steps:{Style.RESET_ALL}")
    print("1. Review scripts/embed_kb_cloud.py for existing embedding logic")
    print("2. Run migration: python scripts/migrate_to_pinecone.py --dry-run")
    print("3. Monitor performance vs ChromaDB baseline")
    
    return index

if __name__ == "__main__":
    try:
        setup_pinecone()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Setup interrupted{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Setup failed: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
