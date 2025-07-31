#!/usr/bin/env python3
"""
Michigan Guardianship AI - Interactive Setup Script
This script guides users through the initial setup process.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from getpass import getpass
import time

# ANSI color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_welcome():
    """Display welcome message"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}Michigan Guardianship AI - Setup Wizard{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    print("Welcome! This setup wizard will help you configure the")
    print("Michigan Guardianship AI system for Genesee County.\n")
    print("This application helps families navigate minor guardianship")
    print("procedures with AI-powered assistance.\n")
    print(f"{YELLOW}Note: This is a one-time setup process.{RESET}\n")

def check_python_version():
    """Ensure Python 3.8 or higher is installed"""
    if sys.version_info < (3, 8):
        print(f"{RED}Error: Python 3.8 or higher is required.{RESET}")
        print(f"You have Python {sys.version}")
        sys.exit(1)
    print(f"{GREEN}✓ Python {sys.version.split()[0]} detected{RESET}")

def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print(f"{GREEN}✓ .env file already exists{RESET}")
        response = input("\nDo you want to update your API keys? (y/n): ").lower()
        if response != 'y':
            return False
    else:
        if env_example_path.exists():
            shutil.copy(env_example_path, env_path)
            print(f"{GREEN}✓ Created .env file from template{RESET}")
        else:
            # Create a basic .env file
            with open(env_path, 'w') as f:
                f.write("# Michigan Guardianship AI Environment Variables\n\n")
            print(f"{GREEN}✓ Created new .env file{RESET}")
    
    return True

def update_env_file(key, value):
    """Update or add a key-value pair in the .env file"""
    env_path = Path(".env")
    lines = []
    key_found = False
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    
    # Update existing key or add new one
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break
    
    if not key_found:
        lines.append(f"{key}={value}\n")
    
    with open(env_path, 'w') as f:
        f.writelines(lines)

def get_api_keys():
    """Interactively prompt for API keys"""
    print(f"\n{BOLD}API Key Configuration{RESET}")
    print("-" * 40)
    
    # Google API Key
    print(f"\n{BOLD}1. Google Gemini API Key{RESET}")
    print("   This key is required for the main language model.")
    print(f"   Get your key from: {BLUE}https://makersuite.google.com/app/apikey{RESET}")
    
    while True:
        google_key = getpass("\nPaste your Google API key (hidden): ").strip()
        if google_key:
            if google_key.startswith("AI"):
                update_env_file("GOOGLE_API_KEY", google_key)
                update_env_file("GEMINI_API_KEY", google_key)
                print(f"{GREEN}✓ Google API key saved{RESET}")
                break
            else:
                print(f"{RED}Invalid key format. Google API keys typically start with 'AI'.{RESET}")
        else:
            print(f"{YELLOW}Skipping Google API key configuration.{RESET}")
            break
    
    # HuggingFace Token
    print(f"\n{BOLD}2. HuggingFace Token{RESET}")
    print("   This token is required for downloading the embedding model.")
    print(f"   Get your token from: {BLUE}https://huggingface.co/settings/tokens{RESET}")
    
    while True:
        hf_token = getpass("\nPaste your HuggingFace token (hidden): ").strip()
        if hf_token:
            if hf_token.startswith("hf_"):
                update_env_file("HUGGINGFACE_TOKEN", hf_token)
                update_env_file("HUGGING_FACE_HUB_TOKEN", hf_token)
                print(f"{GREEN}✓ HuggingFace token saved{RESET}")
                break
            else:
                print(f"{RED}Invalid token format. HuggingFace tokens start with 'hf_'.{RESET}")
        else:
            print(f"{YELLOW}Skipping HuggingFace token configuration.{RESET}")
            break

def install_dependencies():
    """Install Python dependencies"""
    print(f"\n{BOLD}Installing Dependencies{RESET}")
    print("-" * 40)
    print("This may take a few minutes...\n")
    
    try:
        # First, upgrade pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            text=True,
            capture_output=True
        )
        print(f"{GREEN}✓ All dependencies installed successfully{RESET}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}✗ Failed to install dependencies{RESET}")
        print(f"Error: {e.stderr}")
        return False

def choose_embedding_method():
    """Let user choose between cloud and local embeddings"""
    print(f"\n{BOLD}Embedding Method Selection{RESET}")
    print("-" * 40)
    print("\nHow would you like to create document embeddings?")
    print("\n1. Cloud-based (Google AI) - Recommended")
    print("   - Fast and reliable")
    print("   - No memory requirements")
    print("   - Requires internet connection")
    print("\n2. Local embeddings")
    print("   - Fully offline after setup")
    print("   - Requires 4-8GB RAM")
    print("   - May have issues on macOS")
    
    while True:
        choice = input("\nSelect option (1 or 2): ").strip()
        if choice == '1':
            return 'cloud'
        elif choice == '2':
            return 'local'
        else:
            print(f"{RED}Please enter 1 or 2{RESET}")

def setup_embeddings(method):
    """Setup embeddings based on chosen method"""
    if method == 'cloud':
        print(f"\n{GREEN}✓ Cloud embeddings selected{RESET}")
        print("\nYou'll use Google AI's embedding API.")
        print("Run this command after setup to create the database:")
        print(f"{BLUE}python scripts/embed_kb_cloud.py{RESET}")
        return True
    else:
        return download_embedding_model()

def download_embedding_model():
    """Download the embedding model for local use"""
    print(f"\n{BOLD}Downloading Embedding Model{RESET}")
    print("-" * 40)
    
    # Check if using small model
    use_small = os.getenv('USE_SMALL_MODEL', 'false').lower() == 'true'
    
    if use_small:
        model_name = 'all-MiniLM-L6-v2'
        print(f"Downloading {model_name} (small model for testing)")
    else:
        model_name = 'BAAI/bge-m3'
        print(f"Downloading {model_name} (production model)")
    
    print(f"{YELLOW}This may take 5-10 minutes depending on your connection...{RESET}\n")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a Python script to download the model
    download_script = f"""
import os
os.environ['HF_HOME'] = './models'
from sentence_transformers import SentenceTransformer
print("Downloading {model_name} model...")
model = SentenceTransformer('{model_name}', cache_folder='./models')
print("Model downloaded successfully!")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", download_script],
            check=True,
            text=True,
            capture_output=True
        )
        print(f"{GREEN}✓ Embedding model downloaded successfully{RESET}")
        print("\nRun this command after setup to create the database:")
        print(f"{BLUE}python scripts/embed_kb.py{RESET}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}✗ Failed to download embedding model{RESET}")
        print(f"Error: {e.stderr}")
        print(f"\n{YELLOW}You can set USE_SMALL_MODEL=true in .env to use a smaller model.{RESET}")
        return False

def setup_chroma_db():
    """Initialize ChromaDB directory"""
    print(f"\n{BOLD}Setting up ChromaDB{RESET}")
    print("-" * 40)
    
    chroma_dir = Path("chroma_db")
    if chroma_dir.exists():
        print(f"{GREEN}✓ ChromaDB directory already exists{RESET}")
    else:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        print(f"{GREEN}✓ Created ChromaDB directory{RESET}")
    
    return True

def check_knowledge_base():
    """Verify knowledge base files exist"""
    print(f"\n{BOLD}Checking Knowledge Base Files{RESET}")
    print("-" * 40)
    
    kb_dir = Path("kb_files")
    if not kb_dir.exists():
        print(f"{RED}✗ Knowledge base directory not found{RESET}")
        return False
    
    # Check for key directories
    required_dirs = ["KB (Numbered)", "Court Forms", "Instructive"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (kb_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"{YELLOW}⚠ Missing directories: {', '.join(missing_dirs)}{RESET}")
        return False
    else:
        print(f"{GREEN}✓ All knowledge base directories found{RESET}")
        return True

def print_next_steps():
    """Display next steps for the user"""
    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN}Setup Complete!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}\n")
    
    print(f"{BOLD}Next Steps:{RESET}")
    print(f"1. Start the application: {BLUE}./start.sh{RESET} (or {BLUE}python app.py{RESET})")
    print(f"2. Open your browser to: {BLUE}http://127.0.0.1:5000{RESET}")
    print(f"3. Start asking questions about Michigan guardianship!\n")
    
    print(f"{YELLOW}Note: The first run may take a moment to initialize the models.{RESET}")
    print(f"\nFor help or issues, see the README.md file.\n")

def check_port_availability():
    """Check if default port is available and suggest alternatives"""
    import socket
    
    print(f"\n{BOLD}Checking Port Availability{RESET}")
    print("-" * 40)
    
    default_port = 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', default_port))
    sock.close()
    
    if result == 0:
        print(f"{YELLOW}⚠ Port {default_port} is already in use (common on macOS with AirPlay){RESET}")
        print(f"Configuring to use port 5001 instead...")
        update_env_file("PORT", "5001")
        print(f"{GREEN}✓ Configured to use port 5001{RESET}")
    else:
        print(f"{GREEN}✓ Default port {default_port} is available{RESET}")

def main():
    """Main setup flow"""
    print_welcome()
    
    # Check Python version
    check_python_version()
    
    # Create/update .env file
    should_configure_keys = create_env_file()
    
    # Get API keys if needed
    if should_configure_keys:
        get_api_keys()
    
    # Install dependencies
    print(f"\n{YELLOW}Installing dependencies...{RESET}")
    if not install_dependencies():
        print(f"\n{RED}Setup failed. Please check the error messages above.{RESET}")
        sys.exit(1)
    
    # Check port availability
    check_port_availability()
    
    # Choose embedding method
    embedding_method = choose_embedding_method()
    
    # Setup embeddings based on choice
    if not setup_embeddings(embedding_method):
        print(f"\n{YELLOW}Warning: Embedding setup incomplete.{RESET}")
        print("You'll need to set up embeddings manually.")
    
    # Setup ChromaDB
    setup_chroma_db()
    
    # Check knowledge base
    check_knowledge_base()
    
    # Make start.sh executable if it exists
    start_script = Path("start.sh")
    if start_script.exists():
        os.chmod(start_script, 0o755)
        print(f"{GREEN}✓ Made start.sh executable{RESET}")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Setup interrupted by user.{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}An error occurred: {e}{RESET}")
        sys.exit(1)