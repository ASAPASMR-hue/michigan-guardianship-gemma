#!/usr/bin/env python3
"""
Download missing court forms from Michigan Courts website
"""

import os
import requests
from pathlib import Path
import time

# Form URLs from Michigan Courts website
FORM_URLS = {
    "pc650.pdf": "https://www.courts.michigan.gov/4a129a/siteassets/forms/scao-approved/pc650.pdf",
    "pc652.pdf": "https://www.courts.michigan.gov/4a12a6/siteassets/forms/scao-approved/pc652.pdf",
    "pc670.pdf": "https://www.courts.michigan.gov/4a134e/siteassets/forms/scao-approved/pc670.pdf",
    "pc675.pdf": "https://www.courts.michigan.gov/4a136b/siteassets/forms/scao-approved/pc675.pdf",
    "pc564.pdf": "https://www.courts.michigan.gov/4a1300/siteassets/forms/scao-approved/pc564.pdf"
}

def download_form(form_name, url, save_dir):
    """Download a single form PDF"""
    save_path = save_dir / form_name
    
    if save_path.exists():
        print(f"✓ {form_name} already exists")
        return True
    
    print(f"Downloading {form_name}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Successfully downloaded {form_name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {form_name}: {e}")
        return False

def main():
    """Download all missing forms"""
    # Create directory if needed
    court_forms_dir = Path(__file__).parent.parent / "kb_files" / "Court Forms"
    court_forms_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading missing court forms...")
    
    success_count = 0
    for form_name, url in FORM_URLS.items():
        if download_form(form_name, url, court_forms_dir):
            success_count += 1
        time.sleep(1)  # Be polite to the server
    
    print(f"\nDownloaded {success_count}/{len(FORM_URLS)} forms successfully")
    
    # List all PDFs in the directory
    print("\nCurrent PDFs in Court Forms directory:")
    for pdf in sorted(court_forms_dir.glob("*.pdf")):
        print(f"  - {pdf.name}")

if __name__ == "__main__":
    main()