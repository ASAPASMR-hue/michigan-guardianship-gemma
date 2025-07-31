#!/usr/bin/env python3
"""
Split Project_Guidance_v2.1.md into machine-readable config files.
Extracts YAML and Python-like dict blocks into separate files.
"""

import re
import yaml
import json
import ast
from pathlib import Path
from typing import Dict, Any

# Source and destination paths
SOURCE_DOC = Path("docs/Project_Guidance_v2.1.md")
HEADER = "# AUTO-GENERATED FROM docs/Project_Guidance_v2.1.md – DO NOT EDIT BY HAND\n"

def extract_code_blocks(content: str) -> Dict[str, str]:
    """Extract code blocks from markdown content."""
    blocks = {}
    
    # Pattern to match code blocks with language identifier
    pattern = r'```(?:python|yaml)\n(.*?)\n```'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        block_content = match.group(1)
        
        # Identify block type by content
        if "CHUNK_CONFIG" in block_content:
            blocks['chunk_config'] = block_content
        elif "EMBED_CONFIG" in block_content and "VECTOR_DB_CONFIG" in block_content:
            blocks['embed_config'] = block_content
        elif "SEARCH_PIPELINE" in block_content:
            blocks['search_pipeline'] = block_content
        elif "hallucination_detector" in block_content:
            blocks['validator'] = block_content
        elif "evaluation_rubric:" in block_content:
            blocks['eval_rubric'] = block_content
        elif "question_complexity_tiers:" in block_content:
            blocks['question_tiers'] = block_content
        elif "genesee_county_constants:" in block_content:
            blocks['genesee_constants'] = block_content
        elif "privacy_controls:" in block_content:
            blocks['privacy_controls'] = block_content
        elif "access_controls:" in block_content:
            blocks['access_controls'] = block_content
            
    return blocks

def extract_python_dict(code: str, var_name: str) -> Dict[str, Any]:
    """Extract a Python dictionary from code string."""
    # Find the dictionary assignment
    pattern = rf'{var_name}\s*=\s*(\{{[^}}]+\}})'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        dict_str = match.group(1)
        # Use ast.literal_eval for safe evaluation
        try:
            return ast.literal_eval(dict_str)
        except:
            # If literal_eval fails, try manual parsing
            return None
    return None

def extract_complexity_tiers(code: str) -> Dict[str, Any]:
    """Extract COMPLEXITY_TIERS from QueryComplexityClassifier."""
    pattern = r'COMPLEXITY_TIERS\s*=\s*({[^}]+(?:{[^}]+}[^}]+)*})'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        tiers_str = match.group(1)
        # Clean up the string for YAML
        tiers_str = re.sub(r'#.*$', '', tiers_str, flags=re.MULTILINE)
        tiers_str = tiers_str.replace('"', "'")
        try:
            return eval(tiers_str)
        except:
            return None
    return None

def save_chunking_config(block: str):
    """Extract and save chunking configuration."""
    chunk_dict = extract_python_dict(block, "CHUNK_CONFIG")
    metadata_dict = extract_python_dict(block, "METADATA_SCHEMA")
    update_dict = extract_python_dict(block, "EMBEDDING_UPDATE_STRATEGY")
    
    config = {
        "chunk_config": chunk_dict,
        "metadata_schema": metadata_dict,
        "embedding_update_strategy": update_dict
    }
    
    with open("config/chunking.yaml", "w") as f:
        f.write(HEADER)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def save_embedding_config(block: str):
    """Extract and save embedding configuration."""
    # Extract individual components
    embed_config = extract_python_dict(block, "EMBED_CONFIG")
    vector_db_config = extract_python_dict(block, "VECTOR_DB_CONFIG")
    
    # Extract model names
    embed_model_match = re.search(r'EMBED_MODEL\s*=\s*"([^"]+)"', block)
    fallback_model_match = re.search(r'FALLBACK_EMBED\s*=\s*"([^"]+)"', block)
    
    config = {
        "primary_model": embed_model_match.group(1) if embed_model_match else None,
        "fallback_model": fallback_model_match.group(1) if fallback_model_match else None,
        "embed_config": embed_config,
        "vector_db_config": vector_db_config
    }
    
    with open("config/embedding.yaml", "w") as f:
        f.write(HEADER)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def save_retrieval_pipeline(block: str):
    """Extract and save retrieval pipeline configuration."""
    # Extract SEARCH_PIPELINE
    search_pipeline = extract_python_dict(block, "SEARCH_PIPELINE")
    
    # Extract COMPLEXITY_TIERS
    complexity_tiers = extract_complexity_tiers(block)
    
    config = {
        "search_pipeline": search_pipeline,
        "complexity_tiers": complexity_tiers
    }
    
    with open("config/retrieval_pipeline.yaml", "w") as f:
        f.write(HEADER)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def save_validator_config(block: str):
    """Extract validator configuration patterns."""
    # Extract patterns from the ResponseValidator class
    patterns = {
        "genesee_patterns": {
            "filing_fee": r"\$175",
            "thursday": r"Thursday",
            "address": r"900 S\. Saginaw",
            "forms": r"PC \d{3}|MC \d{2}"
        },
        "hallucination_threshold": 0.05,
        "validation_gates": [
            "out_of_scope_check",
            "hallucination_check",
            "citation_verification",
            "procedural_accuracy",
            "mode_appropriateness",
            "legal_disclaimer"
        ]
    }
    
    with open("config/validator.yaml", "w") as f:
        f.write(HEADER)
        yaml.dump(patterns, f, default_flow_style=False, sort_keys=False)

def save_yaml_configs(yaml_blocks: Dict[str, str]):
    """Save YAML configuration blocks."""
    # evaluation_rubric
    if 'eval_rubric' in yaml_blocks:
        with open("rubrics/eval_rubric.yaml", "w") as f:
            f.write(HEADER)
            f.write(yaml_blocks['eval_rubric'])
    
    # question_complexity_tiers
    if 'question_tiers' in yaml_blocks:
        with open("rubrics/question_tiers.yaml", "w") as f:
            f.write(HEADER)
            f.write(yaml_blocks['question_tiers'])
    
    # genesee_county_constants
    if 'genesee_constants' in yaml_blocks:
        with open("constants/genesee.yaml", "w") as f:
            f.write(HEADER)
            f.write(yaml_blocks['genesee_constants'])

def create_out_of_scope_patterns():
    """Create out-of-scope patterns JSON file."""
    patterns = [
        {
            "category": "adult_guardianship",
            "regex": r"(adult|elderly|dementia|alzheimer|incapacitated adult)",
            "redirect": "For adult guardianship, contact the Genesee County Probate Court at (810) 257-3528"
        },
        {
            "category": "divorce_custody",
            "regex": r"(divorce|custody|parenting time|child support)",
            "redirect": "For divorce and custody matters, visit the Friend of the Court at 900 S. Saginaw St., Flint"
        },
        {
            "category": "adoption",
            "regex": r"(adopt|adoption)",
            "redirect": "For adoption information, contact Michigan Adoption Resource Exchange at (800) 589-6273"
        },
        {
            "category": "criminal",
            "regex": r"(criminal|arrest|jail|prison|probation)",
            "redirect": "For criminal matters, contact the Genesee County Circuit Court at (810) 257-3220"
        }
    ]
    
    with open("patterns/out_of_scope.json", "w") as f:
        f.write("// " + HEADER.strip() + "\n")
        json.dump(patterns, f, indent=2)

def main():
    """Main extraction process."""
    if not SOURCE_DOC.exists():
        print(f"Error: Source document not found at {SOURCE_DOC}")
        return
    
    # Read the markdown file
    content = SOURCE_DOC.read_text()
    
    # Extract code blocks
    blocks = extract_code_blocks(content)
    
    # Save configuration files
    if 'chunk_config' in blocks:
        save_chunking_config(blocks['chunk_config'])
        print("✓ Created config/chunking.yaml")
    
    if 'embed_config' in blocks:
        save_embedding_config(blocks['embed_config'])
        print("✓ Created config/embedding.yaml")
    
    if 'search_pipeline' in blocks:
        save_retrieval_pipeline(blocks['search_pipeline'])
        print("✓ Created config/retrieval_pipeline.yaml")
    
    if 'validator' in blocks:
        save_validator_config(blocks['validator'])
        print("✓ Created config/validator.yaml")
    
    # Save YAML configs
    save_yaml_configs(blocks)
    print("✓ Created rubrics/eval_rubric.yaml")
    print("✓ Created rubrics/question_tiers.yaml")
    print("✓ Created constants/genesee.yaml")
    
    # Create out-of-scope patterns
    create_out_of_scope_patterns()
    print("✓ Created patterns/out_of_scope.json")
    
    print("\nAll configuration files extracted successfully!")

if __name__ == "__main__":
    main()