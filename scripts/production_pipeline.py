#!/usr/bin/env python3
"""
Production Pipeline for Michigan Guardianship AI
Provides a unified GuardianshipRAG class that integrates all components
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.adaptive_retrieval import AdaptiveHybridRetriever
from scripts.llm_handler import LLMHandler
from scripts.validator_setup import ResponseValidator
from scripts.logger import log_step

class GuardianshipRAG:
    """
    Main production pipeline for Michigan Guardianship AI.
    Integrates retrieval, generation, and validation components.
    """
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it"):
        """
        Initialize the GuardianshipRAG pipeline.
        
        Args:
            model_name: The LLM model to use for generation
        """
        self.model_name = model_name
        
        # Initialize components
        log_step("Initializing GuardianshipRAG pipeline...")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize retriever
        log_step("Loading retrieval system...")
        self.retriever = AdaptiveHybridRetriever()
        
        # Initialize LLM handler
        log_step("Loading language model...")
        self.llm_handler = LLMHandler(timeout=60)
        
        # Initialize validator (optional)
        self.validator = None
        try:
            self.validator = ResponseValidator()
            log_step("Validator loaded successfully")
        except Exception as e:
            log_step(f"Warning: Validator not available: {e}")
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Load Genesee constants
        self.genesee_constants = self._load_genesee_constants()
        
        log_step("GuardianshipRAG pipeline initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration files"""
        config_dir = Path(__file__).parent.parent / "config"
        config = {}
        
        # Load model configs
        model_config_path = config_dir / "model_configs_phase3.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                config['models'] = yaml.safe_load(f)
        
        # Load retrieval config
        retrieval_config_path = config_dir / "retrieval_pipeline.yaml"
        if retrieval_config_path.exists():
            with open(retrieval_config_path, 'r') as f:
                config['retrieval'] = yaml.safe_load(f)
        
        return config
    
    def _load_system_prompt(self) -> str:
        """Load the master system prompt"""
        prompt_paths = [
            Path(__file__).parent.parent / "kb_files" / "Instructive" / "Master Prompt Template.txt",
            Path(__file__).parent.parent / "docs" / "master_prompt.txt"
        ]
        
        for path in prompt_paths:
            if path.exists():
                return path.read_text()
        
        # Default prompt if file not found
        return """You are the Michigan Guardianship AI assistant, specifically serving Genesee County residents.
Your role is to help families navigate minor guardianship procedures with accurate, actionable guidance.

Key principles:
1. ZERO HALLUCINATION: Every legal statement must include an inline citation
2. LOCAL FOCUS: Always include Genesee County specifics (e.g., $175 filing fee, Thursday hearings)
3. ACTIONABILITY: Provide clear, step-by-step guidance
4. LEGAL COMPLIANCE: Include appropriate disclaimers per Michigan Rule 7.1

Remember: You are not a lawyer and cannot provide legal advice. Always recommend consulting 
with a qualified attorney for specific legal situations."""
    
    def _load_genesee_constants(self) -> Dict[str, Any]:
        """Load Genesee County constants"""
        constants_path = Path(__file__).parent.parent / "constants" / "genesee.yaml"
        
        if constants_path.exists():
            with open(constants_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default constants if file not found
        return {
            "genesee_county_constants": {
                "filing_fee": "$175",
                "hearing_day": "Thursday",
                "courthouse_address": "900 S. Saginaw St., Flint, MI 48502",
                "critical_forms": {
                    "petition": "PC 651",
                    "consent": "PC 654",
                    "annual_review": "PC 562"
                }
            }
        }
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Main method to get an answer for a user question.
        
        Args:
            question: The user's question about guardianship
            
        Returns:
            Dict containing the answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant documents
            log_step(f"Processing question: {question[:100]}...")
            retrieval_results, retrieval_metadata = self.retriever.retrieve_with_latency(question)
            
            # Step 2: Build context from retrieved documents
            context = self._build_context(retrieval_results)
            
            # Step 3: Inject Genesee County constants
            context = self._inject_constants(context)
            
            # Step 4: Generate response
            response = self._generate_response(question, context)
            
            # Step 5: Validate response (if validator available)
            validation_results = None
            if self.validator and response:
                # Extract retrieved chunks from retrieval results
                # Check for both 'content' and 'document' keys for compatibility
                retrieved_chunks = []
                for doc in retrieval_results:
                    if 'content' in doc:
                        retrieved_chunks.append(doc['content'])
                    elif 'document' in doc:
                        retrieved_chunks.append(doc['document'])
                    else:
                        # Fallback to the doc itself if it's a string
                        retrieved_chunks.append(str(doc))
                
                # Get complexity from retrieval metadata
                complexity = retrieval_metadata.get("complexity", "standard")
                
                validation_results = self.validator.validate(
                    response=response,
                    retrieved_chunks=retrieved_chunks,
                    question_type=complexity,  # Use the complexity level as question type
                    original_query=question
                )
            
            # Step 6: Add legal disclaimer
            response = self._add_disclaimer(response)
            
            # Calculate total time
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "answer": response,
                "metadata": {
                    "model": self.model_name,
                    "retrieval_metadata": retrieval_metadata,
                    "validation": validation_results,
                    "processing_time": total_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            log_step(f"Error processing question: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing or contact support if the issue persists.",
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _build_context(self, retrieval_results: list) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        
        for i, result in enumerate(retrieval_results[:5]):  # Top 5 results
            # Check for both 'content' and 'document' keys
            content = result.get('content', result.get('document', ''))
            metadata = result.get('metadata', {})
            
            # Format each result
            context_part = f"[Source {i+1}: {metadata.get('source', 'Unknown')}]\n{content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _inject_constants(self, context: str) -> str:
        """Inject Genesee County constants into context"""
        constants = self.genesee_constants.get('genesee_county_constants', {})
        
        constants_text = "\n[Genesee County Specific Information]\n"
        constants_text += f"- Filing Fee: {constants.get('filing_fee', '$175')}\n"
        constants_text += f"- Hearing Day: {constants.get('hearing_day', 'Thursday')}\n"
        constants_text += f"- Courthouse: {constants.get('courthouse_address', '900 S. Saginaw St., Flint, MI 48502')}\n"
        constants_text += f"- Key Forms: {', '.join(constants.get('critical_forms', {}).values())}\n"
        
        return context + "\n" + constants_text
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using LLM"""
        # Build the full prompt
        prompt = f"""{self.system_prompt}

Context from Michigan Guardianship Knowledge Base:
{context}

User Question: {question}

Please provide a comprehensive, actionable response with inline citations for all legal statements."""
        
        # Generate response
        try:
            # Convert prompt to messages format for LLMHandler
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for Michigan guardianship questions."},
                {"role": "user", "content": prompt}
            ]
            
            # Determine the API based on model name
            if self.model_name.startswith("google/"):
                model_api = "google_ai"
                # Make sure API key is available
                import os
                api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise Exception("Google AI API key not found in environment variables")
            else:
                model_api = "openrouter"
            
            result = self.llm_handler.call_llm(
                model_id=self.model_name,
                messages=messages,
                model_api=model_api,
                max_tokens=1500,
                temperature=0.7
            )
            
            if result["error"]:
                raise Exception(f"LLM Error: {result['error']}")
                
            return result["response"]
        except Exception as e:
            log_step(f"Error generating response: {e}")
            raise
    
    def _add_disclaimer(self, response: str) -> str:
        """Add legal disclaimer to response"""
        disclaimer = "\n\n---\n*This information is provided for educational purposes only and does not constitute legal advice. For specific legal guidance, please consult with a qualified attorney who can review your individual circumstances.*"
        
        return response + disclaimer
    
    def health_check(self) -> Dict[str, Any]:
        """Check health status of all components"""
        health = {
            "status": "healthy",
            "components": {
                "retriever": "unknown",
                "llm": "unknown",
                "validator": "unknown"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Check retriever
        try:
            if hasattr(self.retriever, 'collection'):
                health["components"]["retriever"] = "healthy"
            else:
                health["components"]["retriever"] = "not initialized"
        except:
            health["components"]["retriever"] = "error"
        
        # Check LLM
        try:
            if self.llm_handler:
                health["components"]["llm"] = "healthy"
        except:
            health["components"]["llm"] = "error"
        
        # Check validator
        if self.validator:
            health["components"]["validator"] = "healthy"
        else:
            health["components"]["validator"] = "not available"
        
        # Overall status
        if any(status == "error" for status in health["components"].values()):
            health["status"] = "degraded"
        
        return health