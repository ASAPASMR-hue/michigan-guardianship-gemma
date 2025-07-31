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
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.adaptive_retrieval import AdaptiveHybridRetriever
from scripts.llm_handler import LLMHandler
from scripts.validator_setup import ResponseValidator
from scripts.logger import log_step
from server.schemas import (
    AnswerPayload, Citation, Step, StructuredResponse,
    extract_forms_from_text, extract_fees_from_text, extract_risk_flags
)
from server.disclaimer_policy import DisclaimerPolicy
from server.conversation_state import ConversationState
from server.prompt_security import PromptSecurity

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
            "version_info": {
                "version": "1.0.0",
                "effective_date": "2025-01-31",
                "last_updated": "2025-01-31"
            },
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
    
    def get_answer(self, question: str, conversation_state: Optional[ConversationState] = None) -> Dict[str, Any]:
        """
        Main method to get an answer for a user question.
        
        Args:
            question: The user's question about guardianship
            conversation_state: Optional conversation state for context
            
        Returns:
            Dict containing the structured answer and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant documents
            log_step(f"Processing question: {question[:100]}...")
            retrieval_results, retrieval_metadata = self.retriever.retrieve_with_latency(question)
            
            # Step 2: Build secure context from retrieved documents
            # Use the new secure wrapping for retrieved content
            secure_context = PromptSecurity.wrap_retrieved_context(retrieval_results)
            
            # Step 3: Prepare conversation state context
            state_context = None
            if conversation_state and conversation_state.has_meaningful_context():
                state_context = conversation_state.to_context_string()
            
            # Step 4: Generate structured response with secure prompt
            structured_data = self._generate_structured_response(
                question=question,
                secure_context=secure_context,
                retrieval_results=retrieval_results,
                conversation_state=conversation_state,
                state_context=state_context
            )
            
            # Step 5: Validate response (if validator available)
            validation_results = None
            if self.validator and structured_data:
                # Extract retrieved chunks from retrieval results
                retrieved_chunks = []
                for doc in retrieval_results:
                    if 'content' in doc:
                        retrieved_chunks.append(doc['content'])
                    elif 'document' in doc:
                        retrieved_chunks.append(doc['document'])
                    else:
                        retrieved_chunks.append(str(doc))
                
                # Get complexity from retrieval metadata
                complexity = retrieval_metadata.get("complexity", "standard")
                
                validation_results = self.validator.validate(
                    response=structured_data.answer_markdown,
                    retrieved_chunks=retrieved_chunks,
                    question_type=complexity,
                    original_query=question
                )
            
            # Step 6: Apply disclaimer policy (only add if needed)
            structured_data.answer_markdown = DisclaimerPolicy.apply_disclaimer_policy(
                response_text=structured_data.answer_markdown,
                risk_flags=structured_data.risk_flags,
                user_query=question,
                is_out_of_scope='out_of_scope' in structured_data.risk_flags
            )
            
            # Calculate total time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Add debug information
            structured_data.debug = {
                "retrieval_hits": len(retrieval_results),
                "model": self.model_name,
                "processing_time": total_time,
                "complexity": retrieval_metadata.get("complexity", "standard"),
                "retrieval_latency": retrieval_metadata.get("latency_ms", 0) / 1000.0
            }
            
            # Create structured response
            response = StructuredResponse(
                data=structured_data,
                metadata={
                    "model": self.model_name,
                    "retrieval_metadata": retrieval_metadata,
                    "validation": validation_results,
                    "processing_time": total_time
                },
                timestamp=datetime.now()
            )
            
            # Return as dict for JSON serialization
            return response.model_dump()
            
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
    
    def _inject_conversation_state(self, context: str, conversation_state: ConversationState) -> str:
        """Inject conversation state into context"""
        state_context = "\n\n" + conversation_state.to_context_string() + "\n"
        return state_context + context
    
    def _generate_structured_response(
        self,
        question: str,
        secure_context: str,
        retrieval_results: List[Dict],
        conversation_state: Optional[ConversationState] = None,
        state_context: Optional[str] = None
    ) -> AnswerPayload:
        """Generate structured response using LLM with secure prompt construction"""
        # Build secure prompt using PromptSecurity
        secure_prompt = PromptSecurity.build_secure_prompt(
            system_prompt=self.system_prompt + "\n\nIMPORTANT: You must provide your response in a structured format with the following components:\n1. A comprehensive prose answer (answer_markdown)\n2. All forms mentioned (forms list)\n3. All fees mentioned (fees list)\n4. Step-by-step instructions if applicable (steps list)\n5. Proper citations for all legal statements\n\nPlease provide:\n1. A comprehensive, actionable response with inline citations [Source: form/statute]\n2. Extract ALL form numbers mentioned (e.g., PC 651, MC 20)\n3. Extract ALL fees mentioned (e.g., $175 filing fee)\n4. If the answer involves a process, break it down into clear numbered steps\n5. Each legal statement MUST have an inline citation",
            retrieved_context=secure_context,
            constants=self.genesee_constants,
            user_question=question,
            conversation_state=state_context,
            include_security_policy=True
        )
        
        # Generate response
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for Michigan guardianship questions. You provide structured, actionable responses."},
                {"role": "user", "content": secure_prompt}
            ]
            
            # Determine the API based on model name
            if self.model_name.startswith("google/"):
                model_api = "google_ai"
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
            
            response_text = result["response"]
            
            # Parse the response to extract structured data
            return self._parse_structured_response(response_text, question, retrieval_results)
            
        except Exception as e:
            log_step(f"Error generating structured response: {e}")
            raise
    
    def _parse_structured_response(self, response_text: str, question: str, retrieval_results: List[Dict]) -> AnswerPayload:
        """Parse LLM response into structured format"""
        # Extract forms from the response
        forms = extract_forms_from_text(response_text)
        
        # Extract fees
        fees = extract_fees_from_text(response_text)
        
        # Extract risk flags
        risk_flags = extract_risk_flags(response_text, question)
        
        # Extract steps (look for numbered lists)
        steps = self._extract_steps(response_text)
        
        # Extract citations
        citations = self._extract_citations(response_text, retrieval_results)
        
        # Create structured payload
        return AnswerPayload(
            answer_markdown=response_text,
            citations=citations,
            forms=forms,
            fees=fees,
            steps=steps,
            risk_flags=risk_flags
        )
    
    def _extract_steps(self, text: str) -> List[Step]:
        """Extract numbered steps from text"""
        steps = []
        
        # Look for numbered patterns like "1." or "1)" or "Step 1:"
        step_pattern = r'(?:^|\n)\s*(?:(\d+)[.):]|Step\s+(\d+):)\s*(.+?)(?=\n\s*(?:\d+[.):]|Step\s+\d+:|$))'
        matches = re.finditer(step_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            step_num = match.group(1) or match.group(2)
            step_text = match.group(3).strip()
            
            # Extract any citations from this step
            step_citations = self._extract_citations_from_text(step_text)
            
            steps.append(Step(
                text=step_text,
                citations=step_citations
            ))
        
        return steps
    
    def _extract_citations(self, text: str, retrieval_results: List[Dict]) -> List[Citation]:
        """Extract citations from text and match with sources"""
        citations = []
        seen_citations = set()
        
        # Pattern for inline citations like [Source: PC 651] or [MCL 700.5204]
        citation_pattern = r'\[(?:Source:|Citation:|Ref:)?\s*([^\]]+)\]'
        matches = re.finditer(citation_pattern, text)
        
        for match in matches:
            citation_text = match.group(1).strip()
            
            # Skip if already processed
            if citation_text in seen_citations:
                continue
            seen_citations.add(citation_text)
            
            # Try to match with retrieval results
            source_id = None
            title = citation_text
            
            # Look for this citation in retrieval results
            for i, result in enumerate(retrieval_results):
                metadata = result.get('metadata', {})
                source = metadata.get('source', '')
                
                if citation_text.lower() in source.lower():
                    source_id = f"{source}#chunk_{i}"
                    if 'title' in metadata:
                        title = metadata['title']
                    break
            
            citations.append(Citation(
                source_id=source_id or f"inline#{len(citations)}",
                title=title,
                url=None
            ))
        
        # Also extract form citations that might not be in brackets
        form_matches = re.finditer(r'\b(Form\s+)?(PC|MC|CC|DC)\s*(\d{2,4})\b', text, re.IGNORECASE)
        for match in form_matches:
            form_num = f"{match.group(2)} {match.group(3)}".upper()
            if form_num not in seen_citations:
                seen_citations.add(form_num)
                citations.append(Citation(
                    source_id=f"{form_num.lower().replace(' ', '')}.pdf",
                    title=f"Form {form_num}",
                    url=None
                ))
        
        return citations
    
    def _extract_citations_from_text(self, text: str) -> List[Citation]:
        """Extract citations from a piece of text (used for steps)"""
        citations = []
        
        # Simple pattern for inline citations
        citation_pattern = r'\[(?:Source:|Citation:|Ref:)?\s*([^\]]+)\]'
        matches = re.finditer(citation_pattern, text)
        
        for match in matches:
            citation_text = match.group(1).strip()
            citations.append(Citation(
                source_id=f"step_ref#{len(citations)}",
                title=citation_text,
                url=None
            ))
        
        return citations
    
    def _add_disclaimer(self, response: str, risk_flags: List[str] = None, query: str = "") -> str:
        """Add legal disclaimer using policy-based approach"""
        # For backward compatibility, check if we have the context
        if risk_flags is None:
            # Legacy behavior - always add disclaimer
            disclaimer = "\n\n---\n*This information is provided for educational purposes only and does not constitute legal advice. For specific legal guidance, please consult with a qualified attorney who can review your individual circumstances.*"
            return response + disclaimer
        
        # Use policy-based approach
        return DisclaimerPolicy.apply_disclaimer_policy(
            response_text=response,
            risk_flags=risk_flags,
            user_query=query,
            is_out_of_scope='out_of_scope' in risk_flags
        )
    
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