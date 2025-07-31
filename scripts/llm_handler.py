"""
LLM Handler for Phase 3 Testing
Supports both Google AI Studio and OpenRouter models with timeout handling
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from openai import OpenAI, Timeout
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class LLMHandler:
    """Unified handler for multiple LLM providers"""
    
    def __init__(self, timeout: int = 180):
        """
        Initialize LLM handler
        
        Args:
            timeout: Maximum time in seconds to wait for response
        """
        self.timeout = timeout
        self.openrouter_client = None
        self.google_api_key = None
        
        # Rate limiting for Gemma models
        self.gemma_last_call_time = 0
        self.gemma_tokens_this_minute = 0
        self.gemma_minute_start = time.time()
        self.gemma_rate_limit = 15000  # tokens per minute
        
        # Initialize API clients
        self._init_openrouter()
        self._init_google_ai()
        
        # Cost tracking for Google models (per 1M tokens)
        self.google_pricing = {
            "google/gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
            "google/gemma-3n-e4b-it": {"input": 0.05, "output": 0.15},
            "google/gemma-3-4b-it": {"input": 0.05, "output": 0.15},
            "google/gemini-2.0-flash-lite-001": {"input": 0.075, "output": 0.30},
            "google/gemini-2.0-flash-001": {"input": 0.075, "output": 0.30},
            "google/gemini-flash-1.5-8b": {"input": 0.0375, "output": 0.15}
        }
    
    def _init_openrouter(self):
        """Initialize OpenRouter client"""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
    
    def _init_google_ai(self):
        """Initialize Google AI client"""
        api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if api_key:
            self.google_api_key = api_key
            genai.configure(api_key=api_key)
    
    def call_llm(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        model_api: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Call LLM with unified interface
        
        Args:
            model_id: Model identifier
            messages: List of message dicts with 'role' and 'content'
            model_api: Either 'openrouter' or 'google_ai'
            temperature: Model temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            
        Returns:
            Dict with response, latency, cost_usd, and error fields
        """
        start_time = time.time()
        
        try:
            if model_api == "openrouter":
                return self._call_openrouter(model_id, messages, temperature, max_tokens, start_time)
            elif model_api == "google_ai":
                return self._call_google_ai(model_id, messages, temperature, max_tokens, start_time)
            else:
                return {
                    "response": "",
                    "latency": time.time() - start_time,
                    "cost_usd": 0,
                    "error": f"Unknown API: {model_api}"
                }
        except Exception as e:
            return {
                "response": "",
                "latency": time.time() - start_time,
                "cost_usd": 0,
                "error": str(e)
            }
    
    def _call_openrouter(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> Dict[str, Any]:
        """Call OpenRouter API"""
        if not self.openrouter_client:
            return {
                "response": "",
                "latency": time.time() - start_time,
                "cost_usd": 0,
                "error": "OpenRouter API key not configured"
            }
        
        try:
            completion = self.openrouter_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout
            )
            
            # Extract cost from headers if available
            cost_usd = 0
            if hasattr(completion, '_headers'):
                cost_header = completion._headers.get('X-RateLimit-Cost', '0')
                # OpenRouter returns cost in millionths of USD
                cost_usd = float(cost_header) / 1_000_000
            
            return {
                "response": completion.choices[0].message.content,
                "latency": time.time() - start_time,
                "cost_usd": cost_usd,
                "error": None
            }
            
        except Timeout:
            return {
                "response": "",
                "latency": self.timeout,
                "cost_usd": 0,
                "error": "timeout"
            }
    
    def _call_google_ai(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> Dict[str, Any]:
        """Call Google AI API"""
        if not self.google_api_key:
            return {
                "response": "",
                "latency": time.time() - start_time,
                "cost_usd": 0,
                "error": "Google AI API key not configured"
            }
        
        try:
            # Convert messages to Google AI format
            # Combine all messages into a single prompt
            prompt_parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg['role'] == 'user':
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            full_prompt = "\n\n".join(prompt_parts)
            
            # Check if this is a Gemma model and apply rate limiting
            if 'gemma' in model_id.lower():
                # Estimate tokens (roughly 1 token per 4 characters)
                estimated_tokens = len(full_prompt) // 4
                
                # Check if we need to reset the minute counter
                current_time = time.time()
                if current_time - self.gemma_minute_start > 60:
                    self.gemma_tokens_this_minute = 0
                    self.gemma_minute_start = current_time
                
                # Check if we would exceed rate limit
                if self.gemma_tokens_this_minute + estimated_tokens > self.gemma_rate_limit:
                    # Wait until the next minute
                    wait_time = 60 - (current_time - self.gemma_minute_start)
                    if wait_time > 0:
                        print(f"Rate limit approaching for Gemma model. Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        # Reset counters
                        self.gemma_tokens_this_minute = 0
                        self.gemma_minute_start = time.time()
                
                # Update token count
                self.gemma_tokens_this_minute += estimated_tokens
            
            # Initialize model - remove 'google/' prefix if present
            model_name = model_id.replace('google/', '') if model_id.startswith('google/') else model_id
            model = genai.GenerativeModel(model_name)
            
            # Configure safety settings to be permissive for legal content
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Generate response with timeout
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={"timeout": self.timeout}
            )
            
            # Calculate cost based on token usage
            cost_usd = self._calculate_google_cost(
                model_name,
                full_prompt,
                response.text if response.text else ""
            )
            
            return {
                "response": response.text if response.text else "",
                "latency": time.time() - start_time,
                "cost_usd": cost_usd,
                "error": None
            }
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {
                    "response": "",
                    "latency": self.timeout,
                    "cost_usd": 0,
                    "error": "timeout"
                }
            return {
                "response": "",
                "latency": time.time() - start_time,
                "cost_usd": 0,
                "error": error_msg
            }
    
    def _calculate_google_cost(
        self,
        model_id: str,
        input_text: str,
        output_text: str
    ) -> float:
        """Calculate cost for Google AI models"""
        if model_id not in self.google_pricing:
            return 0
        
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(input_text) / 4
        output_tokens = len(output_text) / 4
        
        pricing = self.google_pricing[model_id]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def test_connectivity(self, model_id: str, model_api: str) -> Dict[str, Any]:
        """
        Test if a model is accessible
        
        Returns:
            Dict with 'success' bool and optional 'error' message
        """
        try:
            result = self.call_llm(
                model_id=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                model_api=model_api,
                max_tokens=5
            )
            
            if result["error"]:
                return {"success": False, "error": result["error"]}
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def sanitize_model_name(model_name: str) -> str:
    """Convert model names to safe filenames"""
    safe_name = model_name.replace("/", "_")
    safe_name = safe_name.replace(":", "_")
    safe_name = safe_name.replace(".", "-")
    return safe_name