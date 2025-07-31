"""
Disclaimer Policy Module
Implements intelligent, context-aware disclaimer insertion
"""

from typing import List, Optional, Dict, Any
import re


class DisclaimerPolicy:
    """Manages when and how disclaimers are added to responses"""
    
    # Disclaimer templates for different scenarios
    DISCLAIMERS = {
        "legal_risk": "\n\n⚠️ *Given the urgent nature of your situation, please note that this information is for educational purposes only. For immediate legal assistance, contact the Genesee County Probate Court at (810) 257-3528 or consult with an attorney.*",
        
        "out_of_scope": "\n\n*This response addresses topics outside minor guardianship in Genesee County. For accurate guidance on your specific situation, please consult with an appropriate legal professional.*",
        
        "personalized_advice": "\n\n*This information is educational and cannot replace personalized legal advice. For guidance specific to your circumstances, please consult with a qualified attorney.*",
        
        "emergency": "\n\n🚨 *This appears to be an emergency situation. While this information may help, please contact emergency services if there's immediate danger, or the court at (810) 257-3528 for urgent guardianship matters.*",
        
        "default": "\n\n*This information is provided for educational purposes only and does not constitute legal advice. For specific legal guidance, please consult with a qualified attorney.*"
    }
    
    @staticmethod
    def should_append_disclaimer(
        risk_flags: List[str],
        user_query: str,
        response_text: str,
        is_out_of_scope: bool = False
    ) -> bool:
        """
        Determine if a disclaimer should be appended based on context
        
        Args:
            risk_flags: List of identified risk indicators
            user_query: The original user question
            response_text: The generated response
            is_out_of_scope: Whether the query is outside scope
            
        Returns:
            bool: True if disclaimer should be appended
        """
        # Always add disclaimer for high-risk situations
        if any(flag in risk_flags for flag in ['legal_risk', 'emergency', 'crisis']):
            return True
        
        # Add disclaimer for out-of-scope queries
        if is_out_of_scope or 'out_of_scope' in risk_flags:
            return True
        
        # Check if user is asking for personalized advice
        if DisclaimerPolicy.detects_personalized_advice_request(user_query):
            return True
        
        # Check if response suggests actions that could have legal consequences
        if DisclaimerPolicy.contains_legal_recommendations(response_text):
            return True
        
        # No disclaimer needed for purely factual responses
        return False
    
    @staticmethod
    def detects_personalized_advice_request(query: str) -> bool:
        """
        Detect if user is asking for personalized legal advice
        
        Args:
            query: User's question
            
        Returns:
            bool: True if personalized advice is being requested
        """
        query_lower = query.lower()
        
        # Patterns indicating request for personal advice
        personal_advice_patterns = [
            r'\b(what should i do|should i file|can i get)\b',
            r'\b(my case|my situation|my child|my nephew|my niece|my grandchild)\b',
            r'\b(advise me|help me decide|recommend|best option for me)\b',
            r'\b(in my circumstances|for my specific|my particular)\b',
            r'\bwhat.*best\b.*\b(for me|in my case|in my situation)\b',
            r'\b(am i eligible|do i qualify)\b'
        ]
        
        for pattern in personal_advice_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    @staticmethod
    def contains_legal_recommendations(response: str) -> bool:
        """
        Check if response contains actionable legal recommendations
        
        Args:
            response: Generated response text
            
        Returns:
            bool: True if response contains legal recommendations
        """
        response_lower = response.lower()
        
        # Patterns indicating legal recommendations
        recommendation_patterns = [
            r'\b(you should|you must|you need to|it is recommended that you)\b',
            r'\b(file.*immediately|contact.*attorney|seek.*legal)\b',
            r'\b(your best option|we recommend|it would be advisable)\b',
            r'\b(you are required to|you have to|mandatory that you)\b'
        ]
        
        for pattern in recommendation_patterns:
            if re.search(pattern, response_lower):
                return True
        
        return False
    
    @staticmethod
    def get_appropriate_disclaimer(
        risk_flags: List[str],
        user_query: str,
        is_out_of_scope: bool = False
    ) -> str:
        """
        Get the most appropriate disclaimer based on context
        
        Args:
            risk_flags: List of identified risk indicators
            user_query: The original user question
            is_out_of_scope: Whether the query is outside scope
            
        Returns:
            str: The appropriate disclaimer text
        """
        # Priority order for disclaimer selection
        if 'emergency' in risk_flags or 'crisis' in user_query.lower():
            return DisclaimerPolicy.DISCLAIMERS["emergency"]
        
        if 'legal_risk' in risk_flags:
            return DisclaimerPolicy.DISCLAIMERS["legal_risk"]
        
        if is_out_of_scope or 'out_of_scope' in risk_flags:
            return DisclaimerPolicy.DISCLAIMERS["out_of_scope"]
        
        if DisclaimerPolicy.detects_personalized_advice_request(user_query):
            return DisclaimerPolicy.DISCLAIMERS["personalized_advice"]
        
        # Default disclaimer if none of the specific cases apply
        return DisclaimerPolicy.DISCLAIMERS["default"]
    
    @staticmethod
    def apply_disclaimer_policy(
        response_text: str,
        risk_flags: List[str],
        user_query: str,
        is_out_of_scope: bool = False
    ) -> str:
        """
        Apply disclaimer policy to response
        
        Args:
            response_text: The generated response
            risk_flags: List of identified risk indicators
            user_query: The original user question
            is_out_of_scope: Whether the query is outside scope
            
        Returns:
            str: Response with disclaimer appended if needed
        """
        if DisclaimerPolicy.should_append_disclaimer(
            risk_flags, user_query, response_text, is_out_of_scope
        ):
            disclaimer = DisclaimerPolicy.get_appropriate_disclaimer(
                risk_flags, user_query, is_out_of_scope
            )
            return response_text + disclaimer
        
        return response_text


def should_append_disclaimer(
    risk_flags: List[str],
    user_intent: str,
    scope: str
) -> bool:
    """
    Convenience function for backward compatibility
    
    Args:
        risk_flags: List of identified risk indicators
        user_intent: Detected user intent (can be query text)
        scope: Whether query is in scope ("in_scope" or "out_of_scope")
        
    Returns:
        bool: True if disclaimer should be appended
    """
    is_out_of_scope = scope == "out_of_scope"
    return DisclaimerPolicy.should_append_disclaimer(
        risk_flags=risk_flags,
        user_query=user_intent,
        response_text="",  # Not needed for basic check
        is_out_of_scope=is_out_of_scope
    )