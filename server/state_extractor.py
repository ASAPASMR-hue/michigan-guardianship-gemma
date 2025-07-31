"""
State Extraction Module
Extracts conversation facts from user questions and system responses
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from server.conversation_state import ConversationState


class StateExtractor:
    """Extracts key facts from conversation exchanges"""
    
    # County patterns
    COUNTY_PATTERNS = [
        r'\b(?:in\s+)?(\w+)\s+[Cc]ounty\b',
        r'\b(?:filing|filed|file)\s+in\s+(\w+)\b',
        r'\b(\w+)\s+[Cc]ounty\s+(?:court|probate)\b'
    ]
    
    # Guardianship type patterns
    GUARDIANSHIP_TYPE_PATTERNS = {
        'full': r'\b(?:full|complete)\s+guardianship\b',
        'limited': r'\b(?:limited|partial)\s+guardianship\b',
        'temporary': r'\b(?:temporary|temp|emergency)\s+guardianship\b',
        'emergency': r'\b(?:emergency|urgent|immediate)\s+guardianship\b'
    }
    
    # Role patterns
    ROLE_PATTERNS = {
        'petitioner': r'\b(?:I\s+am|I\'m)\s+(?:the\s+)?petitioner\b',
        'guardian': r'\b(?:I\s+am|I\'m)\s+(?:the\s+)?(?:proposed\s+)?guardian\b',
        'parent': r'\b(?:I\s+am|I\'m)\s+(?:the\s+)?parent\b',
        'relative': r'\b(?:I\s+am|I\'m)\s+(?:a\s+)?relative\b'
    }
    
    # Relationship patterns
    RELATIONSHIP_PATTERNS = [
        r'\b(?:I\s+am|I\'m)\s+(?:the\s+)?(?:child\'s\s+)?(\w+mother|grandmother|grandfather|aunt|uncle|sister|brother|cousin)\b',
        r'\b(?:their|the\s+child\'s)\s+(\w+mother|grandmother|grandfather|aunt|uncle|sister|brother|cousin)\b',
        r'\b(?:as\s+(?:a|the)\s+)?(\w+mother|grandmother|grandfather|aunt|uncle|sister|brother|cousin)\s+(?:seeking|filing|petitioning)\b'
    ]
    
    # Form patterns
    FORM_PATTERNS = {
        'mentioned': r'\b(?:Form\s+)?(PC|MC|CC|DC)\s*(\d{2,4})\b',
        'filed': r'\b(?:filed|submitted|completed)\s+(?:Form\s+)?(PC|MC|CC|DC)\s*(\d{2,4})\b',
        'pending': r'\b(?:need\s+to\s+file|must\s+file|will\s+file|filing)\s+(?:Form\s+)?(PC|MC|CC|DC)\s*(\d{2,4})\b'
    }
    
    # Common fact patterns
    FACT_PATTERNS = {
        'child_age': [
            r'\b(?:child|minor|kid)\s+is\s+(\d{1,2})\s*(?:years?\s*old)?\b',
            r'\b(\d{1,2})\s*(?:-?\s*year\s*-?\s*old)\s+(?:child|minor|kid)\b',
            r'\b(?:age|aged)\s+(\d{1,2})\b'
        ],
        'urgency': [
            r'\b(emergency|urgent|immediate|crisis|right\s+away|asap|now)\b',
            r'\b(not\s+urgent|no\s+rush|regular|normal)\b'
        ],
        'parents_consent': [
            r'\b(?:parents?|mother|father)\s+(?:consent|agree|willing|support)\b',
            r'\b(?:parents?|mother|father)\s+(?:oppose|disagree|against|won\'t\s+consent)\b'
        ],
        'filing_fee_discussed': [
            r'\$175\s*(?:filing\s*)?fee',
            r'filing\s+fee',
            r'fee\s+waiver',
            r'MC\s*20'
        ],
        'hearing_day_mentioned': [
            r'\b[Tt]hursday\s+hearing',
            r'hearing.*[Tt]hursday',
            r'[Tt]hursday.*court'
        ]
    }
    
    @staticmethod
    def extract_from_text(text: str, is_user_message: bool = True) -> Dict[str, Any]:
        """
        Extract facts from a single text (question or answer)
        
        Args:
            text: The text to extract from
            is_user_message: Whether this is a user message (affects extraction logic)
            
        Returns:
            Dictionary of extracted facts
        """
        extracted = {}
        text_lower = text.lower()
        
        # Extract county
        for pattern in StateExtractor.COUNTY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                county = match.group(1).capitalize()
                # Only update if it's a Michigan county (basic validation)
                if county not in ['The', 'This', 'That', 'My', 'Our']:
                    extracted['county'] = county
                break
        
        # Extract guardianship type
        for g_type, pattern in StateExtractor.GUARDIANSHIP_TYPE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                extracted['guardianship_type'] = g_type
                break
        
        # Extract role (mainly from user messages)
        if is_user_message:
            for role, pattern in StateExtractor.ROLE_PATTERNS.items():
                if re.search(pattern, text, re.IGNORECASE):
                    extracted['party_role'] = role
                    break
        
        # Extract relationship
        for pattern in StateExtractor.RELATIONSHIP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                relationship = match.group(1).lower()
                # Clean up common variations
                relationship = relationship.replace('grand', 'grand').replace('step', 'step-')
                extracted['relationship'] = relationship
                # If relationship found, infer role as relative/petitioner
                if 'party_role' not in extracted and is_user_message:
                    extracted['party_role'] = 'relative'
                break
        
        # Extract forms
        forms_data = StateExtractor._extract_forms(text)
        if forms_data['mentioned']:
            extracted['forms_mentioned'] = forms_data['mentioned']
        if forms_data['filed']:
            extracted['forms_filed'] = forms_data['filed']
        if forms_data['pending']:
            extracted['forms_pending'] = forms_data['pending']
        
        # Extract other facts
        key_facts = {}
        for fact_name, patterns in StateExtractor.FACT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if fact_name == 'child_age':
                        key_facts[fact_name] = match.group(1)
                    elif fact_name == 'urgency':
                        if 'emergency' in match.group(0).lower() or 'urgent' in match.group(0).lower():
                            key_facts[fact_name] = 'high'
                        else:
                            key_facts[fact_name] = 'normal'
                    elif fact_name == 'parents_consent':
                        if any(word in match.group(0).lower() for word in ['oppose', 'against', 'won\'t', 'disagree']):
                            key_facts[fact_name] = 'no'
                        else:
                            key_facts[fact_name] = 'yes'
                    else:
                        key_facts[fact_name] = True
                    break
        
        if key_facts:
            extracted['key_facts'] = key_facts
        
        return extracted
    
    @staticmethod
    def _extract_forms(text: str) -> Dict[str, List[str]]:
        """Extract form references and categorize them"""
        forms = {
            'mentioned': [],
            'filed': [],
            'pending': []
        }
        
        # First find all form mentions
        all_forms = set()
        for match in re.finditer(StateExtractor.FORM_PATTERNS['mentioned'], text, re.IGNORECASE):
            form_type = match.group(1).upper()
            form_num = match.group(2)
            form_code = f"{form_type} {form_num}"
            all_forms.add(form_code)
        
        forms['mentioned'] = list(all_forms)
        
        # Check for filed forms
        for match in re.finditer(StateExtractor.FORM_PATTERNS['filed'], text, re.IGNORECASE):
            form_type = match.group(1).upper()
            form_num = match.group(2)
            form_code = f"{form_type} {form_num}"
            if form_code not in forms['filed']:
                forms['filed'].append(form_code)
        
        # Check for pending forms
        for match in re.finditer(StateExtractor.FORM_PATTERNS['pending'], text, re.IGNORECASE):
            form_type = match.group(1).upper()
            form_num = match.group(2)
            form_code = f"{form_type} {form_num}"
            if form_code not in forms['pending']:
                forms['pending'].append(form_code)
        
        return forms
    
    @staticmethod
    def extract_from_exchange(
        user_question: str,
        assistant_response: str,
        current_state: Optional[ConversationState] = None
    ) -> ConversationState:
        """
        Extract facts from a complete Q&A exchange
        
        Args:
            user_question: The user's question
            assistant_response: The assistant's response
            current_state: Current conversation state (if any)
            
        Returns:
            Updated conversation state
        """
        # Start with current state or create new
        if current_state:
            state = current_state.model_copy()
        else:
            state = ConversationState()
        
        # Extract from user question
        user_facts = StateExtractor.extract_from_text(user_question, is_user_message=True)
        
        # Extract from assistant response
        assistant_facts = StateExtractor.extract_from_text(assistant_response, is_user_message=False)
        
        # Merge facts (user facts take precedence for personal info)
        merged_facts = {}
        
        # User facts about themselves are more authoritative
        for key in ['party_role', 'relationship']:
            if key in user_facts:
                merged_facts[key] = user_facts[key]
            elif key in assistant_facts:
                merged_facts[key] = assistant_facts[key]
        
        # Other facts can come from either source
        for facts in [user_facts, assistant_facts]:
            for key, value in facts.items():
                if key not in ['party_role', 'relationship']:
                    if key == 'key_facts' and key in merged_facts:
                        # Merge key_facts dictionaries
                        merged_facts[key].update(value)
                    else:
                        merged_facts[key] = value
        
        # Update state with merged facts
        state.update_from_dict(merged_facts)
        
        return state
    
    @staticmethod
    def should_update_context(old_state: ConversationState, new_state: ConversationState) -> bool:
        """
        Determine if the context has changed enough to warrant updating
        
        Args:
            old_state: Previous conversation state
            new_state: New conversation state
            
        Returns:
            True if context should be updated
        """
        # Always update if we didn't have meaningful context before
        if not old_state.has_meaningful_context() and new_state.has_meaningful_context():
            return True
        
        # Check for changes in key fields
        if (old_state.guardianship_type != new_state.guardianship_type or
            old_state.party_role != new_state.party_role or
            old_state.relationship != new_state.relationship):
            return True
        
        # Check for new forms
        if (len(new_state.forms_filed) > len(old_state.forms_filed) or
            len(new_state.forms_pending) > len(old_state.forms_pending)):
            return True
        
        # Check for significant new facts
        new_facts = set(new_state.key_facts.keys()) - set(old_state.key_facts.keys())
        if new_facts:
            return True
        
        return False