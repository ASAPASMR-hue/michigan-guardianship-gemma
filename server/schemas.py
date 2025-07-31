"""
Schema definitions for structured backend responses
Enables precise testing and better UI presentation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Citation(BaseModel):
    """Represents a citation/reference to source material"""
    source_id: str = Field(..., description="e.g., pc670.pdf#chunk_17")
    title: str = Field(..., description="Human-readable title of the source")
    url: Optional[str] = Field(None, description="Optional URL to the source")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "pc651.pdf#chunk_3",
                "title": "Form PC 651 - Petition to Appoint Guardian",
                "url": "https://courts.michigan.gov/forms/pc651.pdf"
            }
        }


class Step(BaseModel):
    """Represents a procedural step with its own citations"""
    text: str = Field(..., description="The step description")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting this step")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "File Form PC 651 (Petition to Appoint Guardian) with the court",
                "citations": [
                    {
                        "source_id": "pc651.pdf#chunk_1",
                        "title": "Form PC 651 Instructions",
                        "url": None
                    }
                ]
            }
        }


class AnswerPayload(BaseModel):
    """Main response structure for guardianship questions"""
    answer_markdown: str = Field(..., description="Final prose answer in markdown format")
    citations: List[Citation] = Field(default_factory=list, description="All citations referenced in the answer")
    forms: List[str] = Field(default_factory=list, description='Form numbers mentioned (e.g., ["PC 650", "PC 670"])')
    fees: List[str] = Field(default_factory=list, description='Fees mentioned (e.g., ["$175 filing fee"])')
    steps: List[Step] = Field(default_factory=list, description="Procedural steps if applicable")
    risk_flags: List[str] = Field(default_factory=list, description='Risk indicators (e.g., ["legal_risk", "out_of_scope", "icwa_sensitive"])')
    conversation_state: Optional[Dict[str, Any]] = Field(None, description="Current conversation state")
    state_updates: Optional[Dict[str, Any]] = Field(None, description="Facts extracted from this exchange")
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug info: retrieval hits, tokens, costs, etc.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer_markdown": "To file for guardianship in Genesee County, you'll need to:\n\n1. Complete Form PC 651...",
                "citations": [
                    {
                        "source_id": "pc651.pdf#chunk_1",
                        "title": "Form PC 651 - Petition to Appoint Guardian",
                        "url": None
                    }
                ],
                "forms": ["PC 651", "PC 654", "MC 20"],
                "fees": ["$175 filing fee"],
                "steps": [
                    {
                        "text": "Obtain and complete Form PC 651 (Petition to Appoint Guardian)",
                        "citations": []
                    }
                ],
                "risk_flags": [],
                "debug": {
                    "retrieval_hits": 5,
                    "model": "google/gemma-3-4b-it",
                    "processing_time": 1.23
                }
            }
        }


class StructuredResponse(BaseModel):
    """Complete API response including metadata"""
    data: AnswerPayload = Field(..., description="The structured answer data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "answer_markdown": "To file for guardianship...",
                    "citations": [],
                    "forms": ["PC 651"],
                    "fees": ["$175 filing fee"],
                    "steps": [],
                    "risk_flags": [],
                    "debug": {}
                },
                "metadata": {
                    "model": "google/gemma-3-4b-it",
                    "retrieval_metadata": {
                        "complexity": "standard",
                        "chunks_retrieved": 5
                    },
                    "processing_time": 1.23
                },
                "timestamp": "2025-01-31T12:00:00"
            }
        }


# Validation helpers
def extract_forms_from_text(text: str) -> List[str]:
    """Extract form numbers from text (e.g., PC 651, MC 20)"""
    import re
    # Match forms with or without spaces, and with optional hyphens
    pattern = r'\b(?:PC|MC|CC|DC)[\s-]*(\d{2,4})\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Normalize format to "PC 651" style
    forms = []
    for match in matches:
        # Extract the prefix and number separately
        full_match = re.search(r'\b(PC|MC|CC|DC)[\s-]*' + match + r'\b', text, re.IGNORECASE)
        if full_match:
            prefix = full_match.group(1).upper()
            forms.append(f"{prefix} {match}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_forms = []
    for form in forms:
        if form not in seen:
            seen.add(form)
            unique_forms.append(form)
    
    return unique_forms


def extract_fees_from_text(text: str) -> List[str]:
    """Extract fee amounts from text"""
    import re
    # Match dollar amounts with optional description
    pattern = r'\$\d+(?:\.\d{2})?\s*(?:filing fee|fee waiver|court fee)?'
    fees = re.findall(pattern, text, re.IGNORECASE)
    return [f.strip() for f in fees]


def extract_risk_flags(text: str, question: str) -> List[str]:
    """Identify risk flags from the response and question"""
    import re
    flags = []
    
    # Check for out of scope indicators
    out_of_scope_patterns = [
        "cannot provide legal advice",
        "consult an attorney",
        "outside.*scope",
        "not qualified",
        "adult guardianship"
    ]
    
    for pattern in out_of_scope_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append("out_of_scope")
            break
    
    # Check for ICWA sensitivity
    icwa_patterns = ["indian", "tribal", "native american", "icwa", "mifpa"]
    for pattern in icwa_patterns:
        if re.search(pattern, text + " " + question, re.IGNORECASE):
            flags.append("icwa_sensitive")
            break
    
    # Check for legal risk and emergency situations
    legal_risk_patterns = [
        "urgent",
        "emergency",
        "immediate",
        "crisis",
        "danger",
        "abuse",
        "neglect",
        "hospital",
        "police",
        "cps"
    ]
    
    for pattern in legal_risk_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            flags.append("legal_risk")
            # Also check for emergency specifically
            if pattern in ["emergency", "urgent", "immediate", "crisis"]:
                flags.append("emergency")
            break
    
    # Check if user is requesting personalized legal advice
    personal_advice_patterns = [
        r'\b(what should i do|should i file|can i get)\b',
        r'\b(my case|my situation|my child|my nephew|my niece|my grandchild)\b',
        r'\b(advise me|help me decide|recommend|best option for me)\b',
        r'\b(in my circumstances|for my specific|my particular)\b',
        r'\bwhat.*best\b.*\b(for me|in my case|in my situation)\b',
        r'\b(am i eligible|do i qualify)\b'
    ]
    
    question_lower = question.lower()
    for pattern in personal_advice_patterns:
        if re.search(pattern, question_lower):
            flags.append("personalized_advice_requested")
            break
    
    # Check for CPS involvement
    if re.search(r'\b(cps|child protective|foster care)\b', question_lower):
        flags.append("cps_involved")
    
    return list(set(flags))  # Remove duplicates