"""
Conversation State Management
Maintains essential facts across exchanges without storing full history
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json


class ConversationState(BaseModel):
    """Represents the distilled state of a guardianship conversation"""
    
    # Core fields
    county: str = Field(default="Genesee", description="County for guardianship filing")
    guardianship_type: Optional[str] = Field(None, description="Type: full, limited, temporary, emergency")
    party_role: Optional[str] = Field(None, description="User's role: petitioner, guardian, parent, relative")
    relationship: Optional[str] = Field(None, description="Relationship to child: grandmother, aunt, uncle, etc.")
    
    # Forms tracking
    forms_filed: List[str] = Field(default_factory=list, description="Forms already filed")
    forms_pending: List[str] = Field(default_factory=list, description="Forms that need to be filed")
    forms_mentioned: List[str] = Field(default_factory=list, description="All forms discussed")
    
    # Key facts dictionary for flexibility
    key_facts: Dict[str, Any] = Field(default_factory=dict, description="Other important facts")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    exchange_count: int = Field(default=0, description="Number of Q&A exchanges")
    
    class Config:
        json_schema_extra = {
            "example": {
                "county": "Genesee",
                "guardianship_type": "limited",
                "party_role": "petitioner",
                "relationship": "grandmother",
                "forms_filed": ["PC 650"],
                "forms_pending": ["PC 670", "PC 654"],
                "forms_mentioned": ["PC 650", "PC 670", "PC 654", "MC 20"],
                "key_facts": {
                    "child_age": "14",
                    "child_name_initial": "J",
                    "urgency": "normal",
                    "parents_consent": "yes",
                    "filing_fee_discussed": True,
                    "hearing_day_mentioned": "Thursday"
                },
                "last_updated": "2025-01-31T12:00:00",
                "exchange_count": 3
            }
        }
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update state from a dictionary of extracted facts"""
        for key, value in updates.items():
            if key in ["forms_filed", "forms_pending", "forms_mentioned"]:
                # Append to lists without duplicates
                current_list = getattr(self, key, [])
                if isinstance(value, list):
                    for item in value:
                        if item not in current_list:
                            current_list.append(item)
                elif isinstance(value, str) and value not in current_list:
                    current_list.append(value)
            elif key == "key_facts":
                # Merge key facts
                if isinstance(value, dict):
                    self.key_facts.update(value)
            elif hasattr(self, key):
                # Update direct attributes
                setattr(self, key, value)
            else:
                # Add to key_facts if not a direct attribute
                self.key_facts[key] = value
        
        self.last_updated = datetime.now()
        self.exchange_count += 1
    
    def to_context_string(self) -> str:
        """Convert state to a concise context string for prompt injection"""
        lines = ["Conversation context (distilled):"]
        
        # Always include county
        lines.append(f"- County: {self.county}")
        
        # Include other fields if set
        if self.guardianship_type:
            lines.append(f"- Guardianship Type: {self.guardianship_type.title()}")
        
        if self.party_role:
            role_desc = f"{self.party_role.title()}"
            if self.relationship:
                role_desc += f" ({self.relationship})"
            lines.append(f"- Role: {role_desc}")
        
        if self.forms_filed:
            lines.append(f"- Forms Filed: {', '.join(self.forms_filed)}")
        
        if self.forms_pending:
            lines.append(f"- Forms Pending: {', '.join(self.forms_pending)}")
        
        # Add key facts
        if self.key_facts:
            for key, value in self.key_facts.items():
                # Format key nicely
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, bool):
                    formatted_value = "Yes" if value else "No"
                else:
                    formatted_value = str(value)
                lines.append(f"- {formatted_key}: {formatted_value}")
        
        return "\n".join(lines)
    
    def has_meaningful_context(self) -> bool:
        """Check if state has any meaningful context beyond defaults"""
        return (
            self.guardianship_type is not None or
            self.party_role is not None or
            len(self.forms_filed) > 0 or
            len(self.forms_pending) > 0 or
            len(self.key_facts) > 0
        )
    
    def clear_sensitive_data(self) -> None:
        """Remove any potentially sensitive information while keeping structure"""
        # Remove name initials or specific identifiers
        sensitive_keys = ['child_name', 'child_name_initial', 'address', 'phone']
        for key in sensitive_keys:
            if key in self.key_facts:
                del self.key_facts[key]
    
    def merge_with(self, other: 'ConversationState') -> None:
        """Merge another state into this one, with other taking precedence"""
        if other.guardianship_type:
            self.guardianship_type = other.guardianship_type
        if other.party_role:
            self.party_role = other.party_role
        if other.relationship:
            self.relationship = other.relationship
        
        # Merge form lists
        for form in other.forms_filed:
            if form not in self.forms_filed:
                self.forms_filed.append(form)
        
        for form in other.forms_pending:
            if form not in self.forms_pending:
                self.forms_pending.append(form)
        
        for form in other.forms_mentioned:
            if form not in self.forms_mentioned:
                self.forms_mentioned.append(form)
        
        # Merge key facts
        self.key_facts.update(other.key_facts)
        
        # Update metadata
        self.last_updated = datetime.now()
        self.exchange_count += other.exchange_count
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationState':
        """Deserialize from JSON string"""
        return cls.model_validate_json(json_str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary suitable for logging or display"""
        return {
            "county": self.county,
            "type": self.guardianship_type,
            "role": self.party_role,
            "forms_status": {
                "filed": len(self.forms_filed),
                "pending": len(self.forms_pending)
            },
            "facts_count": len(self.key_facts),
            "exchanges": self.exchange_count
        }