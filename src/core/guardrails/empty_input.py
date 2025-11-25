from typing import Dict, Any
from src.core.guardrails.base_validator import BaseValidator


class EmptyInputValidator(BaseValidator):
    """
    Validates that input is not empty or whitespace-only.
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    def get_name(self) -> str:
        return "empty_input"
    
    def get_description(self) -> str:
        return "Validates input is not empty"
    
    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if validation_type != "input":
            return {
                "valid": True,
                "reason": "N/A for outputs",
                "sanitized_text": text
            }
        
        if not text or not text.strip():
            return {
                "valid": False,
                "reason": "Empty input detected",
                "sanitized_text": ""
            }
        
        return {
            "valid": True,
            "reason": "Input is not empty",
            "sanitized_text": text
        }
