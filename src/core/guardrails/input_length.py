from typing import Dict, Any
from src.core.guardrails.base_validator import BaseValidator


class InputLengthValidator(BaseValidator):
    """
    Validates input length to prevent DoS attacks via excessively long inputs.
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.max_length = self.config.get("max_length", 10000)
    
    def get_name(self) -> str:
        return "input_length"
    
    def get_description(self) -> str:
        return f"Validates input does not exceed {self.max_length} characters"
    
    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if validation_type != "input":
            return {
                "valid": True,
                "reason": "N/A for outputs",
                "sanitized_text": text
            }
        
        if len(text) > self.max_length:
            return {
                "valid": False,
                "reason": f"Input exceeds maximum length of {self.max_length} characters",
                "sanitized_text": text[:self.max_length]
            }
        
        return {
            "valid": True,
            "reason": "Input length within limits",
            "sanitized_text": text
        }
