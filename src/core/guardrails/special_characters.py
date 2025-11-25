from typing import Dict, Any
from src.core.guardrails.base_validator import BaseValidator


class SpecialCharactersValidator(BaseValidator):
    """
    Detects excessive special characters that may indicate encoding attacks.
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.max_ratio = self.config.get("max_ratio", 0.3)
    
    def get_name(self) -> str:
        return "special_characters"
    
    def get_description(self) -> str:
        return f"Detects excessive special characters (max {self.max_ratio * 100}%)"
    
    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if validation_type != "input":
            return {
                "valid": True,
                "reason": "N/A for outputs",
                "sanitized_text": text
            }
        
        if not text:
            return {
                "valid": True,
                "reason": "Empty text",
                "sanitized_text": text
            }
        
        special_char_count = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        )
        ratio = special_char_count / len(text)
        
        if ratio > self.max_ratio:
            return {
                "valid": False,
                "reason": f"Excessive special characters detected ({ratio:.1%} > {self.max_ratio:.1%})",
                "sanitized_text": text
            }
        
        return {
            "valid": True,
            "reason": "Special character ratio acceptable",
            "sanitized_text": text
        }
