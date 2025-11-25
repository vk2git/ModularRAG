import re
from typing import Dict, Any, List
from src.core.guardrails.base_validator import BaseValidator


class PIIDetectorValidator(BaseValidator):
    """
    Detects and redacts Personally Identifiable Information (PII) from outputs.
    """
    
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    }
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        
        redact_types = self.config.get("redact_types", list(self.PII_PATTERNS.keys()))
        
        self.active_patterns = {
            pii_type: pattern 
            for pii_type, pattern in self.PII_PATTERNS.items()
            if pii_type in redact_types
        }
    
    def get_name(self) -> str:
        return "pii_detector"
    
    def get_description(self) -> str:
        types = ", ".join(self.active_patterns.keys())
        return f"Detects and redacts PII: {types}"
    
    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if validation_type != "output":
            return {
                "valid": True,
                "reason": "N/A for inputs",
                "sanitized_text": text
            }
        
        sanitized_text = text
        detected_pii = []
        
        for pii_type, pattern in self.active_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_pii.append(pii_type)
                sanitized_text = re.sub(
                    pattern, 
                    f"[{pii_type.upper()}_REDACTED]", 
                    sanitized_text
                )
        
        if detected_pii:
            return {
                "valid": False,
                "reason": f"PII detected and redacted: {', '.join(detected_pii)}",
                "sanitized_text": sanitized_text
            }
        
        return {
            "valid": True,
            "reason": "No PII detected",
            "sanitized_text": text
        }
