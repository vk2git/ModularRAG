import re
from typing import Dict, Any, List
from src.core.guardrails.base_validator import BaseValidator


class PromptInjectionValidator(BaseValidator):
    """
    Detects common prompt injection attack patterns.
    """
    
    DEFAULT_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?above",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"</\s*system\s*>",
        r"<\s*system\s*>",
        r"act\s+as\s+(a\s+)?different",
        r"pretend\s+(you\s+are|to\s+be)",
        r"roleplay\s+as",
        r"simulate\s+(being|a)",
    ]
    
    STRICT_PATTERNS = DEFAULT_PATTERNS + [
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
        r"god\s+mode",
        r"override\s+",
        r"bypass\s+",
    ]
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        
        pattern_type = self.config.get("patterns", "default")
        
        if pattern_type == "strict":
            patterns = self.STRICT_PATTERNS
        elif isinstance(pattern_type, list):
            patterns = pattern_type
        else:
            patterns = self.DEFAULT_PATTERNS
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in patterns
        ]
    
    def get_name(self) -> str:
        return "prompt_injection"
    
    def get_description(self) -> str:
        return "Detects prompt injection attack patterns"
    
    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if validation_type != "input":
            return {
                "valid": True,
                "reason": "N/A for outputs",
                "sanitized_text": text
            }
        
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return {
                    "valid": False,
                    "reason": f"Potential prompt injection detected: pattern '{pattern.pattern}'",
                    "sanitized_text": text
                }
        
        return {
            "valid": True,
            "reason": "No injection patterns detected",
            "sanitized_text": text
        }
