import re
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# PII Patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}

# Prompt Injection Patterns
DEFAULT_INJECTION_PATTERNS = [
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

STRICT_INJECTION_PATTERNS = DEFAULT_INJECTION_PATTERNS + [
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"god\s+mode",
    r"override\s+",
    r"bypass\s+",
]

def validate_empty_input(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"valid": False, "reason": "Empty input detected", "sanitized_text": ""}
    return {"valid": True, "reason": "Input is not empty", "sanitized_text": text}

def validate_input_length(text: str, max_length: int = 10000) -> Dict[str, Any]:
    if len(text) > max_length:
        return {"valid": False, "reason": f"Input exceeds maximum length of {max_length} characters", "sanitized_text": text[:max_length]}
    return {"valid": True, "reason": "Input length within limits", "sanitized_text": text}

def validate_special_characters(text: str, max_ratio: float = 0.3) -> Dict[str, Any]:
    if not text:
        return {"valid": True, "reason": "Empty text", "sanitized_text": text}
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    ratio = special_char_count / len(text)
    if ratio > max_ratio:
        return {"valid": False, "reason": f"Excessive special characters detected", "sanitized_text": text}
    return {"valid": True, "reason": "Special character ratio acceptable", "sanitized_text": text}

def validate_prompt_injection(text: str, pattern_type: str = "default") -> Dict[str, Any]:
    patterns = STRICT_INJECTION_PATTERNS if pattern_type == "strict" else (pattern_type if isinstance(pattern_type, list) else DEFAULT_INJECTION_PATTERNS)
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"valid": False, "reason": f"Potential prompt injection detected", "sanitized_text": text}
    return {"valid": True, "reason": "No injection patterns detected", "sanitized_text": text}

def validate_pii(text: str, redact_types: List[str] = None) -> Dict[str, Any]:
    if redact_types is None:
        redact_types = list(PII_PATTERNS.keys())
    
    sanitized_text = text
    detected_pii = []
    
    for pii_type in redact_types:
        if pii_type in PII_PATTERNS:
            pattern = PII_PATTERNS[pii_type]
            if re.findall(pattern, text):
                detected_pii.append(pii_type)
                sanitized_text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitized_text)
                
    if detected_pii:
        return {"valid": False, "reason": f"PII detected and redacted: {', '.join(detected_pii)}", "sanitized_text": sanitized_text}
    return {"valid": True, "reason": "No PII detected", "sanitized_text": text}

def validate_topic_restriction(text: str, allowed_topics: list, llm) -> Dict[str, Any]:
    if not llm or not allowed_topics:
        return {"valid": True, "sanitized_text": text}
    try:
        topics_str = ", ".join(allowed_topics)
        template = """
        You are a strict topic classifier.
        
        Allowed Topics: {topics}
        
        User Query: "{query}"
        
        Is the User Query related to ANY of the Allowed Topics?
        Answer only "YES" or "NO".
        If the query is a greeting (hello, hi) or meta-question (who are you), answer "YES".
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"topics": topics_str, "query": text}).strip().upper()
        
        if "NO" in result:
            return {"valid": False, "reason": f"Query is off-topic. Allowed topics: {topics_str}", "sanitized_text": text}
    except Exception as e:
        print(f"⚠️  Topic validation failed (LLM error): {e}")
    return {"valid": True, "sanitized_text": text}

def validate_toxicity(text: str, validation_type: str, llm) -> Dict[str, Any]:
    if not llm:
        return {"valid": True, "sanitized_text": text}
    try:
        template = """
        You are a content moderation system.
        Analyze the following text for toxicity, hate speech, or harmful content.
        
        Text: "{text}"
        
        Is this text TOXIC or HARMFUL?
        Answer only "YES" or "NO".
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": text}).strip().upper()
        
        if "YES" in result:
            return {"valid": False, "reason": "Content detected as toxic or harmful.", "sanitized_text": "[REDACTED]" if validation_type == "output" else text}
    except Exception as e:
        print(f"⚠️  Toxicity validation failed (LLM error): {e}")
    return {"valid": True, "sanitized_text": text}
