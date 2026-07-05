from typing import Dict, Any, List
import src.core.guardrails.validators as v

class GuardrailsManager:
    """
    Manages and orchestrates guardrail validations.
    """
    def __init__(self, config: dict, llm=None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.llm = llm

    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """Validate user input using enabled input validations."""
        if not self.enabled:
            return {"valid": True, "reason": "Guardrails disabled", "sanitized_input": user_input}
        
        sanitized = user_input
        validator_configs = self.config.get("validators", {})
        
        # 1. Empty Input
        if validator_configs.get("empty_input", {}).get("enabled", False):
            res = v.validate_empty_input(sanitized)
            if not res["valid"]: return {"valid": False, "reason": res["reason"], "sanitized_input": res["sanitized_text"]}
            sanitized = res["sanitized_text"]
            
        # 2. Input Length
        if validator_configs.get("input_length", {}).get("enabled", False):
            max_len = validator_configs["input_length"].get("max_length", 10000)
            res = v.validate_input_length(sanitized, max_len)
            if not res["valid"]: return {"valid": False, "reason": res["reason"], "sanitized_input": res["sanitized_text"]}
            sanitized = res["sanitized_text"]
            
        # 3. Special Characters
        if validator_configs.get("special_characters", {}).get("enabled", False):
            max_ratio = validator_configs["special_characters"].get("max_ratio", 0.3)
            res = v.validate_special_characters(sanitized, max_ratio)
            if not res["valid"]: return {"valid": False, "reason": res["reason"], "sanitized_input": res["sanitized_text"]}
            sanitized = res["sanitized_text"]
            
        # 4. Prompt Injection
        if validator_configs.get("prompt_injection", {}).get("enabled", False):
            pattern_type = validator_configs["prompt_injection"].get("patterns", "default")
            res = v.validate_prompt_injection(sanitized, pattern_type)
            if not res["valid"]: return {"valid": False, "reason": res["reason"], "sanitized_input": res["sanitized_text"]}
            sanitized = res["sanitized_text"]
            
        # 5. Topic Restriction
        if validator_configs.get("topic_restriction", {}).get("enabled", False):
            allowed = validator_configs["topic_restriction"].get("allowed_topics", [])
            res = v.validate_topic_restriction(sanitized, allowed, self.llm)
            if not res["valid"]: return {"valid": False, "reason": res["reason"], "sanitized_input": res["sanitized_text"]}
            sanitized = res["sanitized_text"]
            
        # 6. Toxicity Filter
        if validator_configs.get("toxicity_filter", {}).get("enabled", False):
            res = v.validate_toxicity(sanitized, "input", self.llm)
            if not res["valid"]: return {"valid": False, "reason": res["reason"], "sanitized_input": res["sanitized_text"]}
            sanitized = res["sanitized_text"]
            
        return {"valid": True, "reason": "All input validations passed", "sanitized_input": sanitized}

    def validate_output(self, output: str) -> Dict[str, Any]:
        """Validate LLM output using enabled output validations."""
        if not self.enabled:
            return {"valid": True, "reason": "Guardrails disabled", "sanitized_output": output}
            
        sanitized = output
        warnings = []
        validator_configs = self.config.get("validators", {})
        
        # 1. PII Detector
        if validator_configs.get("pii_detector", {}).get("enabled", False):
            redact = validator_configs["pii_detector"].get("redact_types")
            res = v.validate_pii(sanitized, redact)
            if not res["valid"]:
                warnings.append(res["reason"])
                sanitized = res["sanitized_text"]
                
        # 2. Toxicity Filter
        if validator_configs.get("toxicity_filter", {}).get("enabled", False):
            res = v.validate_toxicity(sanitized, "output", self.llm)
            if not res["valid"]:
                warnings.append(res["reason"])
                sanitized = res["sanitized_text"]
                
        if warnings:
            return {"valid": False, "reason": "; ".join(warnings), "sanitized_output": sanitized}
            
        return {"valid": True, "reason": "All output validations passed", "sanitized_output": output}

    def get_safe_response(self, validation_result: Dict[str, Any]) -> str:
        """Generate a safe response when validation fails."""
        return (
            "I apologize, but I cannot process your request due to security concerns. "
            "Please rephrase your question and try again."
        )
