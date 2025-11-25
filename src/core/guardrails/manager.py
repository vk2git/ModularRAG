from typing import Dict, Any, List, Optional
from src.core.guardrails.base_validator import BaseValidator
from src.utils.class_loader import instantiate_class
from src.core.guardrails.prompt_injection import PromptInjectionValidator
from src.core.guardrails.input_length import InputLengthValidator
from src.core.guardrails.special_characters import SpecialCharactersValidator
from src.core.guardrails.pii_detector import PIIDetectorValidator
from src.core.guardrails.empty_input import EmptyInputValidator


class GuardrailsManager:
    """
    Manages and orchestrates multiple guardrail validators.
    
    This is a plugin-based system where each validator is independent
    and can be enabled/disabled via configuration.
    """
    
    DEFAULT_VALIDATORS = {
        "prompt_injection": PromptInjectionValidator,
        "input_length": InputLengthValidator,
        "special_characters": SpecialCharactersValidator,
        "pii_detector": PIIDetectorValidator,
        "empty_input": EmptyInputValidator,
    }
    
    def __init__(self, config: dict, llm=None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.validators = []
        self.input_validators = []
        self.output_validators = []
        self.custom_validators = {}
        self.llm = llm
        self._load_validators()

    def _load_validators(self):
        """
        Loads enabled validators based on configuration.
        """
        validator_configs = self.config.get("validators", {})
        
        if validator_configs.get("prompt_injection", {}).get("enabled", False):
            from src.core.guardrails.prompt_injection import PromptInjectionValidator
            self.validators.append(PromptInjectionValidator(validator_configs["prompt_injection"], llm=self.llm))
            
        if validator_configs.get("input_length", {}).get("enabled", False):
            from src.core.guardrails.input_length import InputLengthValidator
            self.validators.append(InputLengthValidator(validator_configs["input_length"], llm=self.llm))
            
        if validator_configs.get("special_characters", {}).get("enabled", False):
            from src.core.guardrails.special_characters import SpecialCharactersValidator
            self.validators.append(SpecialCharactersValidator(validator_configs["special_characters"], llm=self.llm))
            
        if validator_configs.get("pii_detector", {}).get("enabled", False):
            from src.core.guardrails.pii_detector import PIIDetectorValidator
            self.validators.append(PIIDetectorValidator(validator_configs["pii_detector"], llm=self.llm))

        if validator_configs.get("empty_input", {}).get("enabled", False):
            from src.core.guardrails.empty_input import EmptyInputValidator
            self.validators.append(EmptyInputValidator(validator_configs["empty_input"], llm=self.llm))
            
        if validator_configs.get("topic_restriction", {}).get("enabled", False):
            from src.core.guardrails.topic_validator import TopicRestrictionValidator
            self.validators.append(TopicRestrictionValidator(validator_configs["topic_restriction"], llm=self.llm))

        if validator_configs.get("toxicity_filter", {}).get("enabled", False):
            from src.core.guardrails.toxicity_validator import ToxicityValidator
            self.validators.append(ToxicityValidator(validator_configs["toxicity_filter"], llm=self.llm))

        builtin_keys = ["prompt_injection", "input_length", "special_characters", "pii_detector", "empty_input", "topic_restriction", "toxicity_filter"]
        
        for key, conf in validator_configs.items():
            if key not in builtin_keys and conf.get("enabled", False) and "module_path" in conf:
                try:
                    from src.utils.class_loader import instantiate_class
                    validator = instantiate_class(conf["module_path"], conf["class_name"], config=conf, llm=self.llm)
                    self.validators.append(validator)
                    
                    test_result = validator.validate("test", "input")
                    if "N/A" not in test_result["reason"]:
                        self.input_validators.append(validator)
                    
                    test_result = validator.validate("test", "output")
                    if "N/A" not in test_result["reason"]:
                        self.output_validators.append(validator)
                        
                except Exception as e:
                    print(f"⚠️  Failed to load custom validator '{key}': {e}")
                    continue
    
    def register_custom_validator(self, name: str, validator_class: type):
        """
        Register a custom validator.
        
        Args:
            name: Name of the validator
            validator_class: Class that inherits from BaseValidator
        """
        if not issubclass(validator_class, BaseValidator):
            raise ValueError(f"Validator {name} must inherit from BaseValidator")
        
        self.custom_validators[name] = validator_class
        
        self.input_validators.clear()
        self.output_validators.clear()
        self._initialize_validators()
    
    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input using all enabled input validators.
        
        Args:
            user_input: The user's input query
            
        Returns:
            Dict with 'valid' (bool), 'reason' (str), and 'sanitized_input' (str)
        """
        if not self.enabled:
            return {
                "valid": True,
                "reason": "Guardrails disabled",
                "sanitized_input": user_input
            }
        
        sanitized = user_input
        
        for validator in self.input_validators:
            if not validator.is_enabled():
                continue
            
            result = validator.validate(sanitized, "input")
            
            if not result["valid"]:
                return {
                    "valid": False,
                    "reason": f"[{validator.get_name()}] {result['reason']}",
                    "sanitized_input": result["sanitized_text"]
                }
            
            sanitized = result["sanitized_text"]
        
        return {
            "valid": True,
            "reason": "All input validations passed",
            "sanitized_input": sanitized
        }
    
    def validate_output(self, output: str) -> Dict[str, Any]:
        """
        Validate LLM output using all enabled output validators.
        
        Args:
            output: The LLM's generated output
            
        Returns:
            Dict with 'valid' (bool), 'reason' (str), and 'sanitized_output' (str)
        """
        if not self.enabled:
            return {
                "valid": True,
                "reason": "Guardrails disabled",
                "sanitized_output": output
            }
        
        sanitized = output
        warnings = []
        
        for validator in self.output_validators:
            if not validator.is_enabled():
                continue
            
            result = validator.validate(sanitized, "output")
            
            if not result["valid"]:
                warnings.append(f"[{validator.get_name()}] {result['reason']}")
                sanitized = result["sanitized_text"]
        
        if warnings:
            return {
                "valid": False,
                "reason": "; ".join(warnings),
                "sanitized_output": sanitized
            }
        
        return {
            "valid": True,
            "reason": "All output validations passed",
            "sanitized_output": output
        }
    
    def get_safe_response(self, validation_result: Dict[str, Any]) -> str:
        """
        Generate a safe response when validation fails.
        
        Args:
            validation_result: The validation result dictionary
            
        Returns:
            A safe response string to return to the user
        """
        return (
            "I apologize, but I cannot process your request due to security concerns. "
            "Please rephrase your question and try again."
        )
    
    def list_validators(self) -> List[Dict[str, str]]:
        """
        List all active validators.
        
        Returns:
            List of dicts with validator info
        """
        validators_info = []
        
        for validator in self.input_validators:
            validators_info.append({
                "name": validator.get_name(),
                "description": validator.get_description(),
                "type": "input",
                "enabled": validator.is_enabled()
            })
        
        for validator in self.output_validators:
            # Avoid duplicates if validator handles both input and output
            if validator not in self.input_validators:
                validators_info.append({
                    "name": validator.get_name(),
                    "description": validator.get_description(),
                    "type": "output",
                    "enabled": validator.is_enabled()
                })
        
        return validators_info
