from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseValidator(ABC):
    """
    Abstract base class for all guardrail validators.
    
    Each validator is responsible for one specific type of validation
    and can be independently enabled/disabled via configuration.
    """
    
    def __init__(self, config: Dict[str, Any], llm=None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: Configuration dictionary for this validator
            llm: Optional LLM instance for semantic validation
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.llm = llm
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of this validator.
        
        Returns:
            String identifier for this validator
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Return a description of what this validator checks.
        
        Returns:
            Human-readable description
        """
        pass
    
    @abstractmethod
    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        """
        Validate the given text.
        
        Args:
            text: The text to validate (user input or LLM output)
            validation_type: "input" or "output"
            
        Returns:
            Dict with:
                - valid (bool): Whether validation passed
                - reason (str): Reason for failure (if applicable)
                - sanitized_text (str): Sanitized version of the text
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if this validator is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return self.enabled
