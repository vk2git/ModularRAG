from typing import Dict, Any
from src.core.guardrails.base_validator import BaseValidator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ToxicityValidator(BaseValidator):
    def get_name(self) -> str:
        return "Toxicity Filter"

    def get_description(self) -> str:
        return "Checks for toxic, hateful, or harmful content."

    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if not self.llm:
            print("⚠️  ToxicityValidator: No LLM provided. Skipping validation.")
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
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({"text": text}).strip().upper()
            
            if "YES" in result:
                return {
                    "valid": False,
                    "reason": "Content detected as toxic or harmful.",
                    "sanitized_text": "[REDACTED]" if validation_type == "output" else text
                }
            
            return {"valid": True, "sanitized_text": text}
            
        except Exception as e:
            print(f"⚠️  Toxicity validation failed (LLM error): {e}")
            return {"valid": True, "sanitized_text": text}
