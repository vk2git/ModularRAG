from typing import Dict, Any, List
from src.core.guardrails.base_validator import BaseValidator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class TopicRestrictionValidator(BaseValidator):
    def get_name(self) -> str:
        return "Topic Restriction"

    def get_description(self) -> str:
        return "Ensures the user query is relevant to the allowed topics."

    def validate(self, text: str, validation_type: str = "input") -> Dict[str, Any]:
        if validation_type != "input":
            return {"valid": True, "sanitized_text": text}

        if not self.llm:
            print("⚠️  TopicRestrictionValidator: No LLM provided. Skipping validation.")
            return {"valid": True, "sanitized_text": text}

        allowed_topics = self.config.get("allowed_topics", [])
        if not allowed_topics:
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
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({"topics": topics_str, "query": text}).strip().upper()
            
            if "NO" in result:
                return {
                    "valid": False,
                    "reason": f"Query is off-topic. Allowed topics: {topics_str}",
                    "sanitized_text": text
                }
            
            return {"valid": True, "sanitized_text": text}
            
        except Exception as e:
            print(f"⚠️  Topic validation failed (LLM error): {e}")
            return {"valid": True, "sanitized_text": text}
