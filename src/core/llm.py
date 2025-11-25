from src.utils.config_loader import load_config
from src.utils.class_loader import instantiate_class
import os

class LLMFactory:
    def __init__(self):
        self.config = load_config()
        self.llm_config = self.config.get("llm", {})

    def create_llm(self):
        """
        Returns a Langchain Chat Model based on config settings.
        """
        mode = self.llm_config.get("mode", "local").lower()

        if mode == "local":
            return self._create_local_llm()
        elif mode == "cloud":
            return self._create_cloud_llm()
        elif mode == "custom":
            return self._create_custom_llm()
        else:
            raise ValueError(f"Unsupported LLM mode: {mode}")
    
    def _create_local_llm(self):
        """
        Creates a local llm instance (Ollama) optimized for CPU
        """
        from langchain_ollama import ChatOllama
        local_conf = self.llm_config.get("local",{})
        print(f"Initializing local llm: {local_conf.get('provider')}, {local_conf.get('model_name')}")

        return ChatOllama(
            model = local_conf.get("model_name", "mistral"),
            base_url=local_conf.get("base_url", "https://localhost:11434"),
            temperature=0,
        )
    
    def _create_cloud_llm(self):
        """
        Create a cloud llm instance (OpenAI, Google, Anthropic)
        """

        cloud_conf = self.llm_config.get("cloud", {})
        provider = cloud_conf.get("provider").lower()
        model_name = cloud_conf.get("model_name")

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not found in environment variables")

            print(f"Initializing cloud LLM: Open AI ({cloud_conf.get('model_name')})")
            return ChatOpenAI(
                model=model_name or "gpt-4o",
                temperature=0,
                api_key=api_key
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not found in environment variables")
            
            print(f"Initializing cloud LLM: Google ({cloud_conf.get('model_name')})")
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.5-flash",
                temperature=0,
                google_api_key=api_key,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not found in environment variables"
                )

            print(f"Initializing cloud LLM: Anthropic ({cloud_conf.get('model_name')})")
            return ChatAnthropic(
                model=model_name or "claude-3-opus-20240229",
                temperature=0,
                anthropic_api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported Cloud Provider: {provider}")

    def _create_custom_llm(self):
        """
        Creates a custom LLM instance from a plugin.
        """
        custom_conf = self.llm_config.get("custom", {})
        module_path = custom_conf.get("module_path")
        class_name = custom_conf.get("class_name")
        kwargs = custom_conf.get("kwargs", {})
        
        if not module_path or not class_name:
            raise ValueError("Custom LLM config requires 'module_path' and 'class_name'")
            
        print(f"Initializing Custom LLM: {class_name} from {module_path}")
        return instantiate_class(module_path, class_name, **kwargs)